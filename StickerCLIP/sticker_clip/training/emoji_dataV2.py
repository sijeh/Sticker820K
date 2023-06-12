from math import ceil
import os
import logging
from pathlib import Path
import json
from PIL import Image
import base64
from io import BytesIO
from dataclasses import dataclass

import lmdb
import pickle

import numpy as np
import random

import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import SequentialSampler

from torchvision.transforms import Compose, Resize, ToTensor, Normalize, InterpolationMode
from timm.data import create_transform

from cn_clip.clip import _tokenizer
from cn_clip.clip import tokenize


def _convert_to_rgb(image):
    return image.convert('RGB')


def _preprocess_text(text):
    # adapt the text to Chinese BERT vocab
    text = text.lower().replace("“", "\"").replace("”", "\"")
    return text

TEXT_TEMPLATE ="{caption}；{emotion_class},{emotion_label}；{style_class}；{ocr_text}。"


class EmojiLMDBDataset(Dataset):
    def __init__(self, lmdb_path, split="val", max_txt_length=64, use_augment=False, resolution=224):

        super().__init__()

        self.lmdb_path = lmdb_path
        # assert LMDB directories exist
        assert os.path.isdir(lmdb_path), "The LMDB directory {} of {} split does not exist!".format(lmdb_path, split)

        lmdb_pairs = os.path.join(lmdb_path, "infos")
        assert os.path.isdir(lmdb_pairs), "The LMDB directory {} of {} image-text infos does not exist!".format(lmdb_pairs, split)

        lmdb_imgs = os.path.join(lmdb_path, "imgs")
        assert os.path.isdir(lmdb_imgs), "The LMDB directory {} of {} image base64 strings does not exist!".format(lmdb_imgs, split)

        # open LMDB files
        self.env_pairs = lmdb.open(lmdb_pairs, readonly=True, create=False, lock=False, readahead=False, meminit=False)
        self.txn_pairs = self.env_pairs.begin(buffers=True)
        self.env_imgs = lmdb.open(lmdb_imgs, readonly=True, create=False, lock=False, readahead=False, meminit=False)
        self.txn_imgs = self.env_imgs.begin(buffers=True)

        if split == 'train':
            file_path = os.path.join(lmdb_path,'train.txt')
        elif split == 'val':
            file_path = os.path.join(lmdb_path,'valid.txt')
        else:
            raise ValueError(f'Wrong split: {split}!')

        with open(file_path,'r') as f:
            lines = list(f.readlines())
            self.img_name_list = [line.rstrip('\n') for line in lines]

        print('Total samples: ',len(self.img_name_list))
        self.number_samples = len(self.img_name_list)
        self.dataset_len = len(self.img_name_list)

        # fetch number of pairs and images
        # self.number_samples = int(self.txn_pairs.get(key=b'num_samples').tobytes().decode('utf-8'))
        # self.number_images = int(self.txn_imgs.get(key=b'num_samples').tobytes().decode('utf-8'))
        # logging.info("{} LMDB file contains {} images and {} pairs.".format(split, self.number_images, self.number_samples))

        # the self.dataset_len will be edited to a larger value by calling pad_dataset()
        self.global_batch_size = 1 # will be modified to the exact global_batch_size after calling pad_dataset()

        self.split = split
        self.max_txt_length = max_txt_length        

        self.use_augment = use_augment
        self.transform = self._build_transform(resolution)

    def _build_transform(self, resolution):
        if self.split == "train" and self.use_augment:
            transform = create_transform(
                             input_size=resolution,
                             scale=(0.9, 1.0),
                             is_training=True,
                             color_jitter=None,
                             auto_augment='original',
                             interpolation='bicubic',
                             mean=(0.48145466, 0.4578275, 0.40821073),
                             std=(0.26862954, 0.26130258, 0.27577711),
                         )
            transform = Compose(transform.transforms[:-3] + [_convert_to_rgb] + transform.transforms[-3:])
        else:
            transform = Compose([
                Resize((resolution, resolution), interpolation=InterpolationMode.BICUBIC),
                _convert_to_rgb,
                ToTensor(),
                Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
            ])
        return transform

    def __del__(self):
        if hasattr(self, 'env_pairs'):
            self.env_pairs.close()
        if hasattr(self, 'env_imgs'):
            self.env_imgs.close()

    def __len__(self):
        return self.dataset_len

    def get_text_from_info(self,info):
        # raw_text = TEXT_TEMPLATE.format_map({
        #     'caption':info['caption'],
        #     'emotion_class':info['emotion_class'],
        #     'emotion_label':info['emotion_label'],
        #     'style_class':info['style_class'],
        #     'ocr_text':info['ocr_text'][:20],
        # })

        text_info = [
            info['caption'],
            info['emotion_class'],
            info['emotion_label'],
            info['style_class'],
            info['ocr_text'][:24],
        ]
        random.shuffle(text_info)
        raw_text = '；'.join(text_info)
        return raw_text

    def get_tensor_from_img(self,img):
        frames = []
        if hasattr(img,'n_frames'):
            start = 0
            end = img.n_frames - 1
            middle = end // 2
            frames = []
            for idx in [start, middle, end]:
                img.seek(idx)
                frame = img.convert()
                frames.append(self.transform(frame))
            img_tensor = torch.cat(frames,dim=0)
        else:
            frame = self.transform(img)
            img_tensor = torch.cat([frame, frame, frame], dim=0)
        
        return img_tensor
        

    def __getitem__(self, index):
        sample_index = index % self.number_samples


        try:
            img_name = self.img_name_list[sample_index]

            # img_info = pickle.loads(img_name.encode('utf-8'))
            img_info = pickle.loads(self.txn_pairs.get(img_name.encode('utf-8')))

            img_bytes = self.txn_imgs.get(img_name.encode('utf-8'))
            img = Image.open(BytesIO(img_bytes))
            img_tensor = self.get_tensor_from_img(img)
            raw_text = self.get_text_from_info(img_info)
        except (OSError, TypeError):
            print('Error occured in dataset')
            return None

        text_ids = tokenize([_preprocess_text(raw_text)], context_length=self.max_txt_length)[0]
        eos_index = text_ids.numpy().tolist().index(_tokenizer.vocab['[SEP]'])
        return img_tensor, text_ids, eos_index



class EmojiDemoDataset(EmojiLMDBDataset):
    def __init__(self, lmdb_path, split="val", max_txt_length=64, use_augment=False, resolution=224):
        super(EmojiDemoDataset, self).__init__(lmdb_path, split, max_txt_length, use_augment, resolution)

    def get_text_from_info(self,info):
        raw_text = info['ocr_text'][:50]
        return raw_text
        

    def __getitem__(self, index):
        sample_index = index % self.number_samples


        try:
            img_name = self.img_name_list[sample_index]
            # img_info = pickle.loads(img_name.encode('utf-8'))
            img_info = pickle.loads(self.txn_pairs.get(img_name.encode('utf-8')))

            img_bytes = self.txn_imgs.get(img_name.encode('utf-8'))
            img = Image.open(BytesIO(img_bytes))
            img_tensor = self.get_tensor_from_img(img)
            raw_text = self.get_text_from_info(img_info)
        except (OSError, TypeError):
            print('Error occured in dataset')
            return None

        text_ids = tokenize([_preprocess_text(raw_text)], context_length=self.max_txt_length)[0]
        eos_index = text_ids.numpy().tolist().index(_tokenizer.vocab['[SEP]'])
        return img_tensor, text_ids, eos_index, img_name,raw_text

class EmojiEvalDataset(EmojiLMDBDataset):
    def __init__(self, lmdb_path, split="val", max_txt_length=64, use_augment=False, resolution=224):
        super().__init__(lmdb_path, split, max_txt_length, use_augment, resolution)

    def get_text_from_info(self,info):

        text_info = [
            info['caption'],
            info['emotion_class'],
            info['emotion_label'],
            info['style_class'],
            info['ocr_text'][:24],
        ]
        raw_text = '；'.join(text_info)
        return raw_text

    def __getitem__(self, index):
        sample_index = index % self.number_samples

        try:
            img_name = self.img_name_list[sample_index]
            img_info = pickle.loads(self.txn_pairs.get(img_name.encode('utf-8')))

            img_bytes = self.txn_imgs.get(img_name.encode('utf-8'))
            img = Image.open(BytesIO(img_bytes))
            img_tensor = self.get_tensor_from_img(img)
            raw_text = self.get_text_from_info(img_info)
        except (OSError, TypeError):
            print('Error occured in dataset')
            return None

        text_ids = tokenize([_preprocess_text(raw_text)], context_length=self.max_txt_length)[0]
        eos_index = text_ids.numpy().tolist().index(_tokenizer.vocab['[SEP]'])
        return img_tensor, text_ids, eos_index, img_name,raw_text



def pad_dataset(dataset, global_batch_size):
    # edit dataset.__len__() of the dataset
    dataset.dataset_len = ceil(dataset.dataset_len / global_batch_size) * global_batch_size
    dataset.global_batch_size = global_batch_size


def fetch_resolution(vision_model):
    # fetch the resolution from the vision model config
    vision_model_config_file = Path(__file__).parent.parent / f"clip/model_configs/{vision_model.replace('/', '-')}.json"
    with open(vision_model_config_file, 'r') as fv:
        model_info = json.load(fv)
    return model_info["image_resolution"]

def collate_fn(batch):
    # 过滤掉出错的数据
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)

@dataclass
class DataInfo:
    dataloader: DataLoader
    sampler: DistributedSampler
    dataset: EmojiLMDBDataset
    epoch_id: int


def get_dataset(args, is_train, max_txt_length=64, epoch_id=0):
    if is_train:
        db_path = args.train_data
    else:
        db_path = args.val_data
    assert db_path is not None

    dataset = EmojiLMDBDataset(
        db_path, 
        split="train" if is_train else "val",
        max_txt_length=max_txt_length,
        use_augment=args.use_augment if is_train else False,
        resolution=fetch_resolution(args.vision_model),
    ) 

    # pad the dataset splits using the beginning samples in the LMDB files
    # to make the number of samples enough for a full final global batch
    batch_size = args.batch_size if is_train else args.valid_batch_size
    global_batch_size = batch_size * torch.distributed.get_world_size()
    pad_dataset(dataset, global_batch_size)

    num_samples = dataset.dataset_len
    # Update in 22.12.11: We have changed the **validation** dataset sampler during finetuning
    # from sequential to shuffled (in a determistic order between experiments and epochs). 
    # This is to avoid there being one text matching multiple images (or vice versa) in a local batch
    # which will affect the correctness of computing the validation in-batch accuracy.
    sampler = DistributedSampler(dataset, shuffle=True, seed=args.seed)
    sampler.set_epoch(epoch_id if is_train else 0)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=False,
        num_workers=args.num_workers if is_train else 1,
        sampler=sampler,
        collate_fn=collate_fn,
    )

    dataloader.num_samples = num_samples
    assert num_samples % dataset.global_batch_size == 0
    dataloader.num_batches = num_samples // dataset.global_batch_size

    return DataInfo(dataloader, sampler, dataset, epoch_id)


def get_data(args, epoch_id=0, max_txt_length=64):
    data = {}

    if args.train_data:
        data["train"] = get_dataset(
            args, 
            is_train=True,  
            max_txt_length=max_txt_length, 
            epoch_id=epoch_id)

    if args.val_data:
        data["valid"] = get_dataset(
            args, 
            is_train=False, 
            max_txt_length=max_txt_length, 
            epoch_id=epoch_id)

    return data


def get_demo_data(db_path, splits='val', batch_size=128, max_txt_length=64, num_workers=8, vision_model_type='RN50'):
    dataset = EmojiDemoDataset(db_path,
                                split=splits,
                                max_txt_length=max_txt_length,
                                use_augment=False,
                                resolution=fetch_resolution(vision_model_type))
    sampler = SequentialSampler(dataset)
    dataloader = DataLoader(dataset,
                            batch_size=batch_size,
                            pin_memory=False,
                            num_workers=num_workers,
                            sampler=sampler,
                            drop_last=False,
                            collate_fn=collate_fn,)
    return DataInfo(dataloader, sampler, dataset, 0)


def get_eval_data(db_path, splits='val', batch_size=128, max_txt_length=64, num_workers=8, vision_model_type='RN50'):
    dataset = EmojiEvalDataset(db_path,
                                split=splits,
                                max_txt_length=max_txt_length,
                                use_augment=False,
                                resolution=fetch_resolution(vision_model_type))
    sampler = SequentialSampler(dataset)
    dataloader = DataLoader(dataset,
                            batch_size=batch_size,
                            pin_memory=False,
                            num_workers=num_workers,
                            sampler=sampler,
                            drop_last=False,
                            collate_fn=collate_fn,)
    return DataInfo(dataloader, sampler, dataset, 0)

