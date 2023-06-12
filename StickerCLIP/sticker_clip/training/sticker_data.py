from math import ceil
import os
import logging
from pathlib import Path
import json
from PIL import Image, ImageSequence
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
import random
import numpy as np


def _convert_to_rgb(image):
    return image.convert('RGB')


def _preprocess_text(text):
    # adapt the text to Chinese BERT vocab
    text = text.lower().replace("“", "\"").replace("”", "\"")
    return text


class LMDBStickerDataset(Dataset):

    def __init__(self, root_path, splits='sogou&qq', sample_ratio=1.0, max_txt_length=64, use_augment=False, resolution=224):
        '''
        split : choice of ['sogou', 'qq','sogou&qq']
        '''

        assert splits in ['sogou', 'qq', 'sogou&qq'], 'Got wrong data splits: ' + str(splits)
        self.root_path = root_path
        self.splits = splits.split('&')
        self.sample_ratio = sample_ratio

        self.num_samples = {}
        self.env_imgs = {}
        self.env_pairs = {}
        self.txn_imgs = {}
        self.txn_pairs = {}

        self.total_num = 0
        self.start_idx = {}
        self.end_idx = {}

        for idx, split in enumerate(self.splits):

            lmdb_img_path = os.path.join(self.root_path, split, 'imgs')
            lmdb_pair_path = os.path.join(self.root_path, split, 'pairs')

            assert os.path.isdir(lmdb_img_path), split + lmdb_img_path
            assert os.path.isdir(lmdb_pair_path), split + lmdb_pair_path

            self.env_imgs[split] = lmdb.open(lmdb_img_path,
                                             readonly=True,
                                             create=False,
                                             lock=False,
                                             readahead=False,
                                             meminit=False)
            self.env_pairs[split] = lmdb.open(lmdb_pair_path,
                                              readonly=True,
                                              create=False,
                                              lock=False,
                                              readahead=False,
                                              meminit=False)
            self.txn_imgs[split] = self.env_imgs[split].begin(buffers=True)
            self.txn_pairs[split] = self.env_pairs[split].begin(buffers=True)
            num_img = int(self.txn_imgs[split].get(key=b'num_samples').tobytes().decode('utf-8'))
            num_pair = int(self.txn_pairs[split].get(key=b'num_samples').tobytes().decode('utf-8'))
            assert num_img == num_pair, print(num_img, num_pair)

            self.num_samples[split] = int(num_img * sample_ratio)
            self.start_idx[split] = self.total_num
            self.total_num += self.num_samples[split]
            self.end_idx[split] = self.total_num

        logging.info("LMDB file contains {} images and pairs.".format(self.total_num))
        super(LMDBStickerDataset, self).__init__()

        self.dataset_len = self.total_num
        self.global_batch_size = 1  # will be modified to the exact global_batch_size after calling pad_dataset()
        self.max_txt_length = max_txt_length
        self.use_augment = use_augment
        self.transform = self._build_transform(resolution)

    def _build_transform(self, resolution):
        if self.use_augment:
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
            for split in self.splits:
                self.env_pairs[split].close()
        if hasattr(self, 'env_imgs'):
            for split in self.splits:
                self.env_imgs[split].close()

    def __len__(self):
        return self.dataset_len

    def __getitem__(self, index):
        sample_index = index % self.total_num

        split_idx = sample_index
        cur_split = self.splits[0]
        for split in self.splits:
            if sample_index >= self.start_idx[split] and sample_index < self.end_idx[split]:
                cur_split = split
                split_idx = sample_index - self.start_idx[split]
                break

        img_info = pickle.loads(self.txn_pairs[cur_split].get(str(split_idx).encode('utf-8')))
        img_id = img_info['img_id']
        img_texts = img_info['img_text']

        img_texts = list(map(lambda item: random.choice(item) if isinstance(item, list) else item, img_texts))
        img_texts = list(filter(lambda s: s.strip() != "" and s.strip() != "其它", img_texts))
        img_texts = list(set(img_texts))  # 去重
        random.shuffle(img_texts)
        raw_text = '；'.join(img_texts)

        img_bytes = self.txn_imgs[cur_split].get(img_id.encode('utf-8'))
        image = Image.open(BytesIO(base64.urlsafe_b64decode(img_bytes)))

        if img_id.endswith('.gif'):
            frame_tensors = [self.transform(frame) for frame in ImageSequence.Iterator(image)]

            for i in range(len(frame_tensors), 3):
                frame_tensors.append(frame_tensors[-1])
                # print('frames: ', len(frame_tensors), image.n_frames, img_id)
            image_tensor = torch.cat(frame_tensors, dim=0)
        else:
            frame = self.transform(image)
            image_tensor = torch.cat([frame, frame, frame], dim=0)

        text = tokenize([_preprocess_text(raw_text)], context_length=self.max_txt_length)[0]
        eos_index = text.numpy().tolist().index(_tokenizer.vocab['[SEP]'])
        return image_tensor, text, eos_index, img_id, raw_text


class LMDBStickerDemoDataset(Dataset):

    def __init__(self, root_path, splits='sogou&qq', sample_ratio=1.0, max_txt_length=64, use_augment=False, resolution=224):
        '''
        split : choice of ['sogou', 'qq','sogou&qq']
        '''
        self.root_path = root_path
        self.splits = splits.split('&')
        self.sample_ratio = sample_ratio

        self.num_samples = {}
        self.env_imgs = {}
        self.env_pairs = {}
        self.txn_imgs = {}
        self.txn_pairs = {}

        self.total_num = 0
        self.start_idx = {}
        self.end_idx = {}

        for idx, split in enumerate(self.splits):

            lmdb_img_path = os.path.join(self.root_path, split, 'imgs')
            lmdb_pair_path = os.path.join(self.root_path, split, 'pairs')

            assert os.path.isdir(lmdb_img_path), print(split, lmdb_img_path)
            assert os.path.isdir(lmdb_pair_path), print(split, lmdb_pair_path)

            self.env_imgs[split] = lmdb.open(lmdb_img_path,
                                             readonly=True,
                                             create=False,
                                             lock=False,
                                             readahead=False,
                                             meminit=False)
            self.env_pairs[split] = lmdb.open(lmdb_pair_path,
                                              readonly=True,
                                              create=False,
                                              lock=False,
                                              readahead=False,
                                              meminit=False)
            self.txn_imgs[split] = self.env_imgs[split].begin(buffers=True)
            self.txn_pairs[split] = self.env_pairs[split].begin(buffers=True)
            num_img = int(self.txn_imgs[split].get(key=b'num_samples').tobytes().decode('utf-8'))
            num_pair = int(self.txn_pairs[split].get(key=b'num_samples').tobytes().decode('utf-8'))
            assert num_img == num_pair, print(num_img, num_pair)

            self.num_samples[split] = int(num_img * sample_ratio)
            self.start_idx[split] = self.total_num
            self.total_num += self.num_samples[split]
            self.end_idx[split] = self.total_num

        logging.info("LMDB file contains {} images and pairs.".format(self.total_num))
        super(LMDBStickerDemoDataset, self).__init__()

        self.dataset_len = self.total_num
        self.global_batch_size = 1  # will be modified to the exact global_batch_size after calling pad_dataset()
        self.max_txt_length = max_txt_length
        self.use_augment = use_augment
        self.transform = self._build_transform(resolution)

    def _build_transform(self, resolution):

        transform = Compose([
            Resize((resolution, resolution), interpolation=InterpolationMode.BICUBIC),
            _convert_to_rgb,
            ToTensor(),
            Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])
        return transform

    def __del__(self):
        if hasattr(self, 'env_pairs'):
            for split in self.splits:
                self.env_pairs[split].close()
        if hasattr(self, 'env_imgs'):
            for split in self.splits:
                self.env_imgs[split].close()

    def __len__(self):
        return self.dataset_len

    def __getitem__(self, index):
        sample_index = index % self.total_num

        split_idx = sample_index
        cur_split = self.splits[0]
        for split in self.splits:
            if sample_index >= self.start_idx[split] and sample_index < self.end_idx[split]:
                cur_split = split
                split_idx = sample_index - self.start_idx[split]
                break

        img_info = pickle.loads(self.txn_pairs[cur_split].get(str(split_idx).encode('utf-8')))
        img_id = img_info['img_id']
        img_texts = img_info['img_text']

        # img_texts = list(map(lambda item: random.choice(item) if isinstance(item, list) else item, img_texts))
        # img_texts = list(filter(lambda s: s.strip() != "", img_texts))
        # img_texts = list(set(img_texts))  # 去重
        # random.shuffle(img_texts)
        # raw_text = '；'.join(img_texts)
        if cur_split == 'qq':
            raw_text = img_texts[0][0].strip()
        elif cur_split == 'sogou':
            raw_text = img_texts[2].strip()
        else:
            img_texts = list(map(lambda item: random.choice(item) if isinstance(item, list) else item, img_texts))
            img_texts = list(filter(lambda s: s.strip() != "" and s.strip() != "其它", img_texts))
            img_texts = list(set(img_texts))  # 去重
            raw_text = '；'.join(img_texts)

        img_bytes = self.txn_imgs[cur_split].get(img_id.encode('utf-8'))
        image = Image.open(BytesIO(base64.urlsafe_b64decode(img_bytes)))

        if img_id.endswith('.gif'):
            frame_tensors = [self.transform(frame) for frame in ImageSequence.Iterator(image)]

            for i in range(len(frame_tensors), 3):
                frame_tensors.append(frame_tensors[-1])
                # print('frames: ', len(frame_tensors), image.n_frames, img_id)
            image_tensor = torch.cat(frame_tensors, dim=0)
        else:
            frame = self.transform(image)
            image_tensor = torch.cat([frame, frame, frame], dim=0)

        text = tokenize([_preprocess_text(raw_text)], context_length=self.max_txt_length)[0]
        eos_index = text.numpy().tolist().index(_tokenizer.vocab['[SEP]'])
        return image_tensor, text, eos_index, img_id, raw_text


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


@dataclass
class DataInfo:
    dataloader: DataLoader
    sampler: DistributedSampler
    dataset: LMDBStickerDataset
    epoch_id: int


def get_dataset(args, splits, is_train, batch_size, max_txt_length=64, epoch_id=0):
    db_path = args.data_root

    dataset = LMDBStickerDataset(
        db_path,
        splits=splits,
        sample_ratio=1.0 if is_train else 0.1,
        max_txt_length=max_txt_length,
        use_augment=is_train,
        resolution=fetch_resolution(args.vision_model),
    )

    # pad the dataset splits using the beginning samples in the LMDB files
    # to make the number of samples enough for a full final global batch
    global_batch_size = batch_size * torch.distributed.get_world_size()
    pad_dataset(dataset, global_batch_size)

    num_samples = dataset.dataset_len
    # Update in 22.12.11: We have changed the **validation** dataset sampler during finetuning
    # from sequential to shuffled (in a determistic order between experiments and epochs).
    # This is to avoid there being one text matching multiple images (or vice versa) in a local batch
    # which will affect the correctness of computing the validation in-batch accuracy.
    sampler = DistributedSampler(dataset, shuffle=is_train, seed=args.seed)
    sampler.set_epoch(epoch_id if is_train else 0)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=False,
        num_workers=args.num_workers,
        sampler=sampler,
    )

    dataloader.num_samples = num_samples
    assert num_samples % dataset.global_batch_size == 0
    dataloader.num_batches = num_samples // dataset.global_batch_size

    return DataInfo(dataloader, sampler, dataset, epoch_id)


def get_trainval_data(args, splits, epoch_id=0, max_txt_length=64):
    data = {}

    data["train"] = get_dataset(args,
                                splits=splits,
                                is_train=True,
                                batch_size=args.batch_size,
                                max_txt_length=max_txt_length,
                                epoch_id=epoch_id)
    data["valid"] = get_dataset(args,
                                splits=splits,
                                is_train=False,
                                batch_size=args.valid_batch_size,
                                max_txt_length=max_txt_length,
                                epoch_id=epoch_id)

    return data


def get_demo_data(db_path, splits, batch_size=128, max_txt_length=64, num_workers=8, vision_model_type='RN50'):
    dataset = LMDBStickerDemoDataset(db_path,
                                     splits=splits,
                                     max_txt_length=max_txt_length,
                                     use_augment=False,
                                     resolution=fetch_resolution(vision_model_type))
    sampler = SequentialSampler(dataset)
    dataloader = DataLoader(dataset,
                            batch_size=batch_size,
                            pin_memory=False,
                            num_workers=num_workers,
                            sampler=sampler,
                            drop_last=False)
    return DataInfo(dataloader, sampler, dataset, 0)


def get_dataset_temp(args, splits, is_train, batch_size, max_txt_length=64, epoch_id=0):
    db_path = args.data_root

    dataset = LMDBStickerDataset(
        db_path,
        splits=splits,
        sample_ratio=1.0,
        max_txt_length=max_txt_length,
        use_augment=is_train,
        resolution=fetch_resolution(args.vision_model),
    )

    sampler = SequentialSampler(dataset)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=False,
        num_workers=args.num_workers,
        sampler=sampler,
    )

    return DataInfo(dataloader, sampler, dataset, epoch_id)