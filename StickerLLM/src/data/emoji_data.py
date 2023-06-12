import os
import logging
from pathlib import Path
import json
from PIL import Image, ImageSequence
import base64
from io import BytesIO
from dataclasses import dataclass

import sys
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
import hydra
from transformers.utils import logging
from typing import Any

import pyrootutils

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

logger  = logging.get_logger(__name__)

TXT_RETRIEVAL_PROMPT_TEMPLATE = [
    '根据以下文本检索表情包：{text}',
    '查找表情包：{text}',
    '基于描述文本检索表情图像：{text}',
    '分享一些表情包：{text}',
    '检索表情包：{text}'
]
IMG_RETRIEVAL_PROMPT_TEMPLATE = [
    '{image}根据图像查找表情包',
    '{image}查找表情包',
    '{image}相似的表情图像',
    '{image}检索相似的表情包'
]

IMGTXT_RETRIEVAL_PROMPT_TEMPLATE = [
    '{image}根据图像查找表情包:{text}',
    '{image}查找表情包:{text}',
    '{image}结合图像与文本查找表情图：{text}',
    '{image}根据语义检索表情：{text}'
]


RETRIEVAL_ANSWER_TEMPLATE = [
    '以下是一些检索结果：\n[RET][/RET]\n',
    '\n[RET][/RET]\n',
    '检索结果如下：\n[RET][/RET]\n',
    '查找结果如下：\n[RET][/RET]\n'
]


CHATGLM_INPUT_TEMPLETE = '问：{question}\n答：[gMASK]<sop>{answer}<eop>'


class LMDBEmojiDataset(Dataset):
    def __init__(self,root_path,tokenizer,feat_path,split='train',ratio_t2i=0.5,ratio_i2i=0.25,max_txt_len=128,prefix_ratio=0.5) -> None:
        super().__init__()

        self.max_txt_len = max_txt_len
        self.tokenizer = tokenizer
        self.ratio_t2i = ratio_t2i
        self.ratio_i2i = ratio_i2i
        self.ratio_it2i = 1 - ratio_i2i - ratio_t2i
        self.prefix_ratio = prefix_ratio
        assert self.ratio_it2i > 0, f'ratio_it2i {self.ratio_it2i} must > 0!'

        info_path = os.path.join(root_path,'infos')
        assert os.path.exists(info_path), f'{info_path} is not exist.'
        imgfeat_path = os.path.join(feat_path,'imgs')


        if split == 'train':
            file_path = os.path.join(root_path,'train.txt')
        elif split == 'val':
            file_path = os.path.join(root_path,'valid.txt')
        else:
            raise ValueError(f'Wrong split: {split}!')
        with open(file_path,'r') as f:
            lines = list(f.readlines())
            self.img_name_list = [line.rstrip('\n') for line in lines]

        self.dataset_len = len(self.img_name_list)

        self.env_infos = lmdb.open(info_path, readonly=True, create=False, lock=False, readahead=False, meminit=False)
        self.env_imgfeats = lmdb.open(imgfeat_path, readonly=True, create=False, lock=False, readahead=False, meminit=False)

        self.txn_infos =  self.env_infos.begin(buffers=True)
        self.txn_imgfeats = self.env_imgfeats.begin(buffers=True)

        logger.info(f'num_samles: {self.dataset_len}')

    def __del__(self):
        if hasattr(self,'env_infos'):
            self.env_infos.close()
        if hasattr(self,'env_imgfeats'):
            self.env_imgfeats.close()

    def __len__(self):
        return self.dataset_len
    
    def get_text(self,index):
        img_name = self.img_name_list[index]
        img_info = pickle.loads(self.txn_infos.get(img_name.encode('utf-8')))

        text_info = [
            img_info['caption'],
            img_info['emotion_class'],
            img_info['emotion_label'],
            img_info['style_class'],
            img_info['ocr_text'][:24],
        ]
        random.shuffle(text_info)
        text_info = list(filter(lambda s: s.strip() != '其他类型' and s.strip() != '其他/中性',text_info))
        raw_text = '；'.join(text_info)
        return raw_text
    
    def get_imgfeat(self,index):
        img_name = self.img_name_list[index]
        imgfeat = pickle.loads(self.txn_imgfeats.get(img_name.encode('utf-8')))
        return imgfeat
    

    def __getitem__(self, index) -> Any:
        
        random_index = np.random.randint(len(self))

        random_value = np.random.uniform(0,1)

        try:
            if random_value < self.ratio_t2i:
                text = self.get_text(index)
                imgfeat = self.get_imgfeat(index)
                input_imgfeat = imgfeat

                question = random.choice(TXT_RETRIEVAL_PROMPT_TEMPLATE).format_map({'text': text})
                answer = random.choice(RETRIEVAL_ANSWER_TEMPLATE)
                img_isuse_flag = torch.tensor([False])

            elif random_value < self.ratio_t2i + self.ratio_i2i:
                imgfeat = self.get_imgfeat(index)
                input_imgfeat = imgfeat

                question = random.choice(IMG_RETRIEVAL_PROMPT_TEMPLATE).format_map({'image':'[IMG][IMG][/IMG]'})
                answer = random.choice(RETRIEVAL_ANSWER_TEMPLATE)
                img_isuse_flag = torch.tensor([True])

            else:
                text = self.get_text(index)
                imgfeat = self.get_imgfeat(index)
                input_imgfeat = self.get_imgfeat(random_index)
                question = random.choice(IMGTXT_RETRIEVAL_PROMPT_TEMPLATE).format_map({'image':'[IMG][IMG][/IMG]','text':text})
                answer = random.choice(RETRIEVAL_ANSWER_TEMPLATE)
                img_isuse_flag = torch.tensor([True])
        except TypeError as e:
            print(f'Error occured in data process: {e}')
            return None


        input_text = CHATGLM_INPUT_TEMPLETE.format_map({'question':question,'answer':answer})
        prefix_flag = np.random.uniform(0,1) < self.prefix_ratio

        if prefix_flag:
            input_text = '[PRET] ' + input_text

        token_ids = self.tokenizer(input_text,
                                   return_tensors="pt",
                                    padding='max_length',
                                    truncation=True,
                                    add_special_tokens=False,
                                    max_length= self.max_txt_len).input_ids[0]



        try:
            img_token_idx = torch.where(token_ids == self.tokenizer.imgend_token_id)[0] - 1
            ret_token_idx = torch.where(token_ids == self.tokenizer.ret_token_id)[0]

            if len(ret_token_idx) == 0:
                token_ids[-2] = self.tokenizer.ret_token_id
                token_ids[-1] =self.tokenizer.retend_token_id
                ret_token_idx = torch.where(token_ids == self.tokenizer.ret_token_id)[0]
            
            ret_token_idx = ret_token_idx[-1]

            img_token_flag = torch.zeros_like(token_ids,dtype=torch.bool)
            for idx in img_token_idx:
                img_token_flag[idx:idx+1] = True

            mask_idx = torch.where(token_ids == 150001)[0][-1]
            label_mask = torch.zeros_like(token_ids,dtype=torch.bool)
            label_mask[token_ids==self.tokenizer.pad_token_id] = True
            label_mask[ret_token_idx + 1] = True
            label_mask[:mask_idx] = True

            ids_list = token_ids.tolist()
            attention_mask = self.get_attention_masks(seq=ids_list)
            position_ids = self.get_position_ids(seq=ids_list)

        except (IndexError,ValueError) as e:
            print(f'Error occured in data process: {e}')
            return None

        return {
            'img_feats': imgfeat,
            'input_text': input_text,
            'input_imgfeats': input_imgfeat,
            'token_ids': token_ids,
            'img_token_flag': img_token_flag,
            'img_isuse_flag': img_isuse_flag,
            'ret_token_idx': ret_token_idx,
            'label_mask': label_mask,
            'attention_mask': attention_mask,
            'position_ids': position_ids,
        }
    
    def get_attention_masks(self, seq, device=None):
        context_length = seq.index(150004) + 1

        pad_mask = torch.tensor(seq, dtype=torch.long, device=device) == self.tokenizer.pad_token_id

        attention_mask = torch.ones((1, len(seq), len(seq)), device=device)
        attention_mask.tril_()
        attention_mask[..., :context_length - 1] = 1
        # attention_mask.unsqueeze_(1)
        attention_mask = (attention_mask < 0.5).bool()

        attention_mask[..., pad_mask] = True

        # attn_img = Image.fromarray((attention_mask[0].to(dtype=torch.int).cpu().numpy() * 255).astype(np.uint8))
        # attn_img.show()

        return attention_mask

    def get_position_ids(self, seq, device=None):

        mask_pos = seq.index(150001)
        bos_pos = seq.index(150004)

        position_ids = torch.arange(len(seq), dtype=torch.long, device=device)
        position_ids[bos_pos:] = position_ids[mask_pos]

        block_position_ids = torch.zeros_like(position_ids)
        block_position_ids[bos_pos:] = torch.arange(len(seq) - bos_pos, dtype=torch.long, device=device) + 1

        position_ids = torch.stack((position_ids, block_position_ids), dim=0)
        return position_ids
    

class LMDBEmojiEvalDataset(LMDBEmojiDataset):
    def __init__(self,root_path,tokenizer,feat_path,split='val',max_txt_len=128,text2image=True) -> None:
        super().__init__(root_path,tokenizer,feat_path,split,max_txt_len=max_txt_len)
        self.text2image = text2image

    
    
    def get_text(self,index):
        img_name = self.img_name_list[index]
        img_info = pickle.loads(self.txn_infos.get(img_name.encode('utf-8')))

        text_info = [
            img_info['caption'],
            img_info['emotion_class'],
            img_info['emotion_label'],
            img_info['style_class'],
            img_info['ocr_text'][:24],
        ]
        # random.shuffle(text_info)
        text_info = list(filter(lambda s: s.strip() != '其他类型' and s.strip() != '其他/中性',text_info))
        raw_text = '；'.join(text_info)
        return raw_text

    def __getitem__(self, index) -> Any:

        try:
            img_name = self.img_name_list[index]
            if self.text2image:
                text = self.get_text(index)
                imgfeat = self.get_imgfeat(index)
                input_imgfeat = imgfeat

                question = '根据以下文本检索表情包：{text}'.format_map({'text':text})
                answer = '\n[RET][/RET]\n'
                img_isuse_flag = torch.tensor([False])
            else:

                imgfeat = self.get_imgfeat(index)
                input_imgfeat = imgfeat

                question = '{image}根据图像查找表情包'.format_map({'image':'[IMG][IMG][/IMG]'})
                answer = '\n[RET][/RET]\n'
                img_isuse_flag = torch.tensor([True]) 
        
        except TypeError as e:
            print(f'Error occured in data process: {e}')
            return None

        input_text = CHATGLM_INPUT_TEMPLETE.format_map({'question':question,'answer':answer})

        token_ids = self.tokenizer(input_text,
                                   return_tensors="pt",
                                    padding='max_length',
                                    truncation=True,
                                    add_special_tokens=False,
                                    max_length= self.max_txt_len).input_ids[0]

        try:
            img_token_idx = torch.where(token_ids == self.tokenizer.imgend_token_id)[0] - 1
            ret_token_idx = torch.where(token_ids == self.tokenizer.ret_token_id)[0]

            if len(ret_token_idx) == 0:
                token_ids[-2] = self.tokenizer.ret_token_id
                token_ids[-1] =self.tokenizer.retend_token_id
                ret_token_idx = torch.where(token_ids == self.tokenizer.ret_token_id)[0]
            
            ret_token_idx = ret_token_idx[-1]

            img_token_flag = torch.zeros_like(token_ids,dtype=torch.bool)
            for idx in img_token_idx:
                img_token_flag[idx:idx+1] = True

            mask_idx = torch.where(token_ids == 150001)[0][-1]
            label_mask = torch.zeros_like(token_ids,dtype=torch.bool)
            label_mask[token_ids==self.tokenizer.pad_token_id] = True
            label_mask[ret_token_idx + 1] = True
            label_mask[:mask_idx] = True

            ids_list = token_ids.tolist()
            attention_mask = self.get_attention_masks(seq=ids_list)
            position_ids = self.get_position_ids(seq=ids_list)

        except (IndexError,ValueError) as e:
            print(f'Error occured in data process: {e}')
            return None

        return {
            'img_feats': imgfeat,
            'input_text': input_text,
            'input_imgfeats': input_imgfeat,
            'token_ids': token_ids,
            'img_token_flag': img_token_flag,
            'img_isuse_flag': img_isuse_flag,
            'ret_token_idx': ret_token_idx,
            'label_mask': label_mask,
            'attention_mask': attention_mask,
            'position_ids': position_ids,
            'img_name': img_name
        }
    