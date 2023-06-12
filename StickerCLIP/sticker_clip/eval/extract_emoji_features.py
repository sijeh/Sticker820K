# -*- coding: utf-8 -*-
'''
This script extracts image and text features for evaluation. (with single-GPU)
'''

import os
import argparse
import logging
from pathlib import Path
import json
import numpy as np
import pickle

import torch
from tqdm import tqdm

from cn_clip.clip.model import convert_weights, CLIP
from cn_clip.training.main import convert_models_to_fp32
from cn_clip.eval.data import get_eval_img_dataset, get_eval_txt_dataset
from cn_clip.training.emoji_data import get_demo_data


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root',
                        type=str,
                        required=True,
                        help="The lmdb data path.")
    parser.add_argument('--save-dir',
                        type=str,
                        required=True,
                        help="save dir")
    parser.add_argument('--img-id-url',
                        type=str,
                        required=True,
                        help="img id url path")
    parser.add_argument("--batch-size", type=int, default=128, help="Image batch size.")
    parser.add_argument("--context-length",
                        type=int,
                        default=64,
                        help="The maximum length of input text (include [CLS] & [SEP] tokens).")
    parser.add_argument("--resume", default=None, type=str, help="path to latest checkpoint (default: none)")
    parser.add_argument("--precision", choices=["amp", "fp16", "fp32"], default="amp", help="Floating point precition.")
    parser.add_argument("--vision-model",
                        choices=["ViT-B-32", "ViT-B-16", "ViT-L-14", "ViT-L-14-336", "ViT-H-14", "RN50"],
                        default="ViT-B-16",
                        help="Name of the vision backbone to use.")
    parser.add_argument("--text-model",
                        choices=["RoBERTa-wwm-ext-base-chinese", "RoBERTa-wwm-ext-large-chinese", "RBT3-chinese"],
                        default="RoBERTa-wwm-ext-base-chinese",
                        help="Name of the text backbone to use.")
    parser.add_argument("--debug", default=False, action="store_true", help="If true, more information is logged.")
    parser.add_argument("--gpu",
                        type=int,
                        default=0,
                        help="device id")
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()

    # Log params.
    print("Params:")
    for name in sorted(vars(args)):
        val = getattr(args, name)
        print(f"  {name}: {val}")

    # args.gpu = 0
    torch.cuda.set_device(args.gpu)

    # Initialize the model.
    vision_model_config_file = Path(__file__).parent.parent / f"clip/model_configs/{args.vision_model.replace('/', '-')}.json"
    print('Loading vision model config from', vision_model_config_file)
    assert os.path.exists(vision_model_config_file)

    text_model_config_file = Path(__file__).parent.parent / f"clip/model_configs/{args.text_model.replace('/', '-')}.json"
    print('Loading text model config from', text_model_config_file)
    assert os.path.exists(text_model_config_file)

    with open(vision_model_config_file, 'r') as fv, open(text_model_config_file, 'r') as ft:
        model_info = json.load(fv)
        if isinstance(model_info['vision_layers'], str):
            model_info['vision_layers'] = eval(model_info['vision_layers'])
        for k, v in json.load(ft).items():
            model_info[k] = v

    model = CLIP(**model_info)
    convert_weights(model)

    # See https://discuss.pytorch.org/t/valueerror-attemting-to-unscale-fp16-gradients/81372
    if args.precision == "amp" or args.precision == "fp32":
        convert_models_to_fp32(model)
    model.cuda(args.gpu)
    if args.precision == "fp16":
        convert_weights(model)

    model.eval()

    # Get data.
    print("Preparing image inference dataset.")
    # img_data = get_eval_img_dataset(args)
    data = get_demo_data(db_path=args.data_root,
                         splits='val',
                         batch_size=args.batch_size,
                         max_txt_length=args.context_length,
                         vision_model_type=args.vision_model)

    # Resume from a checkpoint.
    print("Begin to load model checkpoint from {}.".format(args.resume))
    assert os.path.exists(args.resume), "The checkpoint file {} not exists!".format(args.resume)
    # Map model to be loaded to specified single gpu.
    loc = "cuda:{}".format(args.gpu)
    checkpoint = torch.load(args.resume, map_location='cpu')
    start_epoch = checkpoint["epoch"]
    sd = checkpoint["state_dict"]
    if next(iter(sd.items()))[0].startswith('module'):
        sd = {k[len('module.'):]: v for k, v in sd.items() if "bert.pooler" not in k}
    model.load_state_dict(sd)
    print(f"=> loaded checkpoint '{args.resume}' (epoch {checkpoint['epoch']} @ {checkpoint['step']} steps)")

    write_cnt = 0
    img_feats = []
    txt_feats = []
    img_urls = []
    ocr_flags = []

    os.makedirs(args.save_dir, exist_ok=True)
    with open(args.img_id_url, 'r') as f:
        img_id_url = json.load(f)

    model.eval()
    dataloader = data.dataloader
    # 特征会保存为4个数据
    img_feats_path = os.path.join(args.save_dir, 'img_feats.pkl')
    txt_feats_path = os.path.join(args.save_dir, 'txt_feat.pkl')
    img_urls_path = os.path.join(args.save_dir, 'img_urls.pkl')
    ocr_flags_path = os.path.join(args.save_dir, 'ocr_flags.pkl')

    with torch.no_grad():
        for batch in tqdm(dataloader):
            images, texts, eos_indices, image_ids, raw_texts = batch
            # image_ids, images = batch
            images = images.cuda(args.gpu, non_blocking=True)
            texts = texts.cuda(args.gpu, non_blocking=True)
            img_feat, txt_feat, _ = model(images, texts)
            img_feat = img_feat.cpu().numpy()
            txt_feat = txt_feat.cpu().numpy()
            # mask_ = [i for i, e in enumerate(raw_texts) if e == ""]
            ocr_flag = list(map(lambda e: e == "", raw_texts))
            txt_feat[ocr_flag, :] = 0.0
            img_url = [img_id_url[img_id] for img_id in image_ids]

            img_feats.append(img_feat)
            txt_feats.append(txt_feat)
            img_urls.extend(img_url)
            ocr_flags.extend(ocr_flag)
            write_cnt += 1


    img_feats = np.concatenate(img_feats, axis=0)
    txt_feats = np.concatenate(txt_feats, axis=0)
    ocr_flags = np.array(ocr_flags)

    with open(img_feats_path, 'wb') as f:
        pickle.dump(img_feats, f)

    with open(txt_feats_path, 'wb') as f:
        pickle.dump(txt_feats, f)

    with open(img_urls_path, 'wb') as f:
        pickle.dump(img_urls, f)

    with open(ocr_flags_path, 'wb') as f:
        pickle.dump(ocr_flags, f)

    print(img_feats.shape)
    print(txt_feats.shape)
    # print(img_urls)

    print("Done!")