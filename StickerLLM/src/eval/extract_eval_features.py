import torch

import pyrootutils

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
from omegaconf import OmegaConf

from transformers.utils import logging
from src.models.fromage_model import StickerModel
import argparse
import hydra
import os
import tqdm
import json

logger = logging.get_logger(__name__)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-cfg', type=str, required=True, help="")
    parser.add_argument('--tokenizer-cfg', type=str, required=True, help="")
    parser.add_argument('--model-cfg', type=str, required=True, help="")
    parser.add_argument('--save-dir', type=str, required=True, help="")
    parser.add_argument('--ckpt-path', type=str, required=True, help="")

    parser.add_argument('--text2image', default=False,action='store_true', help="")
    parser.add_argument('--gpu', type=int, default=0, help="gpu id")
    return parser.parse_args()


def main(args):
    tokenizer_cfg = OmegaConf.load(args.tokenizer_cfg)
    data_cfg = OmegaConf.load(args.data_cfg)
    model_cfg = OmegaConf.load(args.model_cfg)

    print('Loading tokenizer...')
    tokenizer = hydra.utils.instantiate(tokenizer_cfg)
    print('Loading datamodule...')
    print(data_cfg)
    datamodule = hydra.utils.instantiate(data_cfg,tokenizer=tokenizer,text2image=args.text2image)
    dataloader = datamodule['dataloader']

    
    print('Loading model...')
    model = StickerModel.from_pretrained(tokenizer=tokenizer, model_cfg=model_cfg, ckpt_path=args.ckpt_path)

    model.to(args.gpu).eval()
    model.half()

    os.makedirs(args.save_dir,exist_ok=True)
    if args.text2image:
        llm_feat_save_path = os.path.join(args.save_dir,'llm_text_feat.jsonl')
        img_feat_save_path = os.path.join(args.save_dir,'image_feat.jsonl')
        image_fout = open(img_feat_save_path,'w')
        llm_fout = open(llm_feat_save_path,'w')
    else:
        llm_feat_save_path = os.path.join(args.save_dir,'llm_image_feat.jsonl')
        llm_fout = open(llm_feat_save_path,'w')
        image_fout = None

    cnt_write = 0

    with torch.no_grad():
        for batch in tqdm.tqdm(dataloader):

            img_feats = batch['img_feats'].cuda(args.gpu, non_blocking=True).half()
            input_text = batch['input_text']
            input_imgfeats = batch['input_imgfeats'].cuda(args.gpu, non_blocking=True).half()
            token_ids = batch['token_ids'].cuda(args.gpu, non_blocking=True)
            img_token_flag = batch['img_token_flag'].cuda(args.gpu, non_blocking=True)
            img_isuse_flag = batch['img_isuse_flag'].cuda(args.gpu, non_blocking=True)
            ret_token_idx = batch['ret_token_idx'].cuda(args.gpu, non_blocking=True)
            label_mask = batch['label_mask'].cuda(args.gpu, non_blocking=True)
            attention_mask = batch['attention_mask'].cuda(args.gpu, non_blocking=True)
            position_ids = batch['position_ids'].cuda(args.gpu, non_blocking=True)
            img_name = batch['img_name']

            output, image_features, llm_features = model(
                img_feats =img_feats,
                input_imgfeats=input_imgfeats,
                token_ids=token_ids,
                img_token_flag=img_token_flag,
                img_isuse_flag=img_isuse_flag,
                ret_token_idx=ret_token_idx,
                label_mask=label_mask,
                attention_mask=attention_mask,
                position_ids=position_ids,
                with_logit=False,
            )
            
            for img_id, img_feat,text_feat in zip(img_name,image_features.tolist(),llm_features.tolist()):
                cnt_write += 1
                if image_fout is not None:
                    image_fout.write("{}\n".format(json.dumps({"image_id": img_id, "feature": img_feat})))

                llm_fout.write("{}\n".format(json.dumps({"text_id": img_id, "feature": text_feat})))

        
        print('{} features are stored. '.format(cnt_write))


if __name__ == '__main__':
    args = parse_args()
    main(args)

