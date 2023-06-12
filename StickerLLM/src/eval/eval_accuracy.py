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
import time

logger = logging.get_logger(__name__)

PREFIX = '[PRET]'

eval_map_1 ={
    'emoji_in_domain': 'emoji_in_domain',
    'emoji_out_domain': 'emoji_out_domain',
    'emoji_in_domain_with_prefix': 'emoji_in_domain',
    'emoji_out_domain_with_prefix': 'emoji_out_domain'
}

eval_map_2 ={
    'belle': 'belle'
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', type=str, required=True, help="")
    parser.add_argument('--tokenizer-cfg', type=str, required=True, help="")
    parser.add_argument('--model-cfg', type=str, required=True, help="")
    parser.add_argument('--ckpt-path', type=str, required=True, help="")
    parser.add_argument('--save-path', type=str, required=True, help="")
    parser.add_argument('--eval-map', type=int, required=True, help="")

    parser.add_argument('--gpu', type=int, default=0, help="gpu id")
    return parser.parse_args()


def main(args):
    tokenizer_cfg = OmegaConf.load(args.tokenizer_cfg)
    model_cfg = OmegaConf.load(args.model_cfg)

    print('Loading tokenizer...')
    tokenizer = hydra.utils.instantiate(tokenizer_cfg)

    model = StickerModel.from_pretrained(tokenizer=tokenizer, model_cfg=model_cfg, ckpt_path=args.ckpt_path)

    model.to(args.gpu).eval()
    model.half()

    with open(args.data_path, 'r') as f:
        data = json.load(f)

    results = []

    if args.eval_map == 1:
        eval_map = eval_map_1
    elif args.eval_map == 2:
        eval_map = eval_map_2
    else:
        raise NotImplementedError

    for eval_key,data_key in eval_map.items():
        class_samples = data[data_key]

        prefix = 'prefix' in eval_key
        emoji = 'emoji' in eval_key
        
        count = 0
        time1 = time.time()

        count_true = 0
        print(f'eval_key: {eval_key}')
        for idx,query in tqdm.tqdm(enumerate(class_samples)):
            response,history,img_urls = model.chat(query=query,max_length=256,prefix=prefix)
            count += 1

            if emoji and '[RET]' in response:
                count_true += 1
            
            if not emoji and not '[RET]' in response:
                count_true += 1

            # if idx < 3:
            #     print('history: ',history,'response: ',response)
            # else:
            #     break
        
        time2 = time.time()
        time_delta = time2 - time1 
        accuracy = count_true / count
        
        result = f'eval_key: {eval_key}, time: {time_delta}, accuracy: {accuracy}'
        print(result)
        results.append(result)
    
    with open(args.save_path,'w') as f:
        json.dump(results,f,indent=2)

if __name__ == '__main__':
    args = parse_args()
    main(args)

        






    