import torch
import torch.nn as nn
from omegaconf import OmegaConf
import hydra
from dataclasses import dataclass
from typing import Optional
import logging
import numpy as np
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from PIL import Image, UnidentifiedImageError
import pyrootutils
import json
import os
import pickle
import heapq
import deepspeed
from typing import Optional, Tuple, List

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
import src.utils.model_utils as utils
from src.utils.generation import *
from transformers.generation.utils import LogitsProcessorList, StoppingCriteriaList, GenerationConfig

logger = logging.getLogger(__name__)


@dataclass
class StickerArgs:
    freeze_lm: bool = True
    freeze_vm: bool = True
    num_vis_tokens: int = 1
    # img_embed_drop_prob: float = 0.0
    # task: str = 'captioning'
    shared_emb_dim: Optional[int] = 256
    # ret_token_id: int = 0


class StickerModel(nn.Module):

    def __init__(self, tokenizer, vis_encoder_cfg, lm_model_cfg, fromage_cfg) -> None:
        super().__init__()

        print('initianizing fromage model...')
        self.tokenizer = tokenizer
        self.vis_encoder_cfg = vis_encoder_cfg
        self.lm_model_cfg = lm_model_cfg
        self.fromage_cfg = fromage_cfg

        # self.tokenizer = hydra.utils.instantiate(tokenizer_cfg)
        self.vis_encoder = hydra.utils.instantiate(vis_encoder_cfg) if vis_encoder_cfg is not None else None
        self.lm_model = hydra.utils.instantiate(lm_model_cfg)

        if self.fromage_cfg.freeze_lm:
            logger.info('Frozen language model...')
            self.lm_model.eval()
            for param in self.lm_model.parameters():
                param.requires_grad = False

        if self.fromage_cfg.freeze_vm and self.vis_encoder is not None:
            logger.info('Frozen visual encoder...')
            self.vis_encoder.eval()
            for param in self.vis_encoder.parameters():
                param.requires_grad = False

        if self.lm_model.config.vocab_size < len(tokenizer):
            self.lm_model.resize_token_embeddings(len(tokenizer))
        else:
            self.lm_model.get_output_embeddings().weight.requires_grad = True
            self.lm_model.get_input_embeddings().weight.requires_grad = True
        self.txt_hidden_size = self.lm_model.config.hidden_size
        self.vis_embed_dim = self.fromage_cfg.num_vis_tokens * self.txt_hidden_size

        self.vis_hidden_size = self.vis_encoder.config.hidden_size if self.vis_encoder is not None else self.fromage_cfg.vis_hidden_size

        self.vis_embedding = nn.Linear(self.vis_hidden_size, self.vis_embed_dim)
        self.txt_embedding = self.lm_model.get_input_embeddings()

        self.vis_proj = nn.Linear(self.vis_hidden_size, self.fromage_cfg.shared_emb_dim, bias=False)
        self.txt_proj = nn.Linear(self.txt_hidden_size, self.fromage_cfg.shared_emb_dim, bias=False)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        extra_param_path = os.path.join(fromage_cfg.extra_param_dir, 'extra_param.pt')
        self.load_extra_param(extra_param_path)

        self.register_embedding_grad_hook()

    def load_extra_param(self, extra_param_path):
        logger.info('Loading extra param...')
        extra_param = torch.load(extra_param_path)
        self.vis_proj.weight.data = extra_param['visual_proj'].data.T.contiguous()
        self.vis_proj.requires_grad_(False)
        self.logit_scale.data = extra_param['logit_scale'].data

    def register_embedding_grad_hook(self):
        # mask = torch.ones(self.txt_embedding.weight.shape[0], dtype=torch.bool, device=self.txt_embedding.weight.device)
        mask = torch.ones(self.txt_embedding.weight.shape[0], dtype=torch.bool)
        mask[self.tokenizer.ret_token_id] = False
        mask[self.tokenizer.retend_token_id] = False
        mask[self.tokenizer.img_token_id] = False
        mask[self.tokenizer.imgend_token_id] = False
        mask[self.tokenizer.pret_token_id] = False

        def grad_hook(grad):
            grad[mask, :] = 0.
            # print('grad_ret: ', grad[self.tokenizer.ret_token_id, 0:10])
            return grad

        self.lm_model.get_input_embeddings().weight.register_hook(grad_hook)
        self.lm_model.get_output_embeddings().weight.register_hook(grad_hook)

    def train(self, mode=True):
        super(StickerModel, self).train(mode=mode)
        # Overwrite train() to ensure Frozen models remain frozen.
        if self.fromage_cfg.freeze_lm:
            self.lm_model.eval()
        if self.fromage_cfg.freeze_vm and self.vis_encoder is not None:
            self.vis_encoder.eval()

    def get_vis_ret_embs(self,
                         pixel_values: Optional[torch.Tensor] = None,
                         img_feats: Optional[torch.Tensor] = None,
                         with_logit=True):

        if img_feats is not None:
            vis_feat = img_feats
        else:
            vis_feat = self.vis_encoder(pixel_values).pooler_output
        vis_ret_embs = self.vis_proj(vis_feat)
        vis_ret_embs = torch.reshape(vis_ret_embs, (vis_ret_embs.shape[0], -1))
        vis_ret_embs = vis_ret_embs / vis_ret_embs.norm(dim=-1, keepdim=True)

        if with_logit:
            vis_ret_embs = self.logit_scale.exp() * vis_ret_embs

        return vis_ret_embs

    def get_vis_input_embs(self, pixel_values: Optional[torch.Tensor] = None, img_feats: Optional[torch.Tensor] = None):
        if img_feats is not None:
            vis_feat = img_feats
        else:
            vis_feat = self.vis_encoder(pixel_values).pooler_output
        vis_embs = self.vis_embedding(vis_feat)
        vis_input_embs = torch.reshape(vis_embs, (vis_embs.shape[0], self.fromage_cfg.num_vis_tokens, -1))
        return vis_input_embs

    def ret_forward(self,
                    pixel_values,
                    img_feats,
                    input_imgfeats,
                    input_ids,
                    img_token_flag,
                    img_isuse_flag,
                    ret_token_idx,
                    label_mask,
                    attention_mask,
                    position_ids,
                    with_logit=True):
        vis_ret_embs = self.get_vis_ret_embs(pixel_values=pixel_values, img_feats=img_feats,with_logit=with_logit)

        input_embs = self.txt_embedding(input_ids)
        batch_size = input_embs.shape[0]
        vis_input_embs = self.get_vis_input_embs(pixel_values=None,
                                                img_feats=input_imgfeats)
        vis_input_embs = vis_input_embs.view(batch_size, self.fromage_cfg.num_vis_tokens, self.txt_hidden_size)

        if img_isuse_flag.sum() > 0:
            input_embs[img_token_flag] = vis_input_embs[img_isuse_flag].view(-1,self.txt_hidden_size)
        else:
            # to avoid sync error of gradient in backward
            input_embs[:,0,:] = input_embs[:,0,:] + 0.0 * vis_input_embs[:,0,:]

        target_ids = input_ids.clone().detach()
        target_ids[label_mask] = -100

        output = self.lm_model(input_ids=input_ids,
                               inputs_embeds=input_embs,
                               labels=target_ids,
                               position_ids=position_ids,
                               attention_mask=attention_mask,
                               output_hidden_states=True)
        last_hidden_state = output.hidden_states[-1].permute(1, 0, 2).contiguous()
        # last_hidden_state = output.hidden_states.permute(1, 0, 2).contiguous()
        hidden_state = [last_hidden_state[i, ret_token_idx[i], :] for i in range(input_embs.shape[0])]  # B x D
        hidden_state = torch.stack(hidden_state, dim=0)
        txt_ret_embs = self.txt_proj(hidden_state)  # TODO validate the correctness
        # print('txt_ret_embs before norm: ', txt_ret_embs.min(), txt_ret_embs.max(), txt_ret_embs.isnan().sum())
        txt_ret_embs = txt_ret_embs / txt_ret_embs.norm(dim=-1, keepdim=True)

        return output, vis_ret_embs, txt_ret_embs

    def forward(self,
                pixel_values: torch.FloatTensor = None,
                img_feats: torch.FloatTensor = None,
                input_imgfeats: torch.FloatTensor = None,
                token_ids: torch.LongTensor = None,
                img_token_flag: torch.Tensor = None,
                img_isuse_flag: torch.Tensor = None,
                ret_token_idx: torch.LongTensor = None,
                label_mask: torch.Tensor = None,
                attention_mask: torch.Tensor = None,
                position_ids: torch.Tensor = None,
                with_logit:bool=True,):

        output, vis_ret_embs, txt_ret_embs = self.ret_forward(
            pixel_values=pixel_values,img_feats=img_feats,input_imgfeats=input_imgfeats,input_ids=token_ids,img_token_flag=img_token_flag,img_isuse_flag=img_isuse_flag,ret_token_idx=ret_token_idx,label_mask=label_mask,attention_mask=attention_mask,position_ids=position_ids,with_logit=with_logit,
        )


        return output, vis_ret_embs, txt_ret_embs

    def generate_for_imgs_and_texts(self,
                                    prompts: list,
                                    num_words: int = 0,
                                    ret_scale_factor: float = 1.0,
                                    top_p: float = 1.0,
                                    temperature: float = 0.0,
                                    max_num_rets: int = 1):
        """
        Encode prompts into embeddings.

        Args:
        prompts: List of interleaved PIL.Image.Image and strings representing input to the model.
        num_words: Maximum number of words to generate for. If num_words = 0, the model will run its forward pass and return the outputs.
        ret_scale_factor: Proportion to scale [RET] token logits by. A higher value may increase the probability of the model generating [RET] outputs.
        top_p: If set to < 1, the smallest set of tokens with highest probabilities that add up to top_p or higher are kept for generation.
        temperature: Used to modulate logit distribution.
        max_num_rets: Maximum number of images to return in one generation pass.
        Returns:
        return_outputs: List consisting of either str or List[PIL.Image.Image] objects, representing image-text interleaved model outputs.
        """
        input_embs = []
        input_ids = []
        # add_bos = True
        add_bos = False

        for i, p in enumerate(prompts):
            if type(p) == Image.Image:
                # Encode as image.
                pixel_values = utils.get_pixel_values(self.feature_extractor, p)
                pixel_values = pixel_values.to(device=self.logit_scale.device, dtype=self.logit_scale.dtype)
                pixel_values = pixel_values[None, ...]
                pixel_values = torch.cat([pixel_values, pixel_values, pixel_values], dim=1)
                print(pixel_values.shape)
                visual_embs = self.get_vis_embs(pixel_values, mode='captioning')  # (1, n_visual_tokens, D)
                input_embs.append(visual_embs)
            elif type(p) == str:
                text_ids = self.tokenizer(p, add_special_tokens=False,
                                          return_tensors="pt").input_ids.to(self.logit_scale.device)
                if not add_bos:
                    # Remove <bos> tag.
                    text_ids = text_ids[:, 1:]
                else:
                    # Only add <bos> once.
                    add_bos = False

                text_embs = self.txt_embedding(text_ids)  # (1, T, D)
                input_embs.append(text_embs)
                input_ids.append(text_ids)
            else:
                raise ValueError(f'Input prompts should be either PIL.Image.Image or str types, got {type(p)} instead.')
        input_embs = torch.cat(input_embs, dim=1)
        input_ids = torch.cat(input_ids, dim=1)

        if num_words == 0:
            generated_ids = input_ids
            outputs = self.lm_model(inputs_embeds=input_embs, output_hidden_states=True)
            # Map outputs to embeddings, so we can retrieve embeddings from the [RET] tokens.

            txt_final_embs = self.txt_proj(outputs.hidden_states[-1])
            txt_final_embs = txt_final_embs / txt_final_embs.norm(dim=-1, keepdim=True)

        elif num_words > 0:
            generated_ids, txt_final_embs, _ = self.generate(input_embs,
                                                             num_words,
                                                             temperature=temperature,
                                                             top_p=top_p,
                                                             ret_scale_factor=ret_scale_factor)
            txt_final_embs = txt_final_embs[:, input_embs.shape[1]:]

            # Truncate to newline.
            # newline_token_id = self.tokenizer('\n', add_special_tokens=False).input_ids[0]
            # trunc_idx = 0
            # for j in range(generated_ids.shape[1]):
            #     if generated_ids[0, j] == newline_token_id:
            #         trunc_idx = j
            #         break
            # if trunc_idx > 0:
            #     generated_ids = generated_ids[:, :trunc_idx]
            #     embeddings = embeddings[:, :trunc_idx]
        else:
            raise ValueError

        # Save outputs as an interleaved list.
        return_outputs = []
        # Find up to max_num_rets [RET] tokens, and their corresponding scores.
        all_ret_idx = [i for i, x in enumerate(generated_ids[0, :] == self.tokenizer.ret_token_id) if x][:max_num_rets]
        seen_image_idx = []  # Avoid showing the same image multiple times.

        last_ret_idx = 0
        if len(all_ret_idx) == 0:
            # No [RET] tokens.
            caption = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            return_outputs.append(utils.truncate_caption(caption))
        else:
            for ret_idx in all_ret_idx:
                ret_emb = txt_final_embs[:, ret_idx, :]
                # scores = self.emb_matrix @ ret_emb.T
                scores = self.topk_retrieve(ret_emb)

                # Downweight seen images.
                for seen_idx in seen_image_idx:
                    scores[seen_idx] -= 1000

                # Get the top 3 images for each image.
                _, top_image_idx = scores.squeeze().topk(3)
                image_outputs = []
                for img_idx in top_image_idx:
                    # Find the first image that does not error out.
                    try:
                        seen_image_idx.append(img_idx)
                        img = utils.get_image_from_url(self.img_urls[img_idx])
                        image_outputs.append(img)
                        if len(image_outputs) == max_num_rets:
                            break
                    except UnidentifiedImageError:
                        pass

                caption = self.tokenizer.batch_decode(generated_ids[:, last_ret_idx:ret_idx], skip_special_tokens=True)[0]
                last_ret_idx = ret_idx + 1
                return_outputs.append(utils.truncate_caption(caption) + ' [RET]')
                return_outputs.append(image_outputs)

        return return_outputs


    @classmethod
    def from_pretrained(cls, tokenizer, model_cfg, ckpt_path=None, database_dir=None, extractor_name=None):
        instance = cls(tokenizer=tokenizer,
                       vis_encoder_cfg=model_cfg.vis_encoder,
                       lm_model_cfg=model_cfg.lm_model,
                       fromage_cfg=model_cfg.fromage)

        print(model_cfg)
        if ckpt_path is not None:
            ckpt = torch.load(ckpt_path)
            # sd = checkpoint["state_dict"]
            # if next(iter(sd.items()))[0].startswith('model'):
            #     sd = {k[len('model.'):]: v for k, v in sd.items() if "bert.pooler" not in k}

            ckpt = {k[len('_forward_module.model.'):]: v for k, v in ckpt.items() if "_forward_module.model." in k}
            print('ckpt keys: ',ckpt.keys())
            missing_keys, unexpected_keys = instance.load_state_dict(ckpt, strict=False)
            print('Unexpected_keys', unexpected_keys)
            print('Missing keys: ',missing_keys)
            print('load ckpt successfully...')

        if extractor_name is not None and model_cfg.vis_encoder is not None:
            instance.feature_extractor = utils.get_feature_extractor(extractor_name, image_size=224, train=False)

        if database_dir is not None:
            img_feats_path = os.path.join(database_dir, 'img_feats.pkl')
            txt_feats_path = os.path.join(database_dir, 'txt_feat.pkl')
            img_urls_path = os.path.join(database_dir, 'img_urls.pkl')
            ocr_flags_path = os.path.join(database_dir, 'ocr_flags.pkl')

            with open(img_feats_path, 'rb') as f:
                instance.img_feats = pickle.load(f)

            with open(txt_feats_path, 'rb') as f:
                instance.txt_feats = pickle.load(f)

            with open(img_urls_path, 'rb') as f:
                instance.img_urls = pickle.load(f)

            with open(ocr_flags_path, 'rb') as f:
                instance.ocr_flags = pickle.load(f)

            print('Load image&text features succussfully.')

        return instance

    def topk_retrieve(self, query_embed, topk=None, batch_size=32768, beta=0.5):
        scores = []
        idx = 0
        query_embed = query_embed.to(dtype=torch.float)
        while idx < len(self.img_urls):
            img_feats_tensor = torch.from_numpy(
                self.img_feats[idx:min(idx + batch_size, len(self.img_urls))]).cuda(query_embed.device)  # [batch_size, feature_dim]
            batch_scores = query_embed @ img_feats_tensor.t()  # [1, batch_size]

            txt_feats_tensor = torch.from_numpy(
                self.txt_feats[idx:min(idx + batch_size, len(self.img_urls))]).cuda(query_embed.device)  # [batch_size, feature_dim]
            batch_txt_scores = query_embed @ txt_feats_tensor.t()  # [1, batch_size]

            ocr_flag = self.ocr_flags[idx:min(idx + batch_size, len(self.img_urls))]
            batch_txt_scores[:, ocr_flag] = batch_scores[:, ocr_flag]

            scores.extend((batch_scores + beta * batch_txt_scores).squeeze(0).tolist())
            idx += batch_size

        topk_idx = heapq.nlargest(topk, range(len(scores)), scores.__getitem__)
        topk_urls = [self.img_urls[i] for i in topk_idx]
        return topk_urls
        # scores = torch.tensor(scores)
        # return scores


    def generate(self,input_ids,query_img=None,**gen_kwargs):

        inputs_embeds = self.txt_embedding(input_ids)

        if query_img is not None:
            img_idx = torch.where(input_ids==self.tokenizer.imgend_token_id)[1] - self.fromage_cfg.num_vis_tokens
            pixel_values  = utils.get_pixel_values(self.feature_extractor, query_img).to(inputs_embeds.device).half()
            # print('img_idx: ',img_idx)
            # print('img_idx: ',input_ids)

            img_input_embs = self.get_vis_input_embs(pixel_values=pixel_values.unsqueeze(0).repeat(1,3,1,1))
            inputs_embeds[0,img_idx:img_idx + self.fromage_cfg.num_vis_tokens,:] = img_input_embs.squeeze(0)
        
        outputs = self.lm_model.generate(input_ids=input_ids,inputs_embeds=inputs_embeds,**gen_kwargs)

        return outputs


    @torch.no_grad()
    def chat(self,
             query: str,
             history: List[Tuple[str, str]] = None,
             query_img:Optional[Image.Image] = None,
             max_length: int = 2048,
             num_beams=1,
             do_sample=True,
             top_p=0.7,
             top_k=24,
             temperature=0.95,
             logits_processor=None,
             stopping_criteria=None,
             ret_prob_scale=1.0,
             prefix=True,
             beta=0.3,
             **kwargs):
        if history is None:
            history = []
        if logits_processor is None:
            logits_processor = LogitsProcessorList()
        logits_processor.append(InvalidScoreLogitsProcessor())
        logits_processor.append(RetrievalEndLogitsProcessor(self.tokenizer.ret_token_id, self.tokenizer.retend_token_id))
        logits_processor.append(RetrievalProbLogitsProcessor(self.tokenizer.ret_token_id, 1.0))
        logits_processor.append(RetrievalProbLogitsProcessor(self.tokenizer.pret_token_id, 0.0))
        logits_processor.append(InvalidIdProcessor(self.tokenizer.pret_token_id))
        if stopping_criteria is None:
            stopping_criteria = StoppingCriteriaList()
        # stopping_criteria.append(RetrievalStoppingCriteria(self.tokenizer.ret_token_id))

        gen_kwargs = {
            "max_length": max_length,
            "num_beams": num_beams,
            "do_sample": do_sample,
            "top_p": top_p,
            "temperature": temperature,
            "logits_processor": logits_processor,
            "stopping_criteria": stopping_criteria,
            "output_hidden_states": True,
            "return_dict_in_generate": True,
            **kwargs
        }
        img_placeholder = '[IMG]' + '[IMG]' * self.fromage_cfg.num_vis_tokens + '[/IMG]'
        
        prompt = ""
        for i, (old_query, response) in enumerate(history):
            if i == 0 and query_img is not None:
                prompt += "[Round {}]\n问：{}{}\n答：{}\n".format(i,img_placeholder, old_query, response)
            else:
                prompt += "[Round {}]\n问：{}\n答：{}\n".format(i, old_query, response)
        if len(history) == 0 and query_img is not None:
            prompt += "[Round {}]\n问：{}{}\n答：".format(len(history),img_placeholder, query)
        else:
            prompt += "[Round {}]\n问：{}\n答：".format(len(history), query)
    
        if prefix:
            prompt = '[PRET] ' + prompt

        input_ids = self.tokenizer([prompt], return_tensors="pt", padding=True).input_ids

        input_ids = input_ids.to(self.lm_model.device)
        # outputs = self.lm_model.generate(**input_ids, **gen_kwargs)
        outputs = self.generate(input_ids=input_ids,query_img=query_img,**gen_kwargs)

        output_ids = outputs.sequences[0]
        generated_ids = output_ids[len(input_ids[0]):]
        generated_hidden_states = outputs.hidden_states[1:]

        ret_token_idx = torch.where(generated_ids == self.tokenizer.ret_token_id)[0]

        if len(ret_token_idx) > 0:
            ret_hidden_state = generated_hidden_states[ret_token_idx[0]][-1][:, 0, :]
            txt_ret_embs = self.txt_proj(ret_hidden_state)  # TODO validate the correctness
            txt_ret_embs = txt_ret_embs / txt_ret_embs.norm(dim=-1, keepdim=True)
            img_urls = self.topk_retrieve(txt_ret_embs, topk=top_k,beta=beta)
        else:
            img_urls = None

        # img_urls = None

        # outputs = outputs.tolist()[0][len(input_ids["input_ids"][0]):]
        response = self.tokenizer.decode(generated_ids)
        response = self.lm_model.process_response(response)
        history = history + [(query, response.replace('<eop>', ''))]
        return response, history, img_urls
