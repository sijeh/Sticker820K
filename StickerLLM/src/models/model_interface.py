from typing import Any, List, Optional

import torch
from pytorch_lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.classification.accuracy import Accuracy
from omegaconf import OmegaConf
import numpy as np
from .fromage_model import StickerModel
import torch.distributed.fsdp.wrap as fsdp

import hydra
import pyrootutils

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
import src.utils.metric as metric


class StickerLitModule(LightningModule):
    """Example of LightningModule for MNIST classification.

    A LightningModule organizes your PyTorch code into 6 sections:
        - Computations (init)
        - Train loop (training_step)
        - Validation loop (validation_step)
        - Test loop (test_step)
        - Prediction Loop (predict_step)
        - Optimizers and LR Schedulers (configure_optimizers)

    Docs:
        https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html
    """

    def __init__(self, tokenizer, model_cfg, optimizer_cfg, scheduler_cfg, train_cfg):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.model = StickerModel(tokenizer, model_cfg.vis_encoder, model_cfg.lm_model, model_cfg.fromage)

        self.automatic_optimization = False
        # # loss function
        # self.criterion = torch.nn.CrossEntropyLoss()

        # metric objects for calculating and averaging accuracy across batches

        self.ret_ce_train_loss = MeanMetric()
        self.ret_i2t_train_loss = MeanMetric()
        self.ret_t2i_train_loss = MeanMetric()
        self.total_train_loss = MeanMetric()


        self.ret_ce_valid_loss = MeanMetric()
        self.ret_i2t_valid_loss = MeanMetric()
        self.ret_t2i_valid_loss = MeanMetric()
        self.total_valid_loss = MeanMetric()

        self.ret_i2t_train_top1 = MeanMetric()
        self.ret_i2t_train_top5 = MeanMetric()
        self.ret_t2i_train_top1 = MeanMetric()
        self.ret_t2i_train_top5 = MeanMetric()

        self.ret_i2t_valid_top1 = MeanMetric()
        self.ret_i2t_valid_top5 = MeanMetric()
        self.ret_t2i_valid_top1 = MeanMetric()
        self.ret_t2i_valid_top5 = MeanMetric()

    def forward(self, *args, **kwargs) -> Any:
        return self.model(*args, **kwargs)

    def training_step(self, batch: Any, batch_idx: int):

        opt = self.optimizers()
        sche = self.lr_schedulers()
        # images, tokens, ret_emb_idx = batch
        total_loss = 0
        # for mode in ['captioning', 'retrieval']:


        output, vis_ret_embed, txt_ret_embed = self.forward(
            img_feats = batch['img_feats'],
            input_imgfeats=batch['input_imgfeats'],
            token_ids=batch['token_ids'],
            img_token_flag=batch['img_token_flag'],
            img_isuse_flag=batch['img_isuse_flag'],
            ret_token_idx=batch['ret_token_idx'],
            label_mask=batch['label_mask'],
            attention_mask=batch['attention_mask'],
            position_ids=batch['position_ids']
        )

        self.ret_ce_train_loss(output.loss)
        self.log("train/ret_ce_loss", self.ret_ce_train_loss, on_step=True, on_epoch=True, prog_bar=True)
        ret_ce_loss = output.loss * self.hparams.train_cfg.ret_loss_scale
        total_loss += ret_ce_loss

        all_txt_ret_embed = self.all_gather(txt_ret_embed, sync_grads=True)
        all_vis_ret_embed = self.all_gather(vis_ret_embed, sync_grads=True)

        all_txt_ret_embed = all_txt_ret_embed.view(-1, all_txt_ret_embed.shape[-1])
        all_vis_ret_embed = all_vis_ret_embed.view(-1, all_vis_ret_embed.shape[-1])

        logits_per_img = all_vis_ret_embed @ all_txt_ret_embed.t()
        logits_per_txt = logits_per_img.t()

        dtype = logits_per_img.dtype
        logits_per_img = logits_per_img.to(dtype=torch.float32)
        logits_per_txt = logits_per_txt.to(dtype=torch.float32)

        ret_i2t_loss = metric.contrastive_loss(logits_per_img)
        ret_t2i_loss = metric.contrastive_loss(logits_per_txt)

        logits_per_img = logits_per_img.to(dtype)
        logits_per_txt = logits_per_txt.to(dtype)



        self.ret_i2t_train_loss(ret_i2t_loss)
        self.ret_t2i_train_loss(ret_t2i_loss)
        self.log("train/ret_i2t_loss", self.ret_i2t_train_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/ret_t2i_loss", self.ret_t2i_train_loss, on_step=True, on_epoch=True, prog_bar=True)

        contrastive_loss = (ret_i2t_loss + ret_t2i_loss) / 2.0 * self.hparams.train_cfg.contrastive_loss_scale
        total_loss += contrastive_loss

        self.manual_backward(total_loss)
        self.clip_gradients(opt,
                            gradient_clip_val=self.hparams.train_cfg.gradient_clip_val,
                            gradient_clip_algorithm="norm")
        # self.model.revise_txt_embed_grad()
        opt.step()
        opt.zero_grad()

        ret_i2t_top1, ret_i2t_top5 = metric.contrastive_acc(logits_per_img, topk=(1, 5))
        ret_t2i_top1, ret_t2i_top5 = metric.contrastive_acc(logits_per_txt, topk=(1, 5))
        self.ret_i2t_train_top1(ret_i2t_top1)
        self.ret_i2t_train_top5(ret_i2t_top5)
        self.ret_t2i_train_top1(ret_t2i_top1)
        self.ret_t2i_train_top5(ret_t2i_top5)

        self.log("train/ret_i2t_top1", self.ret_i2t_train_top1, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/ret_i2t_top5", self.ret_i2t_train_top5, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/ret_t2i_top1", self.ret_t2i_train_top1, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/ret_t2i_top5", self.ret_t2i_train_top5, on_step=True, on_epoch=True, prog_bar=True)


        total_loss = total_loss / self.hparams.train_cfg.grad_accumulate_batches
        self.total_train_loss(total_loss)
        self.log("train/total_loss", self.total_train_loss, on_step=True, on_epoch=True, prog_bar=True)


        if (batch_idx + 1) % (self.hparams.train_cfg.grad_accumulate_batches * self.hparams.train_cfg.sche_step_size) == 0:
            sche.step()

        # we can return here dict with any tensors
        # and then read it in some callback or in `training_epoch_end()` below
        # remember to always return loss from `training_step()` or backpropagation will fail!
        return {"loss": total_loss}

    def validation_step(self, batch: Any, batch_idx: int):
        # images, tokens, ret_emb_idx = batch
        total_loss = 0

        output, vis_ret_embed, txt_ret_embed = self.forward(
            img_feats = batch['img_feats'],
            input_imgfeats=batch['input_imgfeats'],
            token_ids=batch['token_ids'],
            img_token_flag=batch['img_token_flag'],
            img_isuse_flag=batch['img_isuse_flag'],
            ret_token_idx=batch['ret_token_idx'],
            label_mask=batch['label_mask'],
            attention_mask=batch['attention_mask'],
            position_ids=batch['position_ids']
        )

        self.ret_ce_valid_loss(output.loss)
        self.log("valid/ret_ce_loss", self.ret_ce_valid_loss, on_step=True, on_epoch=True, prog_bar=True)
        ret_ce_loss = output.loss * self.hparams.train_cfg.ret_loss_scale
        total_loss += ret_ce_loss

        all_txt_ret_embed = self.all_gather(txt_ret_embed)
        all_vis_ret_embed = self.all_gather(vis_ret_embed)

        all_txt_ret_embed = all_txt_ret_embed.view(-1, all_txt_ret_embed.shape[-1])
        all_vis_ret_embed = all_vis_ret_embed.view(-1, all_vis_ret_embed.shape[-1])

        logits_per_img = all_vis_ret_embed @ all_txt_ret_embed.t()
        logits_per_txt = logits_per_img.t()

        ret_i2t_loss = metric.contrastive_loss(logits_per_img)
        ret_t2i_loss = metric.contrastive_loss(logits_per_txt)

        self.ret_i2t_valid_loss(ret_i2t_loss)
        self.ret_t2i_valid_loss(ret_t2i_loss)
        self.log("valid/ret_i2t_loss", self.ret_i2t_valid_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("valid/ret_t2i_loss", self.ret_t2i_valid_loss, on_step=True, on_epoch=True, prog_bar=True)

        contrastive_loss = (ret_i2t_loss + ret_t2i_loss) / 2.0 * self.hparams.train_cfg.contrastive_loss_scale
        total_loss += contrastive_loss

        ret_i2t_top1, ret_i2t_top5 = metric.contrastive_acc(logits_per_img, topk=(1, 5))
        ret_t2i_top1, ret_t2i_top5 = metric.contrastive_acc(logits_per_txt, topk=(1, 5))
        self.ret_i2t_valid_top1(ret_i2t_top1)
        self.ret_i2t_valid_top5(ret_i2t_top5)
        self.ret_t2i_valid_top1(ret_t2i_top1)
        self.ret_t2i_valid_top5(ret_t2i_top5)

        self.log("valid/ret_i2t_top1", self.ret_i2t_valid_top1, on_step=True, on_epoch=True, prog_bar=True)
        self.log("valid/ret_i2t_top5", self.ret_i2t_valid_top5, on_step=True, on_epoch=True, prog_bar=True)
        self.log("valid/ret_t2i_top1", self.ret_t2i_valid_top1, on_step=True, on_epoch=True, prog_bar=True)
        self.log("valid/ret_t2i_top5", self.ret_t2i_valid_top5, on_step=True, on_epoch=True, prog_bar=True)


        total_loss = total_loss / self.hparams.train_cfg.grad_accumulate_batches
        self.total_valid_loss(total_loss)
        self.log("valid/total_loss", self.total_valid_loss, on_step=True, on_epoch=True, prog_bar=True)
        # we can return here dict with any tensors
        # and then read it in some callback or in `training_epoch_end()` below
        # remember to always return loss from `training_step()` or backpropagation will fail!
        return {"loss": total_loss}

    def test_step(self, batch: Any, batch_idx: int):
        return self.validation_step(batch, batch_idx)

    def test_epoch_end(self, outputs: List[Any]):
        pass

    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
            
        """
        optimizer = hydra.utils.instantiate(self.hparams.optimizer_cfg, self.trainer.model.parameters())
        scheduler = hydra.utils.instantiate(self.hparams.scheduler_cfg, optimizer=optimizer)

        # return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "epoch", "frequency": 1}}
        return {"optimizer": optimizer, "lr_scheduler": scheduler}
