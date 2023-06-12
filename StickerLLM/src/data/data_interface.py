from typing import Any, Dict, Optional, Tuple, Union, List

import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split

from transformers import PreTrainedTokenizerBase
from omegaconf import ListConfig
import pyrootutils

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
from src.data.sticker_data import LMDBStickerDataset, LMDBStickerDatasetV2, LMDBStickerDatasetV3
from src.data.emoji_data import LMDBEmojiDataset,LMDBEmojiEvalDataset
from torch.utils.data.sampler import SequentialSampler

def _filter_out_truncated(item):

    pad_idx = torch.where(item['token_ids'] == 20003)[0]
    if len(pad_idx) == 0:
        return False
    return True

def _filter_out_invalid_data_with_pad(batch):


    old_size = len(batch)
    batch = list(filter(lambda x: x is not None, batch))
    batch = list(filter(_filter_out_truncated, batch))
    new_size = len(batch)
    if old_size != new_size:
        batch += batch[:old_size - new_size]
    return torch.utils.data.dataloader.default_collate(batch)

def _filter_out_invalid_data(batch):


    old_size = len(batch)
    batch = list(filter(lambda x: x is not None, batch))
    batch = list(filter(_filter_out_truncated, batch))
    return torch.utils.data.dataloader.default_collate(batch)

class StickerDataModule(LightningDataModule):
    """Example of LightningDataModule for MNIST dataset.

    A DataModule implements 5 key methods:

        def prepare_data(self):
            # things to do on 1 GPU/TPU (not on every GPU/TPU in DDP)
            # download data, pre-process, split, save to disk, etc...
        def setup(self, stage):
            # things to do on every process in DDP
            # load data, set variables, etc...
        def train_dataloader(self):
            # return train dataloader
        def val_dataloader(self):
            # return validation dataloader
        def test_dataloader(self):
            # return test dataloader
        def teardown(self):
            # called on every process in DDP
            # clean up after fit or test

    This allows you to share a full dataset without explaining how to download,
    split, transform and process the data.

    Read the docs:
        https://pytorch-lightning.readthedocs.io/en/latest/data/datamodule.html
    """

    def __init__(self,
                 data_dir: Union[str, List[str]],
                 tokenizer: PreTrainedTokenizerBase,
                 feat_dir: Optional[Union[str, List[str]]] = None,
                 add_special_tokens=False,
                 train_ratio: float = 0.9,
                 batch_size: int = 64,
                 max_txt_len: int = 64,
                 num_workers: int = 0,
                 pin_memory: bool = False,
                 img_size: int = 224,
                 num_vis_tokens: int = 1,
                 two_turn_ratio=0.6,
                 random_concat_ratio=0.3,
                 img_prompt_ratio=0.3,
                 random_img_ratio=0.3,):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False, ignore='tokenizer')

        self.tokenizer = tokenizer

        self.train_set: Optional[Dataset] = None
        self.valid_set: Optional[Dataset] = None
        self.test_set: Optional[Dataset] = None

    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by lightning with both `trainer.fit()` and `trainer.test()`, so be
        careful not to execute things like random split twice!
        """
        # load and split datasets only if not loaded already
        if isinstance(self.hparams.data_dir, (list, ListConfig)):
            datasets = []
            for i in range(len(self.hparams.data_dir)):
                datasets.append(
                    LMDBStickerDatasetV3(self.hparams.data_dir[i],
                                         self.tokenizer,
                                         add_special_tokens=self.hparams.add_special_tokens,
                                         max_txt_len=self.hparams.max_txt_len,
                                         use_augment=True,
                                         resolution=self.hparams.img_size,
                                         num_vis_tokens=self.hparams.num_vis_tokens,
                                         random_concat_ratio=self.hparams.random_concat_ratio,
                                         two_turn_ratio=self.hparams.two_turn_ratio,
                                         img_prompt_ratio=self.hparams.img_prompt_ratio,
                                         random_img_ratio = self.hparams.random_img_ratio,
                                         feat_path=self.hparams.feat_dir[i]))
            dataset = ConcatDataset(datasets=datasets)
        else:
            dataset = LMDBStickerDatasetV3(self.hparams.data_dir,
                                           self.tokenizer,
                                           add_special_tokens=self.hparams.add_special_tokens,
                                           max_txt_len=self.hparams.max_txt_len,
                                           use_augment=True,
                                           resolution=self.hparams.img_size,
                                           num_vis_tokens=self.hparams.num_vis_tokens,
                                           random_concat_ratio=self.hparams.random_concat_ratio,
                                           two_turn_ratio=self.hparams.two_turn_ratio,
                                           img_prompt_ratio=self.hparams.img_prompt_ratio,
                                           random_img_ratio = self.hparams.random_img_ratio,
                                           feat_path=self.hparams.feat_dir)
        num_samples = len(dataset)
        num_trainval = [int(self.hparams.train_ratio * num_samples), num_samples - int(self.hparams.train_ratio * num_samples)]

        self.train_set, self.valid_set = random_split(
            dataset=dataset,
            lengths=num_trainval,
            generator=torch.Generator().manual_seed(42),
        )

        self.test_set = self.valid_set

    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_set,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.valid_set,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.test_set,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def teardown(self, stage: Optional[str] = None):
        """Clean up after fit or test."""
        pass

    def state_dict(self):
        """Extra things to save to checkpoint."""
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Things to do when loading checkpoint."""
        pass


class EmojiDataModule(LightningDataModule):
    """Example of LightningDataModule for MNIST dataset.

    A DataModule implements 5 key methods:

        def prepare_data(self):
            # things to do on 1 GPU/TPU (not on every GPU/TPU in DDP)
            # download data, pre-process, split, save to disk, etc...
        def setup(self, stage):
            # things to do on every process in DDP
            # load data, set variables, etc...
        def train_dataloader(self):
            # return train dataloader
        def val_dataloader(self):
            # return validation dataloader
        def test_dataloader(self):
            # return test dataloader
        def teardown(self):
            # called on every process in DDP
            # clean up after fit or test

    This allows you to share a full dataset without explaining how to download,
    split, transform and process the data.

    Read the docs:
        https://pytorch-lightning.readthedocs.io/en/latest/data/datamodule.html
    """

    def __init__(self,
                 data_dir: str,
                 tokenizer: PreTrainedTokenizerBase,
                 feat_dir: Optional[Union[str, List[str]]] = None,
                 batch_size: int = 64,
                 ratio_t2i=0.5,
                 ratio_i2i=0.25,
                 max_txt_len=128,
                 num_workers: int = 0,
                 pin_memory: bool = False,**kargs):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False, ignore='tokenizer')

        self.tokenizer = tokenizer

        self.train_set: Optional[Dataset] = None
        self.valid_set: Optional[Dataset] = None
        self.test_set: Optional[Dataset] = None

    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by lightning with both `trainer.fit()` and `trainer.test()`, so be
        careful not to execute things like random split twice!
        """
        # load and split datasets only if not loaded already
        self.train_set = LMDBEmojiDataset(self.hparams.data_dir,
                                          self.tokenizer,
                                          feat_path=self.hparams.feat_dir,
                                          split='train',
                                          ratio_t2i=self.hparams.ratio_t2i,
                                          ratio_i2i=self.hparams.ratio_i2i,
                                          max_txt_len=self.hparams.max_txt_len)
        self.valid_set = LMDBEmojiDataset(self.hparams.data_dir,
                                          self.tokenizer,
                                          feat_path=self.hparams.feat_dir,
                                          split='val',
                                          ratio_t2i=self.hparams.ratio_t2i,
                                          ratio_i2i=self.hparams.ratio_i2i,
                                          max_txt_len=self.hparams.max_txt_len)

        self.test_set = self.valid_set

    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_set,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
            collate_fn=_filter_out_invalid_data_with_pad,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.valid_set,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
            collate_fn=_filter_out_invalid_data_with_pad,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.test_set,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
            collate_fn=_filter_out_invalid_data_with_pad,
        )

    def teardown(self, stage: Optional[str] = None):
        """Clean up after fit or test."""
        pass

    def state_dict(self):
        """Extra things to save to checkpoint."""
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Things to do when loading checkpoint."""
        pass


def get_emoji_eval_data(root_path,tokenizer,feat_path,split='val',max_txt_len=128,text2image=True,batch_size=16,num_workers=8):
    dataset = LMDBEmojiEvalDataset(root_path=root_path,tokenizer=tokenizer,feat_path=feat_path,split=split,max_txt_len=max_txt_len,text2image=text2image)
    dataloader = DataLoader(dataset=dataset,batch_size=batch_size,
                            pin_memory=False,
                            num_workers=num_workers,
                            drop_last=False,
                            sampler=SequentialSampler(dataset),
                            collate_fn=_filter_out_invalid_data)
    
    return {
        'dataset': dataset,
        'dataloader': dataloader
    }
