from typing import List, Optional, Tuple
import os
import hydra
import pyrootutils
import pytorch_lightning as pl
from omegaconf import DictConfig
from pytorch_lightning import Callback, LightningDataModule, LightningModule, Trainer
from pytorch_lightning.loggers import Logger
from pytorch_lightning.strategies import DeepSpeedStrategy

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
# ------------------------------------------------------------------------------------ #
# the setup_root above is equivalent to:
# - adding project root dir to PYTHONPATH
#       (so you don't need to force user to install project as a package)
#       (necessary before importing any local modules e.g. `from src import utils`)
# - setting up PROJECT_ROOT environment variable
#       (which is used as a base for paths in "configs/paths/default.yaml")
#       (this way all filepaths are the same no matter where you run the code)
# - loading environment variables from ".env" in root dir
#
# you can remove it if you:
# 1. either install project as a package or move entry files to project root dir
# 2. set `root_dir` to "." in "configs/paths/default.yaml"
#
# more info: https://github.com/ashleve/pyrootutils
# ------------------------------------------------------------------------------------ #

from src import utils
from src.utils.model_utils import add_ret_token, add_img_token, add_pret_token
from src.models.model_interface import StickerLitModule

log = utils.get_pylogger(__name__)


@utils.task_wrapper
def train(cfg: DictConfig) -> Tuple[dict, dict]:
    """Trains the model. Can additionally evaluate on a testset, using best weights obtained during
    training.

    This method is wrapped in optional @task_wrapper decorator which applies extra utilities
    before and after the call.

    Args:
        cfg (DictConfig): Configuration composed by Hydra.

    Returns:
        Tuple[dict, dict]: Dict with metrics and dict with all instantiated objects.
    """

    # set seed for random number generators in pytorch, numpy and python.random
    if cfg.get("seed"):
        pl.seed_everything(cfg.seed, workers=True)

    log.info(f"Instantiating tokenizer <{cfg.tokenizer._target_}>")
    tokenizer = hydra.utils.instantiate(cfg.tokenizer)
    # tokenizer = add_ret_token(tokenizer)
    # tokenizer = add_img_token(tokenizer)
    # tokenizer = add_pret_token(tokenizer)

    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data, tokenizer=tokenizer)

    datamodule.setup()
    num_batches_per_epoch = len(datamodule.train_dataloader())
    total_steps = int(num_batches_per_epoch * cfg.trainer.max_epochs /
                      (cfg.train.grad_accumulate_batches * cfg.train.sche_step_size * cfg.trainer.get('devices', 1)))
    log.info(f'Total training steps: {total_steps}')
    cfg.scheduler.total_steps = total_steps

    log.info(f"Instantiating fromage model...")
    # model: LightningModule = hydra.utils.instantiate(cfg.model)
    model: LightningModule = StickerLitModule(tokenizer, cfg.model, cfg.optimizer, cfg.scheduler, cfg.train)

    log.info("Instantiating callbacks...")
    callbacks: List[Callback] = utils.instantiate_callbacks(cfg.get("callbacks"))

    log.info("Instantiating loggers...")
    logger: List[Logger] = utils.instantiate_loggers(cfg.get("logger"))

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")



    # trainer: Trainer = hydra.utils.instantiate(cfg.trainer, callbacks=callbacks, logger=logger, strategy=strategy)
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer, callbacks=callbacks, logger=logger)

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "callbacks": callbacks,
        "logger": logger,
        "trainer": trainer,
    }

    if logger:
        log.info("Logging hyperparameters!")
        utils.log_hyperparameters(object_dict)

    ckpt_path = os.path.join(cfg.restore_dir, 'last.ckpt')
    os.makedirs(cfg.restore_dir, exist_ok=True)
    if not os.path.exists(ckpt_path):
        ckpt_path = None
    log.info(f'Model restore checkpoint : {ckpt_path}')


    if cfg.get("train"):
        log.info("Starting training!")
        trainer.fit(model=model, datamodule=datamodule, ckpt_path=ckpt_path)

    train_metrics = trainer.callback_metrics

    metric_dict = train_metrics

    return metric_dict, object_dict


config_path = '../configs'


@hydra.main(version_base="1.3", config_path=config_path, config_name="train.yaml")
def main(cfg: DictConfig) -> Optional[float]:

    # train the model
    metric_dict, _ = train(cfg)

    # return optimized metric
    return metric_dict


if __name__ == "__main__":
    main()
