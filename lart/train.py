import os
from typing import List, Optional, Tuple

import hydra
import lightning as L
import pyrootutils
import submitit
import torch
from lightning import Callback, LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig

from lart import utils

log = utils.get_pylogger(__name__)

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=True,
)

# ------------------------------------------------------------------------------------ #
# `pyrootutils.setup_root(...)` is recommended at the top of each start file
# to make the environment more robust and consistent
#
# the line above searches for ".git" or "pyproject.toml" in present and parent dirs
# to determine the project root dir
#
# adds root dir to the PYTHONPATH (if `pythonpath=True`)
# so this file can be run from any place without installing project as a package
#
# sets PROJECT_ROOT environment variable which is used in "configs/paths/default.yaml"
# this makes all paths relative to the project root
#
# additionally loads environment variables from ".env" file (if `dotenv=True`)
#
# you can get away without using `pyrootutils.setup_root(...)` if you:
# 1. move this file to the project root dir or install project as a package
# 2. modify paths in "configs/paths/default.yaml" to not use PROJECT_ROOT
# 3. always run this file from the project root dir
#
# https://github.com/ashleve/pyrootutils
# ------------------------------------------------------------------------------------ #

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
    
    # Determine if we are using submitit
    try:
        env_information = submitit.JobEnvironment()
        log.info(f"Job ID: {int(env_information.job_id)}")
    except RuntimeError:
        log.info("Local job")

    # set seed for random number generators in pytorch, numpy and python.random
    if cfg.get("seed"):
        pl.seed_everything(cfg.seed, workers=True)

    log.info(f"Instantiating datamodule <{cfg.datamodule._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.datamodule)

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model)

    log.info("Instantiating callbacks...")
    callbacks: List[Callback] = utils.instantiate_callbacks(cfg.get("callbacks"))

    log.info("Instantiating loggers...")
    logger: List[Logger] = utils.instantiate_loggers(cfg.get("logger"))

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
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

   
    # look for latest checkpoint in logdir and load it if found
    if(cfg.get("ckpt_path") is None):
        path_tmp = cfg.configs.storage_folder + "/checkpoints/last.ckpt"
        path_tmp_ema = cfg.configs.storage_folder + "/checkpoints/last-EMA.ckpt"
        if(os.path.exists(path_tmp_ema)):
            checkpoint_path = path_tmp_ema
            log.info("Loading weights from last EMA ckpt " + checkpoint_path)
        elif(os.path.exists(path_tmp)):
            checkpoint_path = path_tmp
            log.info("Loading weights from last ckpt " + checkpoint_path)
        else:
            if(cfg.configs.weights_path is not None):
                if os.path.exists(cfg.configs.weights_path):
                    try:
                        av = torch.load(cfg.configs.weights_path, map_location=torch.device('cpu'))['state_dict']
                    except:
                        av = torch.load(cfg.configs.weights_path, map_location=torch.device('cpu'))['model']
                        av = {"encoder."+k: v for k, v in av.items()}

                    # if loading from torch.compile model
                    av = {k.replace("._orig_mod", ""): v for k, v in av.items()}

                    model.load_state_dict(av, strict=cfg.configs.load_strict)
                    log.info("Loading weights from weights " + cfg.configs.weights_path)
            checkpoint_path = None
    else:
        checkpoint_path = cfg.get("ckpt_path")
        if not os.path.exists(checkpoint_path):
            checkpoint_path = None
        else:
            log.info("Loading weights from ckpt " + checkpoint_path)
    
    if cfg.get("train"):
        log.info("Starting training!")
        trainer.fit(model=model, datamodule=datamodule, ckpt_path=checkpoint_path)

    train_metrics = trainer.callback_metrics

    if cfg.get("test"):
        log.info("Starting testing!")
        ckpt_path = cfg.get("ckpt_path")
        if ckpt_path == "":
            log.warning("Best ckpt not found! Using current weights for testing...")
            ckpt_path = None
        trainer.validate(model=model, datamodule=datamodule, ckpt_path=ckpt_path)
        log.info(f"Best ckpt path: {ckpt_path}")

    test_metrics = trainer.callback_metrics

    # merge train and test metrics
    metric_dict = {**train_metrics, **test_metrics}

    return metric_dict, object_dict


@hydra.main(version_base="1.2", config_path="../configs", config_name="train.yaml")
def main(cfg: DictConfig) -> Optional[float]:

    # train the model
    metric_dict, _ = train(cfg)

    # safely retrieve metric value for hydra-based hyperparameter optimization
    metric_value = utils.get_metric_value(
        metric_dict=metric_dict, metric_name=cfg.get("optimized_metric")
    )

    # return optimized metric
    return metric_value


if __name__ == "__main__":
    main()
