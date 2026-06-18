"""Main script to run training or inference on malnutrition dataset.

Author: Adrien Brunel <adrien.brunel1@umontpellier.fr>
Adapted from: examples/RLS/rls_aus_reg.py
"""

# ==================================================== #
# LIBRARIES
# ==================================================== #
from pathlib import Path
from typing import Mapping, Optional, Union
import copy

import hydra

from malpolon.data.data_module import RLSDataModule
from malpolon.logging import Summary
from malpolon.models.custom_models import MultiModalModel
from malpolon.models.standard_prediction_systems import GenericPredictionSystem
from malpolon.models.utils import check_metric

from omegaconf import DictConfig, OmegaConf

import lightning.pytorch as pl
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint

import torch
import torchmetrics.functional as Fmetrics

import numpy as np

# from deep_utils import *

OmegaConf.register_new_resolver("eval", eval)
    

# ==================================================== #
# CLASS DEFINITIONS
# ==================================================== #
class AbundanceSystem(GenericPredictionSystem):
    def __init__(
        self,
        submodels: DictConfig,
        num_species: int,
        aggregator: str = 'MLP',
        freeze_submodels: bool = False,
        loss: Union[torch.nn.modules.loss._Loss, str] = 'filtered_huber_loss',
        optimizer: Union[torch.nn.Module, Mapping] = None,
        metrics: Optional[Mapping] = None,
        data_sizes: Optional[Mapping] = None,
        model = None
    ):
        
        if model is None:
            model = MultiModalModel(
                submodels,
                num_species,
                1,
                aggregator,
                freeze_submodels,
                data_sizes=data_sizes
            )

        # Loss and metrics
        self.metrics = check_metric(metrics)
        super().__init__(model, loss, optimizer, metrics=self.metrics)

        self.model = model
        self.mae_decoder = False
            

def dictMSE(predictions, targets):

        loss = 0
        
        for mod in targets:
            loss += Fmetrics.mean_squared_error(predictions[mod].flatten(),
                                                targets[mod].flatten())
        
        return loss / len(targets)
    

# ==================================================== #
# HYDRA MAIN
# ==================================================== #    
@hydra.main(version_base="1.3", config_path="config", config_name="rls_aus_reg")
def main(cfg: DictConfig) -> None:

    pl.seed_everything(cfg.run.seed, workers=True)

    torch.set_float32_matmul_precision('high')


    log_dir = cfg.loggers.log_dir_name
    logger_csv = pl.loggers.CSVLogger(log_dir, name=cfg.run.run_name, version="")
    logger_csv.log_hyperparams(cfg)
    logger_tb = pl.loggers.TensorBoardLogger(log_dir, name=cfg.run.run_name, version="", default_hp_metric=False)
    logger_tb.log_hyperparams(cfg)

    # Datamodule & Model
    datamodule = RLSDataModule(**cfg.data,
                               modality_names= list(cfg.model.submodels.keys()),
                               target_transform=None)
    
    reg_system = AbundanceSystem(**cfg.model, **cfg.optim, data_sizes = datamodule.get_data_sizes())
    
                                    
    # Lightning Trainer                         
    callbacks = [
        Summary(),
        ModelCheckpoint(
            dirpath=hydra.core.hydra_config.HydraConfig.get().runtime.output_dir,
            filename="checkpoint-{epoch:02d}-{step}-{" + next(iter(reg_system.metrics)) + "/val:.4f}",
            monitor=next(iter(reg_system.metrics)) + "/val",
            mode="max",
            save_on_train_epoch_end=True,
            save_last=True,
            auto_insert_metric_name=False
        ),
        LearningRateMonitor(logging_interval='step')
    ]

    trainer = pl.Trainer(logger=[logger_csv, logger_tb], callbacks=callbacks, **cfg.trainer)

    # Training / Inference
    trainer.fit(reg_system, datamodule=datamodule)
    trainer.validate(reg_system, datamodule=datamodule)


if __name__ == "__main__":
    main()
