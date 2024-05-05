import hydra
from omegaconf import OmegaConf

from .data.datasets.data_pipeline import DataPipeline
from .data.datasets.cf_dataset import AEDataset
from .trainers.base_trainer import BaseTrainer
from .utils import set_seed

import torch
from torch.utils.data import DataLoader

from loguru import logger

@hydra.main(version_base=None, config_path="../config", config_name="train_config")
def main(cfg: OmegaConf):
    logger.info(f"set seed as {cfg.seed}...")
    set_seed(cfg.seed)
    
    if cfg.model_name in ('CDAE', ):
        data_pipeline = AEDataPipeline(cfg)
    else:
        raise ValueError()

    data_pipeline.preprocess()
    train_data, valid_data, test_data = data_pipeline.split()

    if cfg.model_name in ('CDAE', ):
        train_dataset = AEDataset(train_data)
        valid_dataset = AEDataset(valid_data)
        test_dataset = AEDataset(test_data)
    else:
        raise ValueError()

    train_dataloader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=cfg.shuffle)
    valid_dataloader = DataLoader(valid_dataset, batch_size=cfg.batch_size, shuffle=cfg.shuffle)
    test_dataloader = DataLoader(test_dataset, batch_size=cfg.batch_size)

    trainer = BaseTrainer(cfg)
    trainer.run(train_dataloader, valid_dataloader)
    trainer.evaluate(test_dataloader)

if __name__ == '__main__':
    main()
