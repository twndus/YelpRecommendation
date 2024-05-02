import hydra
from omegaconf import OmegaConf

from .data.datasets.data_pipeline import DataPipeline
from .trainers.base_trainer import BaseTrainer
from .utils import set_seed

import torch
from torch.utils.data import Dataset, DataLoader


@hydra.main(version_base=None, config_path="../config", config_name="train_config")
def main(cfg: OmegaConf):
    set_seed(42)
    
    data_pipeline = DataPipeline()
    data_pipeline.preprocess()
    train_data, valid_data, test_data = data_pipeline.split()

    train_dataset = Dataset(train_data)
    valid_dataset = Dataset(valid_data)
    test_dataset = Dataset(test_data)

    train_dataloader = DataLoader(train_dataset, shuffle=cfg.shuffle)
    valid_dataloader = DataLoader(valid_dataset, shuffle=cfg.shuffle)
    test_dataloader = DataLoader(test_dataset, shuffle=cfg.shuffle)

    trainer = BaseTrainer()
    trainer.run(train_dataloader, valid_dataloader)
    trainer.evaluate(test_dataloader)

if __name__ == '__main__':
    main()
