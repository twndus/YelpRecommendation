import hydra
from omegaconf import OmegaConf

from data.datasets.cdae_data_pipeline import CDAEDataPipeline
from data.datasets.cdae_dataset import CDAEDataset
from trainers.base_trainer import BaseTrainer
from utils import set_seed

import torch
from torch.utils.data import DataLoader

from loguru import logger

@hydra.main(version_base=None, config_path="configs", config_name="train_config")
def main(cfg: OmegaConf):
    logger.info(f"set seed as {cfg.seed}...")
    set_seed(cfg.seed)
    
    if cfg.model_name in ('CDAE', ):
        data_pipeline = CDAEDataPipeline(cfg)
    else:
        raise ValueError()

    df = data_pipeline.preprocess()
    train_data, valid_data, test_data = data_pipeline.split(df)

    if cfg.model_name in ('CDAE', ):
        train_dataset = CDAEDataset(train_data, 'train')
        valid_dataset = CDAEDataset(valid_data, 'valid')
        test_dataset = CDAEDataset(test_data, 'test')
    else:
        raise ValueError()

    # pos_samples 를 이용한 negative sample을 수행해줘야 함
    train_dataloader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=cfg.shuffle)
    valid_dataloader = DataLoader(valid_dataset, batch_size=cfg.batch_size, shuffle=cfg.shuffle)
    test_dataloader = DataLoader(test_dataset, batch_size=cfg.batch_size)

#    trainer = BaseTrainer(cfg)
#    trainer.run(train_dataloader, valid_dataloader)
#    trainer.evaluate(test_dataloader)

if __name__ == '__main__':
    main()
