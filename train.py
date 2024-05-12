import hydra
from omegaconf import OmegaConf

import pytz
from datetime import datetime

import wandb
import torch
from torch.utils.data import DataLoader

from loguru import logger

from data.datasets.cdae_data_pipeline import CDAEDataPipeline
from data.datasets.cdae_dataset import CDAEDataset
from trainers.cdae_trainer import CDAETrainer
from utils import set_seed


@hydra.main(version_base=None, config_path="configs", config_name="train_config")
def main(cfg: OmegaConf):

    # wandb init
    if cfg.wandb:
        logger.info("[wandb] init...")
        run_time: str = datetime.now().astimezone(pytz.timezone('Asia/Seoul')).strftime('%Y-%m-%d %H:%M:%S')
        run_name: str = f'[{cfg.model_name}]{run_time}'
        
        wandb.init(
            project=cfg.project,
            name=run_name,
            config=dict(cfg),
            notes=cfg.notes,
            tags=cfg.tags,
        )
    
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

    # set dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=cfg.shuffle)
    valid_dataloader = DataLoader(valid_dataset, batch_size=cfg.batch_size, shuffle=cfg.shuffle)
    test_dataloader = DataLoader(test_dataset, batch_size=cfg.batch_size)

    if cfg.model_name in ('CDAE', ):
        trainer = CDAETrainer(cfg, len(df.columns)-1, len(train_dataset))
        trainer.run(train_dataloader, valid_dataloader)
        trainer.load_best_model()
        trainer.evaluate(test_dataloader)

    if cfg.wandb:
        logger.info("[wandb] finish...")
        wandb.finish()

if __name__ == '__main__':
    main()
