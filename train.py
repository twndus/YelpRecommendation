import hydra
from omegaconf import OmegaConf

import pytz
from datetime import datetime

import wandb
import torch
from torch.utils.data import DataLoader

from loguru import logger

from data.datasets.cdae_data_pipeline import CDAEDataPipeline
from data.datasets.mf_data_pipeline import MFDataPipeline
from data.datasets.cdae_dataset import CDAEDataset
from data.datasets.mf_dataset import MFDataset
from trainers.cdae_trainer import CDAETrainer
from trainers.mf_trainer import MFTrainer
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
    elif cfg.model_name == 'MF':
        data_pipeline = MFDataPipeline(cfg)
    else:
        raise ValueError()

    df = data_pipeline.preprocess()

    if cfg.model_name in ('CDAE', ):
        train_data, valid_data, test_data = data_pipeline.split(df)
        train_dataset = CDAEDataset(train_data, 'train', neg_times=cfg.neg_times)
        valid_dataset = CDAEDataset(valid_data, 'valid', neg_times=cfg.neg_times)
        test_dataset = CDAEDataset(test_data, 'test')
    elif cfg.model_name == 'MF':
        train_data, valid_data, valid_eval_data, test_eval_data = data_pipeline.split(df)
        train_dataset = MFDataset(train_data, num_items=data_pipeline.num_items)
        valid_dataset = MFDataset(valid_data, num_items=data_pipeline.num_items)
    else:
        raise ValueError()

    # set dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=cfg.shuffle)
    valid_dataloader = DataLoader(valid_dataset, batch_size=cfg.batch_size, shuffle=cfg.shuffle)

    if cfg.model_name != 'MF': 
        test_dataloader = DataLoader(test_dataset, batch_size=cfg.batch_size)

    if cfg.model_name in ('CDAE', ):
        trainer = CDAETrainer(cfg, len(df.columns)-1, len(train_dataset))
        trainer.run(train_dataloader, valid_dataloader)
        trainer.load_best_model()
        trainer.evaluate(test_dataloader)
    elif cfg.model_name in ('MF', ):
        trainer = MFTrainer(cfg, data_pipeline.num_items, data_pipeline.num_users)
        trainer.run(train_dataloader, valid_dataloader, valid_eval_data)
        trainer.evaluate(test_eval_data, 'test')

    if cfg.wandb:
        logger.info("[wandb] finish...")
        wandb.finish()

if __name__ == '__main__':
    main()
