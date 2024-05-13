import pytz
from datetime import datetime

import wandb

from torch.utils.data import DataLoader

from loguru import logger

from omegaconf import OmegaConf

from data.datasets.cdae_data_pipeline import CDAEDataPipeline
from data.datasets.cdae_dataset import CDAEDataset
from trainers.cdae_trainer import CDAETrainer
from utils import set_seed


cfg: OmegaConf = None
num_items = 0
num_users = 0

def main():
    logger.info(f"set seed as {cfg.seed}...")
    set_seed(cfg.seed)
    
    logger.info("[wandb] init...")
    run_time: str = datetime.now().astimezone(pytz.timezone('Asia/Seoul')).strftime('%Y-%m-%d %H:%M:%S')
    run_name: str = f'[Sweep][{cfg.model_name}]{run_time}'
    
    wandb.init(
        project=cfg.project,
        name=run_name,
        config=dict(cfg),
        notes=cfg.notes,
        tags=cfg.tags,
    )

    # update hyperparameters to selected values
    logger.info("[Sweep] Update hyper-parameters...")
    for parameter in cfg.parameters:
        cfg[parameter] = wandb.config[parameter]
        logger.info(f"[{parameter}] {cfg.lr}")

    if cfg.model_name in ('CDAE',):
        trainer = CDAETrainer(cfg, num_items, num_users)
    elif cfg.model_name in ('WDN', ):
        trainer = CDAETrainer(cfg, num_items, num_users)

    train_dataloader = DataLoader(train_data, batch_size=cfg.batch_size, shuffle=True)
    valid_dataloader = DataLoader(valid_data, batch_size=cfg.batch_size, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=cfg.batch_size)

    trainer.run(train_dataloader, valid_dataloader)
    trainer.load_best_model()
    trainer.evaluate(test_dataloader)


if __name__ == '__main__':
    cfg = OmegaConf.load('./configs/sweep_config.yaml')

    data_pipeline = CDAEDataPipeline(cfg)
    df = data_pipeline.preprocess()
    train_data, valid_data, test_data = data_pipeline.split(df)

    train_data = CDAEDataset(train_data, 'train')
    valid_data = CDAEDataset(valid_data, 'valid')
    test_data = CDAEDataset(test_data, 'test')

    num_items = len(df.columns) - 1
    num_users = len(train_data)

    sweep_id = wandb.sweep(sweep=OmegaConf.to_container(cfg, resolve=True), project=cfg.project)
    wandb.agent(sweep_id, function=main, count=cfg.sweep_count)
