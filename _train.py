import hydra
from omegaconf import OmegaConf

import pytz
from datetime import datetime

import wandb
from torch.utils.data import DataLoader

from loguru import logger

from data.datasets.cdae_data_pipeline import CDAEDataPipeline
from data.datasets.cdae_dataset import CDAEDataset
from trainers.cdae_trainer import CDAETrainer
from utils import set_seed


def init_wandb_if_needed(cfg):
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

def finish_wandb_if_needed(cfg):
    if cfg.wandb:
        logger.info("[wandb] finish...")
        wandb.finish()

def update_config_hyperparameters(cfg):
    logger.info("[Sweep] Update hyper-parameters...")
    for parameter in cfg.parameters:
        cfg[parameter] = wandb.config[parameter]
        logger.info(f"[{parameter}] {cfg[parameter]}")

def run(cfg, train_dataset, valid_dataset, test_dataset, model_info):
    set_seed(cfg.seed)
    init_wandb_if_needed(cfg)
    train(cfg, train_dataset, valid_dataset, test_dataset, model_info)
    finish_wandb_if_needed(cfg)

def run_sweep(cfg, *datasets):
    sweep_id = wandb.sweep(sweep=OmegaConf.to_container(cfg, resolve=True), project=cfg.project)
    wandb.agent(sweep_id,
                function=lambda: sweep(cfg, *datasets),
                count=cfg.sweep_count)

def sweep(cfg, *datasets):
    set_seed(cfg.seed)
    init_wandb_if_needed(cfg)
    update_config_hyperparameters(cfg)
    train(cfg, *datasets)

def train(cfg, train_dataset, valid_dataset, test_dataset, model_info):
    # set dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=cfg.shuffle)
    valid_dataloader = DataLoader(valid_dataset, batch_size=cfg.batch_size, shuffle=cfg.shuffle)
    test_dataloader = DataLoader(test_dataset, batch_size=cfg.batch_size)

    if cfg.model_name in ('CDAE', ):
        trainer = CDAETrainer(cfg, model_info['num_items'], model_info['num_users'])
        trainer.run(train_dataloader, valid_dataloader)
        trainer.load_best_model()
        trainer.evaluate(test_dataloader)

@hydra.main(version_base=None, config_path="configs", config_name="_train_config")
def main(cfg: OmegaConf):
    if cfg.model_name in ('CDAE', ):
        data_pipeline = CDAEDataPipeline(cfg)
    else:
        raise ValueError()

    df = data_pipeline.preprocess()
    train_data, valid_data, test_data = data_pipeline.split(df)

    model_info = dict() # additional infos needed to create model object
    if cfg.model_name in ('CDAE', ):
        train_dataset = CDAEDataset(train_data, 'train')
        valid_dataset = CDAEDataset(valid_data, 'valid')
        test_dataset = CDAEDataset(test_data, 'test')
        model_info['num_items'], model_info['num_users']  = len(df.columns)-1, len(train_data)
    else:
        raise ValueError()
    
    if cfg.wandb and cfg.sweep:
        run_sweep(cfg, train_dataset, valid_dataset, test_dataset, model_info)
    else:
        run(cfg, train_dataset, valid_dataset, test_dataset, model_info)

if __name__ == '__main__':
    main()
