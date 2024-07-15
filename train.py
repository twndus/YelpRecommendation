import hydra
from omegaconf import OmegaConf

import pytz
from datetime import datetime
from easydict import EasyDict

import wandb
from torch.utils.data import DataLoader

from loguru import logger

from data.datasets.cdae_data_pipeline import CDAEDataPipeline
from data.datasets.mf_data_pipeline import MFDataPipeline
from data.datasets.dcn_data_pipeline import DCNDatapipeline
from data.datasets.ngcf_data_pipeline import NGCFDataPipeline
from data.datasets.s3rec_data_pipeline import S3RecDataPipeline
from data.datasets.cdae_dataset import CDAEDataset
from data.datasets.mf_dataset import MFDataset
from data.datasets.dcn_dataset import DCNDataset
from data.datasets.s3rec_dataset import S3RecDataset
from trainers.cdae_trainer import CDAETrainer
from trainers.dcn_trainer import DCNTrainer
from trainers.mf_trainer import MFTrainer
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

def run(cfg, args):#train_dataset, valid_dataset, test_dataset, model_info):
    set_seed(cfg.seed)
    init_wandb_if_needed(cfg)
    train(cfg, args)#train_dataset, valid_dataset, test_dataset, model_info)
    finish_wandb_if_needed(cfg)

def run_sweep(cfg, args):
    sweep_id = wandb.sweep(sweep=OmegaConf.to_container(cfg, resolve=True), project=cfg.project)
    wandb.agent(sweep_id,
                function=lambda: sweep(cfg, args),
                count=cfg.sweep_count)

def sweep(cfg, args):# *datasets):
    set_seed(cfg.seed)
    init_wandb_if_needed(cfg)
    update_config_hyperparameters(cfg)
    train(cfg, args)#*datasets)

def train(cfg, args):#train_dataset, valid_dataset, test_dataset, model_info):
    # set dataloaders
    train_dataloader = DataLoader(args.train_dataset, batch_size=cfg.batch_size, shuffle=cfg.shuffle)
    valid_dataloader = DataLoader(args.valid_dataset, batch_size=cfg.batch_size, shuffle=cfg.shuffle)

    if cfg.model_name not in ('MF', 'DCN', 'NGCF'): 
        test_dataloader = DataLoader(args.test_dataset, batch_size=cfg.batch_size)

    if cfg.model_name in ('CDAE', ):
        trainer = CDAETrainer(cfg, args.model_info['num_items'], args.model_info['num_users'])
        trainer.run(train_dataloader, valid_dataloader)
        trainer.load_best_model()
        trainer.evaluate(test_dataloader)
    elif cfg.model_name in ('MF', ):
        trainer = MFTrainer(cfg, args.model_info['num_items'], args.model_info['num_users'])
        trainer.run(train_dataloader, valid_dataloader, args.valid_eval_data)
        trainer.load_best_model()
        trainer.evaluate(args.test_eval_data, 'test')
    elif cfg.model_name in ('DCN', ):
        trainer = DCNTrainer(cfg, args.model_info['num_items'], args.model_info['num_users'],
                                args.data_pipeline.item2attributes, args.data_pipeline.attributes_count)
        trainer.run(train_dataloader, valid_dataloader, args.valid_eval_data)
        trainer.load_best_model()
        trainer.evaluate(args.test_eval_data, 'test')
    elif cfg.model_name in ('NGCF', ):
        trainer = MGCFTrainer(cfg, args.model_info['num_items'], args.model_info['num_users'], 
                                args.data_pipeline.laplacian_matrix)
        trainer.run(train_dataloader, valid_dataloader, args.valid_eval_data)
        trainer.load_best_model()
        trainer.evaluate(args.test_eval_data, 'test')

def unpack_model(cfg: OmegaConf) -> OmegaConf:
    if cfg.model_name not in cfg.model:
        raise ValueError(f"model '{cfg.model_name}' is not defined in train_config.yaml")
    
    logger.info(f"[CONFIG] Model '{cfg.model_name}' is set to train...")
    model_cfg = cfg.model[cfg.model_name]
    del cfg.model

    merged = OmegaConf.create({})
    merged.update(cfg)
    merged.update(model_cfg)
    return merged

@hydra.main(version_base=None, config_path="configs", config_name="train_config")
def main(cfg: OmegaConf):
    cfg = unpack_model(cfg)
    
    if cfg.model_name in ('CDAE', ):
        data_pipeline = CDAEDataPipeline(cfg)
    elif cfg.model_name == 'MF':
        data_pipeline = MFDataPipeline(cfg)
    elif cfg.model_name == 'DCN':
        data_pipeline = DCNDatapipeline(cfg)
    elif cfg.model_name == 'NGCF':
        data_pipeline = NGCFDataPipeline(cfg)
    elif cfg.model_name == 'S3Rec':
        data_pipeline = S3RecDataPipeline(cfg)
    else:
        raise ValueError()

    df = data_pipeline.preprocess()

    args = EasyDict()
    args.update({'data_pipeline': data_pipeline})

    model_info = dict() # additional infos needed to create model object
    if cfg.model_name in ('CDAE', ):
        train_data, valid_data, test_data = data_pipeline.split(df)
        train_dataset = CDAEDataset(train_data, 'train', neg_times=cfg.neg_times)
        valid_dataset = CDAEDataset(valid_data, 'valid', neg_times=cfg.neg_times)
        test_dataset = CDAEDataset(test_data, 'test')
        args.update({'test_dataset': test_dataset})
        model_info['num_items'], model_info['num_users']  = len(df.columns)-1, len(train_data)
    elif cfg.model_name == 'MF':
        train_data, valid_data, valid_eval_data, test_eval_data = data_pipeline.split(df)
        train_dataset = MFDataset(train_data, num_items=data_pipeline.num_items)
        valid_dataset = MFDataset(valid_data, num_items=data_pipeline.num_items)
        args.update({'valid_eval_data': valid_eval_data, 'test_eval_data': test_eval_data})
        model_info['num_items'], model_info['num_users']  = data_pipeline.num_items, data_pipeline.num_users
    elif cfg.model_name == 'DCN':
        train_data, valid_data, valid_eval_data, test_eval_data = data_pipeline.split(df)
        train_dataset = DCNDataset(train_data, num_items=data_pipeline.num_items)
        valid_dataset = DCNDataset(valid_data, num_items=data_pipeline.num_items)
        args.update({'valid_eval_data': valid_eval_data, 'test_eval_data': test_eval_data})
        model_info['num_items'], model_info['num_users']  = data_pipeline.num_items, data_pipeline.num_users
    elif cfg.model_name == 'NGCF':
        train_data, valid_data, valid_eval_data, test_eval_data = data_pipeline.split(df)
        train_dataset = MFDataset(train_data, num_items=data_pipeline.num_items)
        valid_dataset = MFDataset(valid_data, num_items=data_pipeline.num_items)
        args.update({'valid_eval_data': valid_eval_data, 'test_eval_data': test_eval_data})
        model_info['num_items'], model_info['num_users']  = data_pipeline.num_items, data_pipeline.num_users
    elif cfg.model_name == 'S3Rec':
        train_data, valid_data, test_data = data_pipeline.split(df)
        train_dataset = S3RecDataset(train_data, num_items=data_pipeline.num_items)
        valid_dataset = S3RecDataset(valid_data, num_items=data_pipeline.num_items)
        test_dataset = S3RecDataset(test_data, num_items=data_pipeline.num_items, train=False)
        args.update({'test_dataset': test_data})
        model_info['num_items'], model_info['num_users']  = data_pipeline.num_items, data_pipeline.num_users
    else:
        raise ValueError()

    args.update({
        'train_dataset': train_dataset, 
        'valid_dataset': valid_dataset, 
        'model_info': model_info,
    })

    if cfg.wandb and cfg.sweep:
        sweep_cfg = OmegaConf.load(f'configs/{cfg.model_name.lower()}_sweep_config.yaml')
        merge_cfg = OmegaConf.create({})
        merge_cfg.update(cfg)
        merge_cfg.update(sweep_cfg)
        run_sweep(merge_cfg, args)
    else:
        run(cfg, args)

if __name__ == '__main__':
    main()
