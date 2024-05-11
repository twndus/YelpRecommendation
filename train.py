import hydra
from omegaconf import OmegaConf

from data.datasets.cdae_data_pipeline import CDAEDataPipeline
from data.datasets.mf_data_pipeline import MFDataPipeline
from data.datasets.cdae_dataset import CDAEDataset
from data.datasets.mf_dataset import MFDataset
from trainers.cdae_trainer import CDAETrainer
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
    elif cfg.model_name == 'MF':
        data_pipeline = MFDataPipeline(cfg)
    else:
        raise ValueError()

    df = data_pipeline.preprocess()
    train_data, valid_data, test_data = data_pipeline.split(df)

    if cfg.model_name in ('CDAE', ):
        train_dataset = CDAEDataset(train_data, 'train', neg_times=cfg.neg_times)
        valid_dataset = CDAEDataset(valid_data, 'valid', neg_times=cfg.neg_times)
        test_dataset = CDAEDataset(test_data, 'test')
    elif cfg.model_name == 'MF':
        train_dataset = MFDataset(train_data, 'train', num_items=data_pipeline.num_items)
        valid_dataset = MFDataset(valid_data, 'valid', num_items=data_pipeline.num_items)
        test_dataset = MFDataset(test_data, 'test')
    else:
        raise ValueError()

    # pos_samples 를 이용한 negative sample을 수행해줘야 함
    train_dataloader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=cfg.shuffle)
    valid_dataloader = DataLoader(valid_dataset, batch_size=cfg.batch_size, shuffle=cfg.shuffle)
    test_dataloader = DataLoader(test_dataset, batch_size=cfg.batch_size)

    if cfg.model_name in ('CDAE', ):
        trainer = CDAETrainer(cfg, len(df.columns)-1, len(train_dataset))
        trainer.run(train_dataloader, valid_dataloader)
        trainer.load_best_model()
        trainer.evaluate(test_dataloader)

if __name__ == '__main__':
    main()
