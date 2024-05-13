import os
import random
import numpy as np
from typing import Callable

import wandb

import torch
from torch.utils.data import DataLoader

from loguru import logger
from functools import wraps

def set_seed(seed: int):
    logger.info(f"[utils] set seed as {seed}...")
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # some cudnn methods can be random even after fixing the seed
    # unless you tell it to be deterministic
    torch.backends.cudnn.deterministic = True


def log_metric(func: Callable):
    @wraps(func)
    def log_wandb(*args, **kwargs):
        precision_at_k, recall_at_k, map_at_k, ndcg_at_k = func(*args, **kwargs)

        if wandb.run is not None: # validate wandb initialization
            logger.info("[Trainer] logging test results...")
            wandb.log({
                'test_Precision@K': precision_at_k,
                'test_Recall@K': recall_at_k,
                'test_MAP@K': map_at_k,
                'test_NDCG@K': ndcg_at_k,
            })
        return (precision_at_k, recall_at_k, map_at_k, ndcg_at_k)
    return log_wandb
