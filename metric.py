import numpy as np
from math import log2

from loguru import logger


def precision_at_k(actual: np.ndarray, predicted: np.ndarray, k: int=20) -> float:
    '''
    description: calculate mean precision@K of all users
        precision@K = (TP@K / K)
    input:
        actual.shape = (num_users, num_items)
        predicted.shape = (num_users, num_items)
    output:
        mean precision@K of all users: float
    '''
    logger.info(f"Calculate precision at k...")
    num_users: int = len(actual)

    precision_sum: float = sum([
        len(set(actual[user_idx]) & set(predicted[user_idx][:k])) / k for user_idx in range(num_users)
    ])

    return precision_sum / num_users

def recall_at_k(actual: np.ndarray, predicted: np.ndarray, k: int=20) -> float:
    '''
    description: calculate mean recall@K of all users
        recall@K = (TP@K / num_positive)
    input:
        actual.shape = (num_users, num_items)
        predicted.shape = (num_users, num_items)
    output:
        mean recall@K of all users: float
    '''
    logger.info(f"Calculate Recall@K...")
    num_users: int = len(actual)

    recall_sum: float = sum([
        len(set(actual[user_idx]) & set(predicted[user_idx][:k])) / len(set(actual[user_idx])) for user_idx in range(num_users)
    ])

    return recall_sum / num_users

def map_at_k(actual: np.ndarray, predicted: np.ndarray, k: int=20) -> float:
    '''
    description: calculate mean AP@K of all users
        AP@K = (Precision@1 * rel_1 + Precision@2 * rel_2 + ... + Precision@K * rel_k) / (num_positive)
    input:
        actual.shape = (num_users, num_items)
        predicted.shape = (num_users, num_items)
    output:
        mean AP@K of all users: float
    '''
    logger.info(f"Calculate MAP@K...")
    num_users: int = len(actual)
 
    ap_k_sum: float = sum([
        _average_precision_at_k(actual[user_idx], predicted[user_idx], k) for user_idx in range(num_users)
    ])

    return ap_k_sum / num_users

def _average_precision_at_k(user_actual: np.ndarray, user_predicted: np.ndarray, k:int) -> float:
    precision_sum: float = sum([
        len(set(user_actual[:i]) & set(user_predicted[:i])) / i for i in range(1, k+1) if user_actual[i-1] == user_predicted[i-1]
    ])
    
    return precision_sum / len(user_actual)

def ndcg_at_k(actual: np.ndarray, predicted: np.ndarray, k: int=20) -> float:
    '''
    description: calculate mean ndcg@K of all users
        NDCG@K = DCG@K / IDCG@K
        CG@K = (rel_1 + rel_2 + ... + rel_K)
        DCG@K = (rel_1/log_2(2) + rel_2/log_2(3) + ... + rel_K/log_2(K+1))
        IDCG@K = (rel_opt_1/log_2(2) + rel_opt_2/log_2(3) + ... + rel_opt_K/log_2(K+1)) 
            where rel_opt is sorted(rel, reverse=True)
    input:
        actual.shape = (num_users, num_items)
        predicted.shape = (num_users, num_items)
    output:
        mean NDCG@K or all users: float
    '''
    logger.info(f"Calculate NDCG@K...")
    num_users: int = len(actual)

    ndcg_sum: float = sum([
        _dcg_at_k(actual[user_idx], predicted[user_idx], k) / _dcg_at_k(actual[user_idx], actual[user_idx], k) \
            for user_idx in range(num_users)
    ])

    return ndcg_sum / num_users

def _dcg_at_k(user_actual: np.ndarray, user_predicted: np.ndarray, k: int) -> float:
    dcg_sum: float = sum([(1.0 / log2(i+1)) for i in range(1, min(len(user_actual), k)+1) if user_predicted[i-1] in user_actual])

    return dcg_sum
