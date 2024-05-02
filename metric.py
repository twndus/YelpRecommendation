import numpy as np

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
    precision_sum: float = .0
    num_users: int = len(actual)

    for user_idx in range(num_users):
        actual_set: set = set(actual[user_idx]) # user's total items
        predicted_set: set = set(predicted[user_idx][:k]) # user's predicted top-k items
        true_positive_count: int = len(actual_set & predicted_set)
        user_precision: float = true_positive_count / k

        precision_sum += user_precision

    return precision_sum / num_users

def recall_at_k(actual: np.ndarray, predicted: np.ndarray, k: int=20):
    '''
    description: calculate mean recall@K of all users
        recall@K = (TP@K / TP + FP)
    input:
        actual.shape = (num_users, num_items)
        predicted.shape = (num_users, num_items)
    output:
        mean recall@K or all users: float
    '''
    logger.info(f"Calculate Recall@K...")
    recall_sum: float = .0
    num_users: int = len(actual)

    for user_idx in range(num_users):
        actual_set: set = set(actual[user_idx])
        predicted_set: set = set(predicted[user_idx][:k])
        true_positive_count: int = len(actual_set & predicted_set)
        user_recall: float = true_positive_count / len(actual_set)

        recall_sum += user_recall

    return recall_sum / num_users

def ndcg_ag_k():
    '''
    description: calculate mean ndcg@K of all users
        NDCG@K = ()
    '''
    logger.info(f"Calculate NDCG@K...")
    pass

def map_at_k():
    logger.info(f"Calculate MAP@K...")
    pass
