import numpy as np

from loguru import logger

def precision_at_k(actual: np.ndarray, predicted: np.ndarray, k: int=20) -> float:
    '''
    input:
        actual.shape = (num_users, num_items)
        predicted.shape = (num_users, num_items)
    output:
        mean precision at k of all users: float
    '''
    logger.info(f"Calculate precision at k...")
    precision_sum: float = .0
    num_users: int = len(actual)

    for user_idx in range(num_users):
        actual_set: set = set(actual[user_idx]) # user's total items
        predicted_set: set = set(predicted[user_idx][:k]) # user's predicted top-k items
        correct_count: int = len(actual_set & predicted_set)
        user_precision: float = correct_count / k

        precision_sum += user_precision

    return precision_sum / num_users

def recall_at_k():
    logger.info(f"Calculate precision at k...")
    pass

def ndcg_ag_k():
    logger.info(f"Calculate precision at k...")
    pass

def map_at_k():
    logger.info(f"Calculate precision at k...")
    pass
