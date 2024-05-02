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
    recall_sum: float = .0
    num_users: int = len(actual)

    for user_idx in range(num_users):
        actual_set: set = set(actual[user_idx])
        predicted_set: set = set(predicted[user_idx][:k])
        true_positive_count: int = len(actual_set & predicted_set)
        user_recall: float = true_positive_count / len(actual_set)

        recall_sum += user_recall

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
    ap_k_sum: float = .0
    num_users: int = len(actual)

    for user_idx in range(num_users):
        user_actual: np.ndarray = actual[user_idx]
        user_predicted: np.ndarray = predicted[user_idx]

        ap_k: float = _average_precision_at_k(user_actual, user_predicted, k)
        
        ap_k_sum += ap_k

    return ap_k_sum / num_users

def _average_precision_at_k(user_actual: np.ndarray, user_predicted: np.ndarray, k:int) -> float:
    precision_sum: float= .0
    for i in range(1, k+1):
        actual_set: set = set(user_actual[:i])
        predicted_set: set = set(user_predicted[:i])

        precision_at_i: float = len(actual_set & predicted_set) / i
        rel_i: int = 1 if user_actual[i-1] == user_predicted[i-1] else 0

        precision_sum += (precision_at_i * rel_i)
    
    return precision_sum / len(user_actual)

def ndcg_ag_k(actual: np.ndarray, predicted: np.ndarray, k: int=20) -> float:
    '''
    description: calculate mean ndcg@K of all users
        NDCG@K = ( / K)
    input:
        actual.shape = (num_users, num_items)
        predicted.shape = (num_users, num_items)
    output:
        mean NDCG@K or all users: float
    '''
    logger.info(f"Calculate NDCG@K...")
    ndcg_sum: float = .0
    num_users: int = len(actual)

    return ndcg_sum / num_users
