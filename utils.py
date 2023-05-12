import pandas as pd
import heapq
import numpy as np


def train_val_split(target_event: pd.DataFrame, seed=2023, frac=0.5):
    # 将target event划分验证集与训练集
    val = target_event.sample(frac=frac, random_state=seed)
    train = target_event.drop(labels=val.index)
    val = val.reset_index(drop=True)
    train = train.reset_index(drop=True)
    return train, val


def to_one_hot(x, mx):
    # example: to_one_hot([0,2], 3) = [1, 0, 1, 0]
    oh = [0] * (mx + 1)
    for idx in x:
        oh[idx] = 1
    return oh


def show_graph_info(g):
    """
    对图简单描述
    """
    print("graph info:")
    print(f"number of nodes: {g.num_nodes()}")
    print(f"number of edges: {g.num_edges()}")


def mean_reciprocal_rank(eva_prediction):
    """
    :param eva_prediction: inviter_id,event_id,voter_id,candidate_voter_list
    :return:
    """
    mrr = []
    rank_score = [1, 1 / 2, 1 / 3, 1 / 4, 1 / 5]
    candidate_voter_list = eva_prediction['candidate_voter_list'].tolist()
    true_voter_list = eva_prediction['voter_list'].tolist()
    for i in range(eva_prediction.shape[0]):
        candidate_voter, true_voter = candidate_voter_list[i], true_voter_list[i]
        is_right = [1 if k in true_voter else 0 for k in candidate_voter]
        mrr.append(np.multiply(is_right, rank_score).mean())
    return np.mean(mrr)








