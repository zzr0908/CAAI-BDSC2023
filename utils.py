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


def to_one_hot(x, size):
    # example: to_one_hot([0,2], 4) = [1, 0, 1, 0]
    oh = [0] * size
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


def flatten_tensor_list(l):
    """
    用于展开dgl bfs search的结果
    """
    flatted = []
    for item in l:
        flatted += item.numpy().tolist()
    return flatted






