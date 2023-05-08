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