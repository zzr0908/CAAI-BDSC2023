import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, accuracy_score, recall_score, roc_auc_score, f1_score


def mean_reciprocal_rank(eva_prediction):
    """
    mrr5 evaluation for link prediction
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


def edge_classification_evaluation(pred: np.array, target: np.array):
    """
    input:
        pred and target shape (samples, class)
    return:
        acc/recall/precision/f1/auc for edge classification
    """
    # 基础信息
    pred = np.around(pred, 0).astype(int)
    mean_target = list(target.mean(axis=0))
    mean_pred = list(pred.mean(axis=0))
    acc = [accuracy_score(target[:, i], pred[:, i]) for i in range(pred.shape[1])]
    recall = [recall_score(target[:, i], pred[:, i]) for i in range(pred.shape[1])]
    precision = [precision_score(target[:, i], pred[:, i]) for i in range(pred.shape[1])]
    f1 = [f1_score(target[:, i], pred[:, i]) for i in range(pred.shape[1])]

    return {
            "mean_target": mean_target,
            "mean_pred": mean_pred,
            "acc": acc,
            "recall": recall,
            "precision": precision,
            "f1": f1
            }


def cal_mrr(true, pred):
    if len(pred) == 0:
        return 0
    for item in true:
        for i in range(len(pred)):
            if pred[i] == item:
                return 1/(i + 1)
    return 0

