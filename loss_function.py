import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn
from sklearn.metrics import roc_auc_score


# def entropy_loss(pos_score, neg_score):
#     # 二分类交叉熵
#     score = torch.cat([pos_score, neg_score])
#     label = torch.cat(
#         [torch.ones_like(pos_score), torch.zeros_like(neg_score)]
#     )
#     loss = F.binary_cross_entropy_with_logits(score, label)
#     return loss

# def value_of_auc(pos_score, neg_score):
#     # 计算auc
#     score = torch.cat([pos_score, neg_score])
#     label = torch.cat(
#         [torch.ones_like(pos_score), torch.zeros_like(neg_score)]
#     )
#     auc = roc_auc_score(y_true=label , y_score=score)
#     return auc

def auc_surrogate_loss(pos_score, neg_score):
    # auc替代损失函数
    pos_score = pos_score   # 正样本预测为正的概率
    neg_score = neg_score - 1   # 负样本预测为正的概率
    loss = 0
    # pos_score = 1 - pos_score + neg_score
    # loss = torch.dot(pos_score, pos_score)
    for i in range(pos_score.shape[0]):
        for j in range(neg_score.shape[0]):
            loss = loss + (1 - pos_score[i] + neg_score[j]) ** 2
    return loss

# def auc_surrogate_loss(pos_score, neg_score):
#     # auc替代损失函数
#     pos_score = pos_score   # 正样本预测为正的概率
#     neg_score = neg_score - 1   # 负样本预测为正的概率
#     loss = 0
#     for i in range(pos_score.shape[0]):
#         for j in range(neg_score.shape[0]):
#             loss = loss + (max(0, 1 - pos_score[i] + neg_score[j])) ** 2
#     return loss
