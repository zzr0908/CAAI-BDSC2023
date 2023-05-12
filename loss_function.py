import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn


def entropy_loss(pos_score, neg_score):
    # 二分类交叉熵
    score = torch.cat([pos_score, neg_score])
    label = torch.cat(
        [torch.ones_like(pos_score), torch.zeros_like(neg_score)]
    )
    loss = F.binary_cross_entropy_with_logits(score, label)
    return loss


def multi_label_loss(score, target):
    """
    用于边分类预训练任务的损失函数
    input, target: (batch size, class num)
    """
    loss = F.binary_cross_entropy_with_logits(score, target)
    return loss


def auc_surrogate_loss(score, label):
    """
        按类别分别计算Loss后取均值
        score: 预测分数，正+负
        label: 样本label
    """
    num_class = score.shape[1]
    score_unbind = torch.unbind(score, dim=1)
    label_unbind = torch.unbind(label, dim=1)
    loss = 0
    for k in range(num_class):
        score_k = score_unbind[k]    # 类别k的预测得分
        label_k = label_unbind[k]    # 类别k的label
        pos_score = score_k * label_k
        neg_score = score_k * (1 - label_k)
        pos_score = pos_score[torch.nonzero(pos_score)]  # 正样本的预测得分
        neg_score = neg_score[torch.nonzero(neg_score)]  # 负样本的预测得分
        loss_tensor = 1 - pos_score + torch.reshape(neg_score, (1, -1))
        loss = loss + torch.mul(loss_tensor, loss_tensor).sum()
    return loss / num_class


