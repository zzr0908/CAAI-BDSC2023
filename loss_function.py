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

