import torch.nn as nn
import torch.nn.functional as F
import torch


class NodeEmbedding(nn.Module):
    """
    边的多分类任务中的分类头
    :param:
        node_cnt: 节点总数
        out_feat: embeddinng 维度
        nodes: (batch, )
    """
    def __init__(self, node_cnt, out_feat):
        super(NodeEmbedding, self).__init__()
        # A linear submodule for projecting the input and neighbor feature to the output.
        self.embedding = nn.Embedding(node_cnt, out_feat)

    def forward(self, nodes: torch.Tensor):     # (batch, )
        output = self.embedding(nodes)     # (batch, out_feat)
        output = F.relu(output)
        return output


class FeatureEmbedding(nn.Module):
    """
    节点本身的特征
    :param:
        feat_cnt: 每个特征的类别总数
        out_feat: embeddinng 维度
        input_feat: (batch, feat_num)
    """
    def __init__(self, feat_cnt, out_feat):
        super(FeatureEmbedding, self).__init__()
        self.acu_feat_cnt = nn.Parameter(torch.tensor([sum(feat_cnt[: i]) for i in range(len(feat_cnt))]),
                                         requires_grad=False)   # (len(feat_cnt), )
        self.embedding = nn.Embedding(sum(feat_cnt), out_feat)

    def forward(self, input_feat):  # (batch, num_feat)
        output = input_feat + self.acu_feat_cnt   # (batch, num_feat)
        output = self.embedding(output)    # (batch, num_feat, out_feat)
        output = output.mean(axis=1)    # (batch, out_feat)
        output = F.relu(output)
        return output


class DenseEmbedding(nn.Module):
    """
    将连续特征投影到embedding空间的embedding层 (两层）
    :param:
        num_feat: 每个特征的类别总数
        out_feat: embeddinng 维度
        input_feat: (batch, num_feat)
    """
    pass


