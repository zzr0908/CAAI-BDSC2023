import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn
from models.embedding import *
from models.graph_model import *
from models.head import *
import torch


class GraphSageModel(nn.Module):
    """
    用graph
    :param:
        node_cnt: 节点总数
        embedding: int embedding层的大小
        hidden: list[int] 每个sage conv 隐藏层大小
        out: 输出的类别数量
        mfgs: 图集合
        nodes: 输入节点id
        feats: 节点的特征
    """
    def __init__(self, node_cnt, embedding, hidden, out):
        super(GraphSageModel, self).__init__()
        self.embedding_layer = NodeEmbedding(node_cnt, embedding)
        hidden = [embedding] + hidden
        self.conv_layers = nn.ModuleList([SAGEConv(hidden[i], hidden[i+1]) for i in range(len(hidden) - 1)])
        self.out_layer = EdgeClassifyHead(hidden[-1], out)

    def forward(self, mfgs, nodes, subgraph):
        h = self.embedding_layer(nodes)    # (batch, mfgs[0].srcdata.num_nodes(), embedding)
        for i in range(len(self.conv_layers)):
            g = mfgs[i]
            h_dst = h[: mfgs[i].num_dst_nodes()]
            h = self.conv_layers[i](g, h, h_dst)
            h = F.relu(h)
        h = self.out_layer(subgraph, h)
        return F.sigmoid(h)




