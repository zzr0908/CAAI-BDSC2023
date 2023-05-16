import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn
import torch


class DotPredictor(nn.Module):
    def forward(self, g, h):
        with g.local_scope():
            g.ndata["h"] = h
            g.apply_edges(fn.u_dot_v("h", "h", "score"))
            return g.edata["score"][:, 0]


class EdgeClassifyHead(nn.Module):
    """
    边的多分类任务中的分类头
    :param:
        emd: 模型输出层embedding长度
        c: 类别数量
    """
    def __init__(self, emd, out):
        super(EdgeClassifyHead, self).__init__()
        # A linear submodule for projecting the input and neighbor feature to the output.
        self.out = out
        self.src_linear = nn.Linear(emd, out)
        self.dst_linear = nn.Linear(emd, out)

    def forward(self, g, h):
        with g.local_scope():
            g.ndata["src_emd"] = self.src_linear(h)
            g.ndata["dst_emd"] = self.dst_linear(h)
            g.apply_edges(fn.u_add_v("src_emd", "dst_emd", "score"))
            return F.sigmoid(g.edata["score"][:, :self.out])


import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn
import torch


class FcLinkPredictor(nn.Module):
    """
    用fc层映射起点与终点向量到a，将边类型通过embedding层映射到b,然后计算向量内积dot(a, b)
    input = (h_src, h_dst, edge_type)
    output = score
    """
    def __init__(self, node_feat, hidden_feat, edge_type):
        super(FcLinkPredictor, self).__init__()
        self.node_linear = nn.Linear(2*node_feat, hidden_feat)
        self.out_linear = nn.Linear(hidden_feat, edge_type)

    def forward(self, u_feat, v_feat):    # edge_type 是shape=1 的tensor, 与模型在相同device
        node_feat = self.node_linear(torch.cat((u_feat, v_feat), 1))
        node_feat = F.relu(node_feat)
        score = self.out_linear(node_feat)
        return F.sigmoid(score)
