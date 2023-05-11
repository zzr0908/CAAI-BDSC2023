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
            return g.edata["score"][:, :self.out]


class SAGEConv(nn.Module):
    """Graph convolution module used by the GraphSAGE model.

    Parameters
    ----------
    in_feat : int
        Input feature size.
    out_feat : int
        Output feature size.
    """

    def __init__(self, in_feat, out_feat):
        super(SAGEConv, self).__init__()
        # A linear submodule for projecting the input and neighbor feature to the output.
        self.linear = nn.Linear(in_feat * 2, out_feat)

    def forward(self, g, h):
        """Forward computation

        Parameters
        ----------
        g : Graph
            The input MFG.
        h : (Tensor, Tensor)
            The feature of source nodes and destination nodes as a pair of Tensors.
        """
        with g.local_scope():
            h_src, h_dst = h
            g.srcdata["h"] = h_src  # <---
            g.dstdata["h"] = h_dst  # <---
            # update_all is a message passing API.
            g.update_all(fn.copy_u("h", "m"), fn.mean("m", "h_N"))
            h_N = g.dstdata["h_N"]
            h_total = torch.cat([h_dst, h_N], dim=1)  # <---
            return self.linear(h_total)


class SageModel(nn.Module):
    def __init__(self, in_feats, h_feats, num_classes):
        super(SageModel, self).__init__()
        self.conv1 = SAGEConv(in_feats, h_feats)
        self.conv2 = SAGEConv(h_feats, num_classes)

    def forward(self, mfgs, x):
        h_dst = x[: mfgs[0].num_dst_nodes()]
        h = self.conv1(mfgs[0], (x, h_dst))
        h = F.relu(h)
        h_dst = h[: mfgs[1].num_dst_nodes()]
        h = self.conv2(mfgs[1], (h, h_dst))
        return h


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
        return score



