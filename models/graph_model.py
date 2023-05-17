import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn
import torch


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
        self.linear = nn.Linear(in_feat * 2, out_feat)

    def forward(self, g, h_src, h_dst):
        with g.local_scope():
            g.srcdata["h"] = h_src
            g.dstdata["h"] = h_dst
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
