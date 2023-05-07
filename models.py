import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn
import torch


class DotPredictor(nn.Module):
    def forward(self, g, h):
        with g.local_scope():
            g.ndata["h"] = h
            # Compute a new edge feature named 'score' by a dot-product between the
            # source node feature 'h' and destination node feature 'h'.
            g.apply_edges(fn.u_dot_v("h", "h", "score"))
            # u_dot_v returns a 1-element vector for each edge so you need to squeeze it.
            return g.edata["score"][:, 0]


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
