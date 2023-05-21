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

    def __init__(self, in_feat, out_feat, comp_fn):
        super(SAGEConv, self).__init__()
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.comp_fn = comp_fn
        # A linear submodule for projecting the input and neighbor feature to the output.
        self.linearI = nn.Linear(in_feat, out_feat)   # 正向边
        self.linearO = nn.Linear(in_feat, out_feat)   # 逆向边
        self.linearR = nn.Linear(in_feat, out_feat)   # 关系

    def forward(self, g, h, e):
        """Forward computation
        Parameters
        ----------
        g : Graph
            The input MFG.
        h : (Tensor, Tensor)
            The feature of source nodes and destination nodes as a pair of Tensors.
        e :
        """
        with g.local_scope():
            h_src, h_dst = h
            reverse_edge, forward_edge = e
            g.srcdata["h"] = h_src  # <---
            g.dstdata["h"] = h_dst
            g.edata["h"] = e

            # Step 1: compute composition by edge in the edge direction, and store results in edges.
            if self.comp_fn == "sub":
                g.apply_edges(fn.u_sub_e("h", "h", out="comp_h"))
            elif self.comp_fn == "mul":
                g.apply_edges(fn.u_mul_e("h", "h", out="comp_h"))
            elif self.comp_fn == "ccorr":
                g.apply_edges(
                    lambda edges: {
                        "comp_h": ccorr(edges.src["h"], edges.data["h"])
                    }
                )
            else:
                raise Exception("Only supports sub, mul, and ccorr")

            # Step 2: use extracted edge direction to compute in and out edges
            comp_h = g.edata["comp_h"]

            forward_edges_idx = torch.nonzero(
                g.edata["source_forward_mask"], as_tuple=False
            ).squeeze()
            reverse_edges_idx = torch.nonzero(
                g.edata["source_reverse_mask"], as_tuple=False
            ).squeeze()

            comp_h_I = self.LinearI(comp_h[forward_edges_idx])
            comp_h_O = self.LinearO(comp_h[reverse_edges_idx])

            new_comp_h = torch.zeros(comp_h.shape[0], self.out_feat)
            new_comp_h[reverse_edges_idx] = comp_h_O
            new_comp_h[forward_edges_idx] = comp_h_I
            g.edata["new_comp_h"] = new_comp_h

            # Step 3: sum comp results to both src and dst nodes
            g.update_all(fn.copy_e("new_comp_h", "m"), fn.sum("m", "comp_edge"))
            n_output = g.ndata["comp_edge"]

            # Compute relation output
            e_output = self.LinearR(e)
            return n_output, e_output

            # update_all is a message passing API.
            # g.update_all(fn.copy_u("h", "m"), fn.mean("m", "h_N"))
            # h_N = g.dstdata["h_N"]
            # h_total = torch.cat([h_dst, h_N], dim=1)  # <---
            # return self.linear(h_total)


class SageModel(nn.Module):
    def __init__(self, in_feats, h_feats, num_classes, comp_fn):
        super(SageModel, self).__init__()
        self.conv1 = SAGEConv(in_feats, h_feats)
        self.conv2 = SAGEConv(h_feats, num_classes)
        self.comp_fn = comp_fn

    def forward(self, mfgs, x, e):
        h_dst = x[: mfgs[0].num_dst_nodes()]
        h, e = self.conv1(mfgs[0], (x, h_dst), e, self.comp_fn)
        h = F.relu(h)
        h_dst = h[: mfgs[1].num_dst_nodes()]
        h, e = self.conv2(mfgs[1], (h, h_dst), e, self.comp_fn)
        return h
