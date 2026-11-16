"""Inductive fusion model: descriptor towers fused with graph embeddings.
- Compound and protein towers encode the standardized node features (inductive; work for any node, connected or
  isolated).
- Graph embeddings from the encoder are projected into the same space.
- A per-node gate (input: tower code, graph code, binary connectivity flag) decides how much graph evidence to add.
  The flag is binary, so it cannot leak the number of known interactions.
- Pair scorer: MLP over [h_c, h_p, h_c*h_p, |h_c-h_p|] plus a bilinear term.
"""

import torch
import torch.nn as nn

from model_connectivity_gate import GNNConnectivityGated


def _tower(in_dim, hidden, out_dim, dropout):
    return nn.Sequential(
        nn.Dropout(0.10),
        nn.Linear(in_dim, hidden),
        nn.LayerNorm(hidden),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(hidden, out_dim),
        nn.LayerNorm(out_dim),
    )


class TrustGatedFusion(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.gate = nn.Sequential(nn.Linear(2 * d + 1, d), nn.GELU(), nn.Linear(d, d))
        # Descriptor-only ablation: when True, the graph term is ignored and only the
        # descriptor-tower code f is used, with the gate parameters left untouched
        # (state-dict compatible with the full model). Set by the model constructor.
        self.descriptor_only = False

    def forward(self, f, g, connected):
        if self.descriptor_only:
            return f
        trust = torch.sigmoid(self.gate(torch.cat([f, g, connected.unsqueeze(1)], dim=1)))
        return f + trust * g


class InductiveFusionHead(nn.Module):
    def __init__(self, emb_dim, feat_dim, d=192, hidden=384, dropout=0.25, descriptor_only=False):
        super().__init__()
        self.c_tower = _tower(feat_dim, hidden, d, dropout)
        self.p_tower = _tower(feat_dim, hidden, d, dropout)
        self.c_graph = nn.Sequential(nn.Linear(emb_dim, d), nn.LayerNorm(d))
        self.p_graph = nn.Sequential(nn.Linear(emb_dim, d), nn.LayerNorm(d))
        self.c_fuse = TrustGatedFusion(d)
        self.p_fuse = TrustGatedFusion(d)
        self.c_fuse.descriptor_only = descriptor_only
        self.p_fuse.descriptor_only = descriptor_only
        self.scorer = nn.Sequential(
            nn.LayerNorm(4 * d),
            nn.Linear(4 * d, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Linear(hidden // 2, 1),
        )
        self.bilinear = nn.Bilinear(d, d, 1)

    def forward(self, z_c, z_p, x_c, x_p, conn_c, conn_p):
        h_c = self.c_fuse(self.c_tower(x_c), self.c_graph(z_c), conn_c)
        h_p = self.p_fuse(self.p_tower(x_p), self.p_graph(z_p), conn_p)
        pair = torch.cat([h_c, h_p, h_c * h_p, torch.abs(h_c - h_p)], dim=1)
        return (self.scorer(pair) + self.bilinear(h_c, h_p)).squeeze(1)


class GNNInductiveFusion(GNNConnectivityGated):
    def __init__(self, *args, input_dim, pair_d=192, pair_hidden=384, pair_dropout=0.25, descriptor_only=False, **kwargs):
        super().__init__(*args, input_dim=input_dim, **kwargs)
        emb_dim = self.raw_skip[0].out_features
        self.pair_classifier = InductiveFusionHead(emb_dim, input_dim, pair_d, pair_hidden, pair_dropout,
                                                      descriptor_only=descriptor_only)
        self._cx = self._px = None

    def forward(self, x, edge_index, edge_attr=None, node_types=None, apply_edge_mask=False, data=None):
        out = super().forward(x, edge_index, edge_attr=edge_attr, node_types=node_types,
                              apply_edge_mask=apply_edge_mask, data=data)
        x_std = self._standardize(x).detach()
        self._cx = x_std[node_types == 0]
        self._px = x_std[node_types == 1]
        return out

    def classify_pairs(self, z_compounds, z_proteins, c_idx, p_idx):
        conn_c = (self._compound_log_degree[c_idx] > 0).float()
        conn_p = (self._protein_log_degree[p_idx] > 0).float()
        logit = self.pair_classifier(
            z_compounds[c_idx], z_proteins[p_idx], self._cx[c_idx], self._px[p_idx], conn_c, conn_p,
        )
        return logit, torch.sigmoid(logit)
