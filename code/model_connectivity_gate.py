"""Encoder with a connectivity-confidence gate: a small gate (2 -> 16 -> 1) reads the log-degree of the compound
and protein nodes of a pair and shrinks the classifier's confidence toward 0.5 when either node is poorly connected.
Base class of the backbone and inductive fusion models.
"""
import torch
import torch.nn as nn
from torch_geometric.utils import degree

from model_pair_base import GNNPairClassifierBase


class GNNConnectivityGated(GNNPairClassifierBase):
    def __init__(self, *args, gate_hidden_dim=16, **kwargs):
        super().__init__(*args, **kwargs)
        self.degree_gate = nn.Sequential(
            nn.Linear(2, gate_hidden_dim),
            nn.ReLU(),
            nn.Linear(gate_hidden_dim, 1),
        )
        # start with gate ~= 0.5 (uncertain) so the sigmoid isn't saturated and
        # gradients can actually flow; bias=5.0 (near-1, "no effect") was tried
        # first but has ~zero gradient there and the gate never moved at all.
        nn.init.constant_(self.degree_gate[-1].bias, 0.0)
        self._compound_log_degree = None
        self._protein_log_degree = None

    def forward(self, x, edge_index, edge_attr=None, node_types=None, apply_edge_mask=False, data=None):
        with torch.no_grad():
            num_nodes = x.size(0)
            deg = (
                degree(edge_index[0], num_nodes=num_nodes, dtype=x.dtype)
                + degree(edge_index[1], num_nodes=num_nodes, dtype=x.dtype)
            )
            log_deg = torch.log1p(deg)
        x_out, z = super().forward(
            x, edge_index, edge_attr=edge_attr, node_types=node_types,
            apply_edge_mask=apply_edge_mask, data=data,
        )
        self._compound_log_degree = log_deg[node_types == 0].detach()
        self._protein_log_degree = log_deg[node_types == 1].detach()
        return x_out, z

    def classify_pairs(self, z_compounds, z_proteins, c_idx, p_idx):
        logits, probs = super().classify_pairs(z_compounds, z_proteins, c_idx, p_idx)
        if self._compound_log_degree is not None:
            c_deg = self._compound_log_degree[c_idx].unsqueeze(-1)
            p_deg = self._protein_log_degree[p_idx].unsqueeze(-1)
            # probs is 1D [N]; gate must match exactly (not broadcast to [N,N])
            gate = torch.sigmoid(self.degree_gate(torch.cat([c_deg, p_deg], dim=-1))).flatten()
            probs = gate * probs + (1 - gate) * 0.5
        return logits, probs
