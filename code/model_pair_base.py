"""Base encoder with two changes for compounds outside the training graph:
1. LayerNorm instead of BatchNorm in the convolution stack, so normalization does not depend on the training
   feature distribution.
2. A raw-feature residual skip, so nodes with few or no edges keep an informative embedding of their own features.
"""
import torch
import torch.nn as nn

from model_encoder import DrugTargetGNN


class GNNPairClassifierBase(DrugTargetGNN):
    def __init__(self, input_dim, hidden_dim=128, output_dim=64, num_layers=3, edge_dim=2,
                 use_edge_mask=False, edge_dropout_p: float = 0.15,
                 feature_mean=None, feature_std=None):
        super().__init__(
            input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim,
            num_layers=num_layers, edge_dim=edge_dim, use_edge_mask=use_edge_mask,
        )
        self.edge_dropout_p = edge_dropout_p

        # per-sample normalization is immune to train/test running-stat drift
        self.bns = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in self.bns])

        self.raw_skip = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.LayerNorm(output_dim),
        )
        self.raw_skip_scale = nn.Parameter(torch.ones(1) * 0.5)

        self.pair_classifier = nn.Sequential(
            nn.LayerNorm(output_dim * 2),
            nn.Linear(output_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

        if feature_mean is None:
            feature_mean = torch.zeros(input_dim)
        if feature_std is None:
            feature_std = torch.ones(input_dim)
        self.register_buffer("feature_mean", torch.as_tensor(feature_mean, dtype=torch.float32))
        self.register_buffer("feature_std", torch.as_tensor(feature_std, dtype=torch.float32).clamp_min(1e-6))

    def _standardize(self, x):
        return (x - self.feature_mean) / self.feature_std

    def _drop_edges(self, edge_index, edge_attr):
        num_edges = edge_index.size(1)
        if num_edges == 0 or self.edge_dropout_p <= 0:
            return edge_index, edge_attr
        keep_mask = torch.rand(num_edges, device=edge_index.device) >= self.edge_dropout_p
        edge_index = edge_index[:, keep_mask]
        if edge_attr is not None:
            edge_attr = edge_attr[keep_mask]
        return edge_index, edge_attr

    def forward(self, x, edge_index, edge_attr=None, node_types=None, apply_edge_mask=False, data=None):
        x = self._standardize(x)
        if self.training and self.edge_dropout_p > 0:
            edge_index, edge_attr = self._drop_edges(edge_index, edge_attr)
        raw_x = x
        x_out, z = super().forward(
            x, edge_index, edge_attr=edge_attr, node_types=node_types,
            apply_edge_mask=apply_edge_mask, data=data,
        )
        z = z + self.raw_skip(raw_x) * self.raw_skip_scale
        return x_out, z
