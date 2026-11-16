"""Encoder model used for Phases 1-2 (contrastive pretraining and soft-edge learning): the edge-aware GNN
with a hierarchical mixture-of-experts pair head (interaction, metric, and topology experts with learned pair-wise
gating). The pair head is used only during backbone training; GIFT predictions use the inductive fusion classifier.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from model_connectivity_gate import GNNConnectivityGated


class HierMoEPairHead(nn.Module):
    def __init__(self, emb_dim: int, hidden_dim: int = 288, geo_dim: int = 112, confidence_scale: float = 0.15):
        super().__init__()
        pair_dim = emb_dim * 4
        self.confidence_scale = confidence_scale

        self.compound_proj = nn.Linear(emb_dim, emb_dim)
        self.protein_proj = nn.Linear(emb_dim, emb_dim)

        self.pair_skip = nn.Linear(pair_dim, 1)
        self.interaction_expert = nn.Sequential(
            nn.LayerNorm(pair_dim),
            nn.Linear(pair_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.12),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.GELU(),
            nn.Linear(hidden_dim // 4, 1),
        )
        self.interaction_residual = nn.Sequential(
            nn.LayerNorm(pair_dim),
            nn.Linear(pair_dim, hidden_dim // 3),
            nn.GELU(),
            nn.Linear(hidden_dim // 3, 1),
        )
        self.bilinear = nn.Bilinear(emb_dim, emb_dim, 1)
        self.projected_bilinear = nn.Bilinear(emb_dim, emb_dim, 1)

        self.metric_proj = nn.Linear(emb_dim, geo_dim)
        self.metric_expert = nn.Sequential(
            nn.LayerNorm(8),
            nn.Linear(8, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.GELU(),
            nn.Linear(hidden_dim // 4, 1),
        )

        self.topology_expert = nn.Sequential(
            nn.LayerNorm(6),
            nn.Linear(6, hidden_dim // 3),
            nn.GELU(),
            nn.Linear(hidden_dim // 3, hidden_dim // 6),
            nn.GELU(),
            nn.Linear(hidden_dim // 6, 1),
        )

        self.reliability = nn.Sequential(
            nn.LayerNorm(4),
            nn.Linear(4, hidden_dim // 4),
            nn.GELU(),
            nn.Linear(hidden_dim // 4, 1),
        )

        self.expert_gate = nn.Sequential(
            nn.LayerNorm(8),
            nn.Linear(8, hidden_dim // 3),
            nn.GELU(),
            nn.Linear(hidden_dim // 3, 3),
        )

        self.fusion = nn.Sequential(
            nn.LayerNorm(7),
            nn.Linear(7, hidden_dim // 3),
            nn.GELU(),
            nn.Linear(hidden_dim // 3, hidden_dim // 5),
            nn.GELU(),
            nn.Linear(hidden_dim // 5, 1),
        )

    def forward(self, c_emb, p_emb, c_deg, p_deg):
        pair_emb = torch.cat([c_emb, p_emb, torch.abs(c_emb - p_emb), c_emb * p_emb], dim=1)

        interaction_logit = self.interaction_expert(pair_emb).squeeze(1)
        interaction_logit = interaction_logit + self.interaction_residual(pair_emb).squeeze(1)
        interaction_logit = interaction_logit + self.pair_skip(pair_emb).squeeze(1)
        interaction_logit = interaction_logit + self.bilinear(c_emb, p_emb).squeeze(1)

        c_proj = self.compound_proj(c_emb)
        p_proj = self.protein_proj(p_emb)
        interaction_logit = interaction_logit + self.projected_bilinear(c_proj, p_proj).squeeze(1)

        c_geo = self.metric_proj(c_emb)
        p_geo = self.metric_proj(p_emb)
        cosine = F.cosine_similarity(c_geo, p_geo, dim=1)
        sq_dist = (c_geo - p_geo).pow(2).sum(dim=1)
        l1_dist = torch.abs(c_geo - p_geo).mean(dim=1)
        norm_gap = torch.abs(c_geo.norm(dim=1) - p_geo.norm(dim=1))
        dot = (c_geo * p_geo).sum(dim=1)
        l2_dist = torch.sqrt((c_geo - p_geo).pow(2).sum(dim=1) + 1e-8)
        geo_gap = torch.abs((c_geo * c_geo).mean(dim=1) - (p_geo * p_geo).mean(dim=1))
        cosine_raw = F.cosine_similarity(c_emb, p_emb, dim=1)
        metric_in = torch.stack([cosine, dot, -sq_dist, -l1_dist, -norm_gap, -l2_dist, -geo_gap, cosine_raw], dim=1)
        metric_logit = self.metric_expert(metric_in).squeeze(1)

        rel_in = torch.stack([
            c_deg,
            p_deg,
            c_emb.norm(dim=1),
            p_emb.norm(dim=1),
        ], dim=1)
        reliability = torch.sigmoid(self.reliability(rel_in)).squeeze(1)

        topo_in = torch.stack([
            c_deg,
            p_deg,
            torch.abs(c_deg - p_deg),
            c_deg * p_deg,
            c_emb.norm(dim=1),
            p_emb.norm(dim=1),
        ], dim=1)
        topology_logit = self.topology_expert(topo_in).squeeze(1)

        gate_in = torch.stack([
            interaction_logit,
            metric_logit,
            topology_logit,
            reliability,
            torch.abs(interaction_logit - metric_logit),
            torch.abs(interaction_logit - topology_logit),
            torch.abs(metric_logit - topology_logit),
            interaction_logit * metric_logit,
        ], dim=1)
        expert_weights = torch.softmax(self.expert_gate(gate_in), dim=1)
        e_interaction = expert_weights[:, 0] * interaction_logit
        e_metric = expert_weights[:, 1] * metric_logit
        e_topology = expert_weights[:, 2] * topology_logit

        fusion_in = torch.stack([
            e_interaction,
            e_metric,
            e_topology,
            reliability,
            e_interaction + e_metric + e_topology,
            torch.abs(e_interaction - e_metric),
            torch.abs(e_metric - e_topology),
        ], dim=1)
        base_logit = self.fusion(fusion_in).squeeze(1)

        conf = 1.0 + self.confidence_scale * ((reliability - 0.5) * 2.0)
        final_logit = base_logit * conf

        return final_logit, interaction_logit, metric_logit, reliability


class GNNBackbone(GNNConnectivityGated):
    def __init__(self, *args, pair_hidden_dim=288, pair_geo_dim=112, confidence_scale=0.15, **kwargs):
        super().__init__(*args, **kwargs)
        emb_dim = self.raw_skip[0].out_features
        self.pair_classifier = HierMoEPairHead(
            emb_dim=emb_dim,
            hidden_dim=pair_hidden_dim,
            geo_dim=pair_geo_dim,
            confidence_scale=confidence_scale,
        )

    def classify_pairs(self, z_compounds, z_proteins, c_idx, p_idx):
        c_emb = z_compounds[c_idx]
        p_emb = z_proteins[p_idx]

        if self._compound_log_degree is None or self._protein_log_degree is None:
            c_deg = torch.zeros(c_idx.size(0), dtype=c_emb.dtype, device=c_emb.device)
            p_deg = torch.zeros(p_idx.size(0), dtype=p_emb.dtype, device=p_emb.device)
        else:
            c_deg = self._compound_log_degree[c_idx]
            p_deg = self._protein_log_degree[p_idx]

        final_logit, _, _, _ = self.pair_classifier(c_emb, p_emb, c_deg, p_deg)
        final_prob = torch.sigmoid(final_logit)
        return final_logit, final_prob
