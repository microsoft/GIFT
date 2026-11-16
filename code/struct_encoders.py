"""
Structure-aware feature encoders for the Phase 3 pair head (optional struct_enc, off in the paper), on the published node features.

Feature layout (run graphs, standardized, block [compound 435 | protein 453]):
  compound: 0-9 ChEMBL descriptors | 10-265 Morgan bits (256) | 266-432 MACCS keys (167) | 433 Fsp3 | 434 stereocentres
  protein : 435-437 organism | 438 kinase flag | 439-458 amino-acid composition (20)
            | 459-858 dipeptide composition (20x20, first residue major) | 859-883 physicochemistry (25) | 884-887 complexity
CompoundFragmentEncoder: every set Morgan bit / MACCS key is a learned fragment token; learned latent queries
  cross-attend to the compound's fragment set, and a small transformer combines them with numeric descriptor tokens.
ProteinDipeptideEncoder: amino acids get learned embeddings and the 20x20 dipeptide grid is encoded through them
  (factorised bilinear pooling A1^T D A2), together with composition and numeric annotation/physicochemistry features.
Both are added to the published MLP towers (residual), so the head keeps all information the towers already had.
"""

import torch
import torch.nn as nn

C_NUM = list(range(0, 10)) + [433, 434]
C_BITS = slice(10, 433)
N_BITS = 423
P0 = 435
P_NUM = [P0 + i for i in (0, 1, 2, 3)] + list(range(P0 + 424, P0 + 453))
P_AAC = slice(P0 + 4, P0 + 24)
P_DPC = slice(P0 + 24, P0 + 424)


class CompoundFragmentEncoder(nn.Module):
    def __init__(self, d_out, d_tok=64, n_latent=4, n_heads=4, dropout=0.1):
        super().__init__()
        self.register_buffer("bit_threshold", torch.zeros(N_BITS))
        self.frag = nn.Parameter(torch.randn(N_BITS + 1, d_tok) * 0.02)  # last row: always-present null token
        self.num_w = nn.Parameter(torch.randn(len(C_NUM), d_tok) * 0.02)
        self.num_b = nn.Parameter(torch.zeros(len(C_NUM), d_tok))
        self.latent = nn.Parameter(torch.randn(n_latent, d_tok) * 0.02)
        self.cls = nn.Parameter(torch.zeros(1, d_tok))
        self.cross = nn.MultiheadAttention(d_tok, n_heads, dropout=dropout, batch_first=True)
        self.cross_norm = nn.LayerNorm(d_tok)
        self.mix = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_tok, n_heads, 2 * d_tok, dropout, activation="gelu", batch_first=True,
                                       norm_first=True), num_layers=1)
        self.out = nn.Sequential(nn.LayerNorm(d_tok), nn.Linear(d_tok, d_out), nn.LayerNorm(d_out))

    def forward(self, x):
        b = x.size(0)
        on = x[:, C_BITS] > self.bit_threshold
        pad = torch.cat([~on, torch.zeros(b, 1, dtype=torch.bool, device=x.device)], dim=1)
        keys = self.frag.unsqueeze(0).expand(b, -1, -1)
        lat, _ = self.cross(self.latent.unsqueeze(0).expand(b, -1, -1), keys, keys, key_padding_mask=pad,
                            need_weights=False)
        lat = self.cross_norm(lat)
        num = x[:, C_NUM].unsqueeze(-1) * self.num_w + self.num_b
        seq = torch.cat([self.cls.unsqueeze(0).expand(b, -1, -1), lat, num], dim=1)
        return self.out(self.mix(seq)[:, 0])


class ProteinDipeptideEncoder(nn.Module):
    def __init__(self, d_out, hidden=384, k=16, dropout=0.25):
        super().__init__()
        self.a1 = nn.Parameter(torch.randn(20, k) / 20 ** 0.5)
        self.a2 = nn.Parameter(torch.randn(20, k) / 20 ** 0.5)
        self.a3 = nn.Parameter(torch.randn(20, k) / 20 ** 0.5)
        self.num = nn.Linear(len(P_NUM), 64)
        self.mlp = nn.Sequential(nn.LayerNorm(k * k + k + 64), nn.Linear(k * k + k + 64, hidden), nn.GELU(),
                                 nn.Dropout(dropout), nn.Linear(hidden, d_out), nn.LayerNorm(d_out))

    def forward(self, x):
        dpc = x[:, P_DPC].view(-1, 20, 20)
        pooled = torch.einsum("bij,ik,jl->bkl", dpc, self.a1, self.a2).flatten(1)
        aac = x[:, P_AAC] @ self.a3
        return self.mlp(torch.cat([pooled, aac, self.num(x[:, P_NUM])], dim=1))


def set_bit_thresholds(model, node_features, node_types):
    """Midpoint between the two standardized values of each fingerprint column over the compounds (features only)."""
    with torch.no_grad():
        x = model._standardize(torch.as_tensor(node_features, dtype=torch.float32,
                                               device=model.feature_mean.device))
        xc = x[torch.as_tensor(node_types == 0, device=x.device)][:, C_BITS]
        thr = 0.5 * (xc.min(0).values + xc.max(0).values)
        thr[xc.min(0).values == xc.max(0).values] = float("inf")  # constant column: never a fragment
        model.pair_classifier.c_struct.bit_threshold.copy_(thr)
