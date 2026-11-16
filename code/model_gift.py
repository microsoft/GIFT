"""
GIFT model: the inductive fusion model (model_inductive_fusion.GNNInductiveFusion) with optional architecture changes, all off by default (the paper uses the defaults).

- n_heads: the pair scorer (MLP + bilinear term) is replicated n_heads times and the logits are averaged.
- desc_aux: an auxiliary pair scorer on the descriptor towers alone, trained with its own binary
  cross-entropy (training only; predictions come from the fused scorer).
- potency: an auxiliary regression head on the pair representation that predicts the standardized pChEMBL
  value of training interactions (training only; uses training labels only).
- ban: a bilinear attention interaction term between multi-token compound and protein representations,
  added to the pair logit (uses only the fused compound and protein representations).
- soft_ccpp: label-free compound-compound and protein-protein soft edges proposed by the frozen Phase 2
  soft-edge scorers are added to the message-passing graph in every forward pass, strictly inductively (see
  _soft_homogeneous_edges) (training, validation,
  test, external alike). No compound-protein soft edges are added, so no interaction can be re-introduced.
- soft_real_flag: with soft_ccpp, the binary connectivity flag of the trust gate is computed from the real
  (observed) edges only, so soft edges never mark a node as having graph evidence.
- nbr_lig: neighbour-ligand attention; in addition to lig_attn, a second attention channel compares the compound
  with the known ligands of the protein's STRING neighbours (protein-protein edges of the current graph),
  so that proteins with few known ligands borrow evidence from related proteins. Same exclusions as lig_attn.
- lig_attn: known-ligand attention; a compound is compared, by learned attention, with the compounds that
  have a compound-protein edge to the protein in the current message-passing graph. The scored compound is
  always excluded from this ligand set, and in end-to-end training the masked target edges are absent
  from the graph, so a pair can never attend to its own interaction.
- struct_enc: structure-aware compound (fragment-token) and protein (factorised dipeptide) encoders added to the
  feature towers of the pair head (see struct_encoders.py); same published node features, no new inputs.
- prop_bias: learned compound and protein propensity terms beta_c(h_c) + beta_p(h_p) added to the pair logit (biases of
  matrix factorisation), computed from the fused node representations; no degree input; zero at initialisation.
- unc_out (H1): the pair logit s becomes softplus(a(h)) * s + b(h), with a, b linear in the normalised pair
  representation (no degree input); identity at initialisation.
The connectivity flag of the trust gate is unchanged from the inductive fusion model. With all options off the parameters and outputs
are identical to the inductive fusion model.
"""

import math

import torch
import torch.nn as nn
from torch_geometric.utils import degree

from model_inductive_fusion import GNNInductiveFusion, InductiveFusionHead
from struct_encoders import CompoundFragmentEncoder, ProteinDipeptideEncoder

SOFT_EPOCH = 199  # final Phase 2 epoch: 20,000 compound-compound and 20,000 protein-protein soft edges


def _mlp_scorer(d, hidden, dropout):
    return nn.Sequential(nn.LayerNorm(4 * d), nn.Linear(4 * d, hidden), nn.GELU(), nn.Dropout(dropout),
                         nn.Linear(hidden, hidden // 2), nn.GELU(), nn.Linear(hidden // 2, 1))


class GIFTFusionHead(InductiveFusionHead):
    def __init__(self, emb_dim, feat_dim, d=192, hidden=384, dropout=0.25, n_heads=1, desc_aux=False,
                 potency=False, ban=False, lig_attn=False, ban_tokens=8, ban_dim=32, lig_dim=64, nbr_lig=False,
                 struct_enc=False, unc_out=False, prop_bias=False, unc_node=False, unc_hetero=False, evidential=False,
                 edl_exp=False, descriptor_only=False):
        super().__init__(emb_dim, feat_dim, d, hidden, dropout, descriptor_only=descriptor_only)
        self.evidential = evidential
        self.edl_exp = edl_exp
        if evidential:
            # Binary evidential head (Beta(alpha, beta)): the existing scorer gives positive evidence, a second scorer of
            # the same form gives negative evidence; the returned logit is log(alpha / beta).
            self.neg_scorer = _mlp_scorer(d, hidden, dropout)
            self.neg_bilinear = nn.Bilinear(d, d, 1)
            self._last_ab = None
        self.unc_node, self.unc_hetero = unc_node, unc_hetero
        if unc_node:
            # H1-node: output scale/offset from the compound and the protein separately (transfers to unseen pairs).
            self.un_norm_c, self.un_norm_p = nn.LayerNorm(d), nn.LayerNorm(d)
            self.un_scale_c, self.un_scale_p = nn.Linear(d, 1), nn.Linear(d, 1)
            self.un_shift_c, self.un_shift_p = nn.Linear(d, 1), nn.Linear(d, 1)
            for lin in (self.un_scale_c, self.un_scale_p, self.un_shift_c, self.un_shift_p):
                nn.init.zeros_(lin.weight)
                nn.init.zeros_(lin.bias)
            self.un_scale_b = nn.Parameter(torch.tensor(math.log(math.e - 1.0)))
        if unc_hetero:
            # H1-unc: per-pair uncertainty sigma; probability = sigmoid(mu / sqrt(1 + pi sigma^2 / 8)) (probit approx.).
            self.uh_norm = nn.LayerNorm(4 * d)
            self.uh_sigma = nn.Linear(4 * d, 1)
            nn.init.zeros_(self.uh_sigma.weight)
            nn.init.constant_(self.uh_sigma.bias, -5.0)
        self.prop_bias = prop_bias
        if prop_bias:
            # Learned compound and protein propensity (promiscuity) terms from the fused node representations; zero at init.
            def _bias_net():
                net = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d // 2), nn.GELU(), nn.Linear(d // 2, 1))
                nn.init.zeros_(net[-1].weight)
                nn.init.zeros_(net[-1].bias)
                return net
            self.c_prop = _bias_net()
            self.p_prop = _bias_net()
        self.unc_out = unc_out
        if unc_out:
            # H1: per-pair scale a(h) > 0 and offset b(h) learned from the pair representation; identity at init.
            self.unc_norm = nn.LayerNorm(4 * d)
            self.unc_scale = nn.Linear(4 * d, 1)
            self.unc_shift = nn.Linear(4 * d, 1)
            for lin in (self.unc_scale, self.unc_shift):
                nn.init.zeros_(lin.weight)
            nn.init.constant_(self.unc_scale.bias, math.log(math.e - 1.0))
            nn.init.zeros_(self.unc_shift.bias)
        self.struct_enc = struct_enc
        if struct_enc:
            self.c_struct = CompoundFragmentEncoder(d)
            self.p_struct = ProteinDipeptideEncoder(d, hidden, dropout=dropout)
        self.aux_scorer = (nn.Sequential(nn.LayerNorm(4 * d), nn.Linear(4 * d, hidden), nn.GELU(), nn.Dropout(dropout),
                                         nn.Linear(hidden, 1)) if desc_aux else None)
        self.extra_scorers = nn.ModuleList(_mlp_scorer(d, hidden, dropout) for _ in range(n_heads - 1))
        self.extra_bilinears = nn.ModuleList(nn.Bilinear(d, d, 1) for _ in range(n_heads - 1))
        self.potency_head = (nn.Sequential(nn.LayerNorm(4 * d), nn.Linear(4 * d, hidden // 2), nn.GELU(),
                                           nn.Linear(hidden // 2, 1)) if potency else None)
        self.ban = ban
        if ban:
            self.ban_tokens, self.ban_dim = ban_tokens, ban_dim
            self.ban_c = nn.Linear(d, ban_tokens * ban_dim)
            self.ban_p = nn.Linear(d, ban_tokens * ban_dim)
            self.ban_u = nn.Linear(ban_dim, 2 * ban_dim)
            self.ban_v = nn.Linear(ban_dim, 2 * ban_dim)
            self.ban_q = nn.Parameter(torch.randn(2 * ban_dim) / math.sqrt(2 * ban_dim))
            self.ban_out = nn.Sequential(nn.LayerNorm(2 * ban_dim), nn.Linear(2 * ban_dim, 1))
        self.lig_attn = lig_attn
        if lig_attn:
            self.lig_q = nn.Linear(d, lig_dim)
            self.lig_k = nn.Linear(d, lig_dim)
            self.lig_v = nn.Linear(d, lig_dim)
            self.lig_out = nn.Sequential(nn.LayerNorm(lig_dim), nn.Linear(lig_dim, 1))
        self.nbr_lig = nbr_lig
        if nbr_lig:
            self.nbr_q = nn.Linear(d, lig_dim)
            self.nbr_k = nn.Linear(d, lig_dim)
            self.nbr_v = nn.Linear(d, lig_dim)
            self.nbr_out = nn.Sequential(nn.LayerNorm(lig_dim), nn.Linear(lig_dim, 1))
        self._last_pair = None

    def encode_c(self, x_c):
        return self.c_tower(x_c) + self.c_struct(x_c) if self.struct_enc else self.c_tower(x_c)

    def encode_p(self, x_p):
        return self.p_tower(x_p) + self.p_struct(x_p) if self.struct_enc else self.p_tower(x_p)

    def fuse_compounds(self, z_c, x_c, conn_c):
        return self.c_fuse(self.encode_c(x_c), self.c_graph(z_c), conn_c)

    def ban_logit(self, h_c, h_p):
        b = h_c.size(0)
        tc = torch.relu(self.ban_u(self.ban_c(h_c).view(b, self.ban_tokens, self.ban_dim)))
        tp = torch.relu(self.ban_v(self.ban_p(h_p).view(b, self.ban_tokens, self.ban_dim)))
        att = torch.einsum("bih,bjh,h->bij", tc, tp, self.ban_q).flatten(1).softmax(-1)
        att = att.view(b, self.ban_tokens, self.ban_tokens)
        pooled = torch.einsum("bij,bih,bjh->bh", att, tc, tp)
        return self.ban_out(pooled)

    def forward(self, z_c, z_p, x_c, x_p, conn_c, conn_p):
        h_c = self.fuse_compounds(z_c, x_c, conn_c)
        h_p = self.p_fuse(self.encode_p(x_p), self.p_graph(z_p), conn_p)
        pair = torch.cat([h_c, h_p, h_c * h_p, torch.abs(h_c - h_p)], dim=1)
        self._last_pair, self._last_hc = pair, h_c
        logits = [self.scorer(pair) + self.bilinear(h_c, h_p)]
        logits += [s(pair) + b(h_c, h_p) for s, b in zip(self.extra_scorers, self.extra_bilinears)]
        out = torch.stack(logits, 0).mean(0)
        if self.ban:
            out = out + self.ban_logit(h_c, h_p)
        if self.prop_bias:
            out = out + self.c_prop(h_c) + self.p_prop(h_p)
        if self.evidential:
            s_neg = self.neg_scorer(pair) + self.neg_bilinear(h_c, h_p)
            if self.edl_exp:
                # Exponential evidence (Pandey & Yu, ICML 2023): no zero-evidence region, every sample keeps a gradient.
                alpha = torch.exp(out.clamp(-10.0, 10.0)) + 1.0
                beta = torch.exp(s_neg.clamp(-10.0, 10.0)) + 1.0
            else:
                alpha = nn.functional.softplus(out) + 1.0
                beta = nn.functional.softplus(s_neg) + 1.0
            self._last_ab = (alpha.squeeze(1), beta.squeeze(1))
            out = torch.log(alpha) - torch.log(beta)
        if self.unc_node:
            nc_, np_ = self.un_norm_c(h_c), self.un_norm_p(h_p)
            scale = nn.functional.softplus(self.un_scale_c(nc_) + self.un_scale_p(np_) + self.un_scale_b)
            out = scale * out + self.un_shift_c(nc_) + self.un_shift_p(np_)
        if self.unc_hetero:
            sigma = nn.functional.softplus(self.uh_sigma(self.uh_norm(pair)))
            out = out / torch.sqrt(1 + math.pi * sigma ** 2 / 8)
        if self.unc_out:
            u = self.unc_norm(pair)
            out = nn.functional.softplus(self.unc_scale(u)) * out + self.unc_shift(u)
        return out.squeeze(1)

    def aux_logits(self, x_c, x_p):
        f_c, f_p = self.c_tower(x_c), self.p_tower(x_p)
        return self.aux_scorer(torch.cat([f_c, f_p, f_c * f_p, torch.abs(f_c - f_p)], dim=1)).squeeze(1)


class GIFT(GNNInductiveFusion):
    def __init__(self, *args, input_dim, pair_d=192, pair_hidden=384, pair_dropout=0.25, n_heads=1, desc_aux=False,
                 potency=False, ban=False, lig_attn=False, soft_ccpp=False, nbr_lig=False, soft_real_flag=False,
                 struct_enc=False, unc_out=False, prop_bias=False, unc_node=False, unc_hetero=False, evidential=False,
                 edl_exp=False, descriptor_only=False, **kwargs):
        super().__init__(*args, input_dim=input_dim, pair_d=pair_d, pair_hidden=pair_hidden,
                         pair_dropout=pair_dropout, descriptor_only=descriptor_only, **kwargs)
        lig_attn = lig_attn or nbr_lig
        self.soft_ccpp, self.lig_attn, self.nbr_lig = soft_ccpp, lig_attn, nbr_lig
        self.soft_real_flag = soft_real_flag
        if n_heads > 1 or desc_aux or potency or ban or lig_attn or struct_enc or unc_out or prop_bias or unc_node or unc_hetero or evidential:
            emb_dim = self.raw_skip[0].out_features
            self.pair_classifier = GIFTFusionHead(emb_dim, input_dim, pair_d, pair_hidden, pair_dropout,
                                                          n_heads, desc_aux, potency, ban, lig_attn, nbr_lig=nbr_lig,
                                                          struct_enc=struct_enc, unc_out=unc_out, prop_bias=prop_bias,
                                                          unc_node=unc_node, unc_hetero=unc_hetero, evidential=evidential,
                                                          edl_exp=edl_exp, descriptor_only=descriptor_only)
        self._lig_adj = self._nbr_adj = None

    def _soft_homogeneous_edges(self, x, edge_index, edge_attr, node_types):
        """Label-free CC and PP soft edges from the frozen Phase 2 scorers (no gradient, no CP edges), strictly inductive.

        Graph nodes (compounds and proteins of the training graph) receive the top-k soft edges of the final Phase 2
        epoch, selected among graph nodes only. Compounds appended after the graph's proteins (external compounds,
        which have no edges) do not take part in this selection; each is linked independently to the graph
        compounds whose score with it reaches the score threshold of the selected graph edges, by directed edges from
        the graph compound to the external compound. External compounds are never linked to each other and send no
        messages, so the embeddings of graph nodes and the prediction for an external compound do not depend on which
        other external compounds are present.
        """
        captured = {}

        def capture(module, inp, out):
            captured.setdefault("x", inp[0])

        hook = self.convs[-1].register_forward_hook(capture)
        with torch.no_grad():
            super().forward(x, edge_index, edge_attr=edge_attr, node_types=node_types)
            hook.remove()
            inter = captured["x"]
            first_protein = int(torch.nonzero(node_types == 1).flatten().min())
            all_c = torch.nonzero(node_types == 0).flatten()
            graph_c, ext_c = all_c[all_c < first_protein], all_c[all_c > first_protein]
            graph_p = torch.nonzero(node_types == 1).flatten()
            new_edges, new_w = [], []
            for ids, mask_module in ((graph_c, self.compound_edge_mask), (graph_p, self.protein_edge_mask)):
                scores = mask_module.compute_self_attention_scores(inter[ids], batch_size=100)
                e, w = mask_module.select_edges_progressive(scores, SOFT_EPOCH, ids.numel(), node_offset=0)
                new_edges.append(ids[e])
                new_w.append(w)
                if mask_module is self.compound_edge_mask and ext_c.numel() > 0:
                    tau = w.min()
                    both = torch.cat([graph_c, ext_c])
                    s = mask_module.compute_self_attention_scores(inter[both], batch_size=100)
                    block = s[: graph_c.numel(), graph_c.numel():]  # rows: graph compounds, columns: external
                    gi, ej = torch.nonzero(block >= tau, as_tuple=True)
                    ws = block[gi, ej]
                    # Directed graph -> external: external compounds receive messages but never send any, so
                    # graph-node embeddings, and hence other external compounds, are unaffected by them.
                    new_edges.append(torch.stack([graph_c[gi], ext_c[ej]]))
                    new_w.append(ws)
            e = torch.cat(new_edges, 1)
            w = torch.cat(new_w, 0)
        attr = torch.ones((e.size(1), edge_attr.size(1)), dtype=edge_attr.dtype, device=edge_attr.device) * w.unsqueeze(1)
        return torch.cat([edge_index, e], 1), torch.cat([edge_attr, attr], 0)

    def forward(self, x, edge_index, edge_attr=None, node_types=None, apply_edge_mask=False, data=None):
        real_edge_index = edge_index
        if self.soft_ccpp and node_types is not None:
            edge_index, edge_attr = self._soft_homogeneous_edges(x, edge_index, edge_attr, node_types)
        out = super().forward(x, edge_index, edge_attr=edge_attr, node_types=node_types,
                              apply_edge_mask=False if self.soft_ccpp else apply_edge_mask, data=data)
        if self.soft_ccpp and self.soft_real_flag and node_types is not None:
            with torch.no_grad():
                n = x.size(0)
                log_deg = torch.log1p(degree(real_edge_index[0], num_nodes=n, dtype=x.dtype)
                                      + degree(real_edge_index[1], num_nodes=n, dtype=x.dtype))
            self._compound_log_degree = log_deg[node_types == 0]
            self._protein_log_degree = log_deg[node_types == 1]
        if self.lig_attn:
            with torch.no_grad():
                cm = node_types == 0
                local = torch.cumsum(cm.long(), 0) - 1
                plocal = torch.cumsum((~cm).long(), 0) - 1
                src, dst = edge_index
                cp = cm[src] & ~cm[dst]
                adj = torch.zeros((int((~cm).sum()), int(cm.sum())), dtype=torch.bool, device=x.device)
                adj[plocal[dst[cp]], local[src[cp]]] = True
                pc = ~cm[src] & cm[dst]
                adj[plocal[src[pc]], local[dst[pc]]] = True
                if self.nbr_lig:
                    pp = ~cm[src] & ~cm[dst]
                    ppi = torch.zeros((adj.size(0), adj.size(0)), device=x.device)
                    ppi[plocal[src[pp]], plocal[dst[pp]]] = 1.0
                    ppi.fill_diagonal_(0.0)
                    self._nbr_adj = (ppi @ adj.float()) > 0
            self._lig_adj = adj
        return out

    def classify_pairs(self, z_compounds, z_proteins, c_idx, p_idx):
        logit, _ = super().classify_pairs(z_compounds, z_proteins, c_idx, p_idx)
        if self.lig_attn:
            head = self.pair_classifier
            conn_all = (self._compound_log_degree > 0).float()
            h_all = head.fuse_compounds(z_compounds, self._cx, conn_all)
            q = head.lig_q(head._last_hc)
            k, v = head.lig_k(h_all), head.lig_v(h_all)
            rows = torch.arange(c_idx.numel(), device=c_idx.device)
            s = (q @ k.T) / math.sqrt(q.size(1))
            mask = self._lig_adj[p_idx].clone()
            mask[rows, c_idx] = False
            s = s.masked_fill(~mask, float("-inf"))
            att = torch.nan_to_num(torch.softmax(s, dim=1), nan=0.0)
            logit = logit + head.lig_out(att @ v).squeeze(1)
            if self.nbr_lig:
                qn, kn, vn = head.nbr_q(head._last_hc), head.nbr_k(h_all), head.nbr_v(h_all)
                sn = (qn @ kn.T) / math.sqrt(qn.size(1))
                nmask = self._nbr_adj[p_idx].clone()
                nmask[rows, c_idx] = False
                sn = sn.masked_fill(~nmask, float("-inf"))
                attn = torch.nan_to_num(torch.softmax(sn, dim=1), nan=0.0)
                logit = logit + head.nbr_out(attn @ vn).squeeze(1)
        return logit, torch.sigmoid(logit)

    def desc_aux_logits(self, c_idx, p_idx):
        """Auxiliary descriptor-only logits (training only); requires a preceding forward pass."""
        return self.pair_classifier.aux_logits(self._cx[c_idx], self._px[p_idx])

    def potency_from_last_pairs(self):
        """Standardized pChEMBL predictions for the pairs of the last classify_pairs call (training only)."""
        return self.pair_classifier.potency_head(self.pair_classifier._last_pair).squeeze(1)
