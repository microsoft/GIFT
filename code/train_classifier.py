"""Phase 3a: train the inductive fusion classifier of GIFT on a frozen Phase 2 encoder (training CP edges only in
the message-passing graph). Each epoch a random fraction of compounds and proteins is edge-isolated (masked) before
embedding, so the classifier learns to score nodes with and without graph connectivity. Optional features
(compound-matched training negatives, pairwise ranking term) are off in the paper's configuration.
"""

import argparse
import contextlib
import io
import json
import os
import pickle

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

from model_encoder import (
    analyze_meaningful_validation_categories,
    calculate_balanced_metrics,
    convert_numpy_to_python,
    evaluate_with_classifier,
    set_seed,
    split_interaction_data,
)
from model_gift import GIFT
from graph_cleaning import remove_cp_pairs
from negative_pool import evaluation_exclusion, reserved_eval_mask, training_exclusion

GRAPH_PATH = "data/graphs/graph_similarity_input.pkl"
# Validation negatives per positive; fixed so every method is selected on the same validation set.
VAL_NEG_RATIO = 5


def build_isolated_edge_index(edge_index, edge_attr, isolate_node_ids):
    keep = ~(torch.isin(edge_index[0], isolate_node_ids) | torch.isin(edge_index[1], isolate_node_ids))
    return edge_index[:, keep], (edge_attr[keep] if edge_attr is not None else None)


def sample_negatives(rng, compound_pool, num_proteins, n, exclude):
    out, seen = [], set()
    while len(out) < n:
        c = rng.choice(compound_pool, size=2 * (n - len(out)))
        p = rng.randint(0, num_proteins, size=c.size)
        for pair in zip(c.tolist(), p.tolist()):
            if pair not in exclude and pair not in seen:
                out.append(pair)
                seen.add(pair)
                if len(out) == n:
                    break
    return out


def nnpu_loss(logits, y, eta):
    """Non-negative PU risk (Kiryo et al. 2017) with balanced classes: sampled negatives are unlabelled pairs, a
    fraction eta of which are untested actives. Returns the loss to back-propagate."""
    pos, unl = y > 0.5, y <= 0.5
    if not pos.any() or not unl.any():
        return F.binary_cross_entropy_with_logits(logits, y)
    r_pos = F.binary_cross_entropy_with_logits(logits[pos], torch.ones_like(logits[pos]))
    r_neg = (F.binary_cross_entropy_with_logits(logits[unl], torch.zeros_like(logits[unl]))
             - eta * F.binary_cross_entropy_with_logits(logits[pos], torch.zeros_like(logits[pos]))) / (1.0 - eta)
    return 0.5 * (r_pos + r_neg) if r_neg.item() >= 0 else -0.5 * r_neg


def popularity_negatives(rng, positives, n, exclude):
    """`n` negatives whose compounds and proteins are drawn with their frequencies among the training positives
    (AI-Bind-style), so a node's identity carries no label information; same exclusion rule as the default sampler."""
    cs = np.array([c for c, _ in positives])
    ps = np.array([p for _, p in positives])
    out, seen = [], set()
    while len(out) < n:
        c = cs[rng.randint(0, len(cs), size=2 * (n - len(out)))]
        p = ps[rng.randint(0, len(ps), size=c.size)]
        for pair in zip(c.tolist(), p.tolist()):
            if pair not in exclude and pair not in seen:
                out.append(pair)
                seen.add(pair)
                if len(out) == n:
                    break
    return out


def positive_sample_weights(logits, y, masked, masked_pos_weight=0.0, hard_pos_focus=False, hard_pos_strength=1.0,
                            hard_pos_masked_only=False):
    """Per-pair weights for the Phase 3 BCE (negatives always 1).
    A1 masked_pos_weight > 0: positives whose compound or protein is masked in this step get this weight.
    A2 hard_pos_focus: positives get 1 + k (1 - p)^2 (p detached, k = hard_pos_strength), so confidently missed
       positives count up to (1 + k)x; with hard_pos_masked_only only positives with a masked endpoint are focused."""
    w = torch.ones_like(y)
    pos = y > 0.5
    if masked_pos_weight > 0:
        w = torch.where(pos & masked, torch.full_like(w, masked_pos_weight), w)
    if hard_pos_focus:
        sel = pos & masked if hard_pos_masked_only else pos
        w = torch.where(sel, w * (1 + hard_pos_strength * (1 - torch.sigmoid(logits.detach())) ** 2), w)
    return w


def l2rw_weights(model, lr, zc, zp, c_idx, p_idx, y, meta_pos, meta_neg, n_meta=512):
    """Meta-learned per-pair weights (learning to reweight, Ren et al. 2018) from a one-step look-ahead of the pair
    classifier: a pair's weight is the (clipped) decrease of a pairwise ROC-AUC surrogate on held-out TRAINING pairs
    that up-weighting it would cause; weights are normalised to mean 1 over the minibatch. Embeddings are detached."""
    head = model.pair_classifier
    zc, zp = zc.detach(), zp.detach()

    def hargs(c, p):
        conn_c = (model._compound_log_degree[c] > 0).float()
        conn_p = (model._protein_log_degree[p] > 0).float()
        return (zc[c], zp[p], model._cx[c], model._px[p], conn_c, conn_p)

    params = dict(head.named_parameters())
    lg = torch.func.functional_call(head, params, hargs(c_idx, p_idx))
    per = F.binary_cross_entropy_with_logits(lg, y, reduction="none")
    eps = torch.zeros_like(per, requires_grad=True)
    grads = torch.autograd.grad((eps * per).sum(), list(params.values()), create_graph=True, allow_unused=True)
    fast = {k: (v - lr * g if g is not None else v) for (k, v), g in zip(params.items(), grads)}
    (mcp, mpp), (mcn, mpn) = meta_pos, meta_neg
    bi = torch.randint(0, mcp.numel(), (n_meta,), device=mcp.device)
    bj = torch.randint(0, mcn.numel(), (n_meta,), device=mcn.device)
    s_pos = torch.func.functional_call(head, fast, hargs(mcp[bi], mpp[bi]))
    s_neg = torch.func.functional_call(head, fast, hargs(mcn[bj], mpn[bj]))
    meta = F.softplus(-(s_pos[:, None] - s_neg[None, :])).mean()
    g = torch.autograd.grad(meta, eps)[0]
    w = (-g).clamp_min(0)
    if float(w.sum()) <= 0:
        return torch.ones_like(per)
    return (w / w.sum() * w.numel()).detach()


def _pos_weighted_mean(per_pair, y, pos_weight):
    """Mean over pairs with positives weighted by pos_weight (float or tensor; None = unweighted)."""
    if pos_weight is None:
        return per_pair.mean()
    return ((1 + (pos_weight - 1) * y) * per_pair).mean()


def edl_loss(alpha, beta, y, kl_coef, pos_weight=None):
    """Binary evidential loss (Sensoy et al. 2018, Beta form): expected cross-entropy under Beta(alpha, beta) plus
    kl_coef * KL(Beta(alpha~, beta~) || Beta(1, 1)), where the evidence of the true class is removed (alpha~, beta~).
    pos_weight: optional positive class weight (e.g. the learned weight of --learn_pos_weight)."""
    s = alpha + beta
    ece = y * (torch.digamma(s) - torch.digamma(alpha)) + (1 - y) * (torch.digamma(s) - torch.digamma(beta))
    a_t = y + (1 - y) * alpha
    b_t = (1 - y) + y * beta
    s_t = a_t + b_t
    kl = (torch.lgamma(s_t) - torch.lgamma(a_t) - torch.lgamma(b_t)
          + (a_t - 1) * (torch.digamma(a_t) - torch.digamma(s_t)) + (b_t - 1) * (torch.digamma(b_t) - torch.digamma(s_t)))
    return _pos_weighted_mean(ece + kl_coef * kl, y, pos_weight)


def edl_mse_loss(alpha, beta, y, kl_coef, pos_weight=None):
    """Binary evidential loss, squared-error form (Sensoy et al. 2018, eq. 5): (y - p)^2 + p (1 - p) / (S + 1) for both
    classes, plus kl_coef * KL(Beta(alpha~, beta~) || Beta(1, 1)) on the misleading evidence."""
    s = alpha + beta
    p, q = alpha / s, beta / s
    mse = (y - p) ** 2 + ((1 - y) - q) ** 2 + (p * (1 - p) + q * (1 - q)) / (s + 1)
    a_t = y + (1 - y) * alpha
    b_t = (1 - y) + y * beta
    s_t = a_t + b_t
    kl = (torch.lgamma(s_t) - torch.lgamma(a_t) - torch.lgamma(b_t)
          + (a_t - 1) * (torch.digamma(a_t) - torch.digamma(s_t)) + (b_t - 1) * (torch.digamma(b_t) - torch.digamma(s_t)))
    return _pos_weighted_mean(mse + kl_coef * kl, y, pos_weight)


def asl_loss(logits, y, gamma_pos=0.0, gamma_neg=4.0, clip=0.05, eps=1e-8):
    """Asymmetric loss (Ridnik et al., ICCV 2021), binary form with the paper's default values."""
    p = torch.sigmoid(logits)
    loss_pos = -((1 - p) ** gamma_pos) * torch.log(p.clamp_min(eps))
    p_m = (p - clip).clamp_min(0)
    loss_neg = -(p_m ** gamma_neg) * torch.log((1 - p_m).clamp_min(eps))
    return torch.where(y > 0.5, loss_pos, loss_neg).mean()


def liganded_negatives(rng, positives, compound_pool, n, exclude):
    """`n` negatives as the default sampler, but with the protein drawn uniformly among proteins that have at least
    one training interaction (proteins without training interactions never appear only as negatives)."""
    prot_pool = np.array(sorted({p for _, p in positives}))
    out, seen = [], set()
    while len(out) < n:
        c = rng.choice(compound_pool, size=2 * (n - len(out)))
        p = prot_pool[rng.randint(0, len(prot_pool), size=c.size)]
        for pair in zip(c.tolist(), p.tolist()):
            if pair not in exclude and pair not in seen:
                out.append(pair)
                seen.add(pair)
                if len(out) == n:
                    break
    return out


def mixed_negatives(rng, positives, compound_pool, num_proteins, n, matched_frac, exclude):
    """`n` training negatives; a fraction `matched_frac` pairs the compound of a training positive with a random
    protein (same exclusion rule), the rest are drawn as before. matched_frac=0 reproduces sample_negatives."""
    n_m = int(round(matched_frac * n))
    if n_m == 0:
        return sample_negatives(rng, compound_pool, num_proteins, n, exclude)
    out, seen = [], set()
    while len(out) < n_m:
        c = positives[rng.randint(len(positives))][0]
        p = int(rng.randint(num_proteins))
        if (c, p) not in exclude and (c, p) not in seen:
            out.append((c, p))
            seen.add((c, p))
    rest = [q for q in sample_negatives(rng, compound_pool, num_proteins, n - n_m + len(seen), exclude) if q not in seen]
    return out + rest[:n - n_m]


def soft_neighbors(model, x, edge_index, edge_attr, node_types):
    """Compound and protein soft-edge neighbours from the frozen Phase 2 scorers on the training graph (no labels)."""
    model.eval()
    with torch.no_grad():
        ei2, _ = model._soft_homogeneous_edges(x, edge_index, edge_attr, node_types)
    new = ei2[:, edge_index.size(1):].cpu().numpy()
    nt = node_types.cpu().numpy()
    local = np.full(len(nt), -1, dtype=np.int64)
    local[nt == 0] = np.arange((nt == 0).sum())
    local[nt == 1] = np.arange((nt == 1).sum())
    nbr_c, nbr_p = {}, {}
    for a, b in new.T:
        target = nbr_c if nt[a] == 0 else nbr_p
        target.setdefault(int(local[a]), []).append(int(local[b]))
    return nbr_c, nbr_p


def soft_neighbor_negatives(rng, positives, nbr_c, nbr_p, n, exclude):
    """Hard negatives from soft-edge neighbours: for a random training interaction (c, p), (c, p') with p' a protein
    soft-edge neighbour of p, or (c', p) with c' a compound soft-edge neighbour of c; training interactions and
    the evaluation reserve are excluded as for every other negative."""
    out, seen, tries = [], set(), 0
    while len(out) < n and tries < 50 * n:
        tries += 1
        c, p = positives[rng.randint(len(positives))]
        if rng.rand() < 0.5 and nbr_p.get(p):
            pair = (c, nbr_p[p][rng.randint(len(nbr_p[p]))])
        elif nbr_c.get(c):
            pair = (nbr_c[c][rng.randint(len(nbr_c[c]))], p)
        else:
            continue
        if pair not in exclude and pair not in seen:
            out.append(pair)
            seen.add(pair)
    return out


def pairwise_rank_loss(logits, y):
    """Mean softplus(s_neg - s_pos) over all positive-negative pairs of the batch (AUC surrogate)."""
    pos, neg = logits[y > 0.5], logits[y < 0.5]
    if pos.numel() == 0 or neg.numel() == 0:
        return logits.new_zeros(())
    return F.softplus(neg.unsqueeze(0) - pos.unsqueeze(1)).mean()


def main():
    ap = argparse.ArgumentParser(description="GIFT Phase 3a: inductive fusion classifier")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--results_dir", type=str, default="legacy_results/seed_42")
    ap.add_argument("--phase2_checkpoint", type=str, required=True)
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--batch_size", type=int, default=512)
    ap.add_argument("--neg_ratio", type=int, default=5)
    ap.add_argument("--iso_fraction", type=float, default=0.35)
    ap.add_argument("--protein_mask_fraction", type=float, default=0.0)
    ap.add_argument("--matched_frac", type=float, default=0.0)
    ap.add_argument("--rank_weight", type=float, default=0.0)
    ap.add_argument("--protein_cp_mask_fraction", type=float, default=0.0,
                    help="protein interaction-edge masking: fraction of proteins whose compound-protein edges (not STRING edges) are masked per step")
    ap.add_argument("--n_heads", type=int, default=1)
    ap.add_argument("--desc_aux_weight", type=float, default=0.0,
                    help="weight of the auxiliary descriptor-only BCE (0 = off)")
    ap.add_argument("--descriptor_only", action="store_true",
                    help="Descriptor-only control: disable the graph path in the fusion (descriptor towers + pair head only)")
    ap.add_argument("--potency_weight", type=float, default=0.0,
                    help="weight of the auxiliary pChEMBL regression on training interactions (0 = off)")
    ap.add_argument("--ban", action="store_true", help="bilinear attention interaction term")
    ap.add_argument("--lig_attn", action="store_true", help="known-ligand attention (scored compound excluded)")
    ap.add_argument("--soft_ccpp", action="store_true", help="label-free CC/PP soft edges in every forward pass")
    ap.add_argument("--soft_real_flag", action="store_true", help="connectivity flag from real edges only (with --soft_ccpp)")
    ap.add_argument("--softneg_frac", type=float, default=0.0,
                    help="fraction of training negatives drawn from soft-edge neighbours of training interactions")
    ap.add_argument("--nbr_lig", action="store_true", help="neighbour-ligand attention (implies --lig_attn)")
    ap.add_argument("--struct_enc", action="store_true", help="structure-aware feature encoders in the pair head")
    ap.add_argument("--unc_out", action="store_true", help="H1: learned per-pair output scale and offset")
    ap.add_argument("--prop_bias", action="store_true", help="learned compound/protein propensity terms")
    ap.add_argument("--unc_node", action="store_true", help="H1-node: per-node learned output scale and offset")
    ap.add_argument("--unc_hetero", action="store_true", help="H1-unc: learned per-pair uncertainty (probit approximation)")
    ap.add_argument("--evidential", action="store_true", help="binary evidential (Beta) head with the EDL loss")
    ap.add_argument("--edl_exp", action="store_true", help="with --evidential: exponential evidence activation")
    ap.add_argument("--nnpu_eta", type=float, default=0.0, help="nnPU loss with this contamination rate (0 = BCE)")
    ap.add_argument("--popneg", action="store_true", help="popularity-matched training negatives")
    ap.add_argument("--liganded_negs", action="store_true", help="negatives only at proteins with training interactions")
    ap.add_argument("--asl", action="store_true", help="asymmetric loss (Ridnik et al. 2021, default values)")
    ap.add_argument("--masked_pos_weight", type=float, default=0.0, help="A1: weight of positives with a masked endpoint")
    ap.add_argument("--hard_pos_focus", action="store_true", help="A2: positive weight 1 + k (1 - p)^2")
    ap.add_argument("--hard_pos_strength", type=float, default=1.0, help="A2: k")
    ap.add_argument("--hard_pos_masked_only", action="store_true", help="A2 only on positives with a masked endpoint")
    ap.add_argument("--l2rw", action="store_true", help="meta-learned per-pair weights on a rotating inner training slice")
    ap.add_argument("--sam", action="store_true", help="sharpness-aware minimisation (rho 0.05)")
    ap.add_argument("--edl_mse", action="store_true", help="with --evidential: squared-error EDL loss (Sensoy eq. 5)")
    ap.add_argument("--mean_teacher", action="store_true",
                    help="Mean-Teacher consistency: EMA teacher on the unmasked graph, student under masking, on labelled "
                         "+ as many unlabelled training-pool pairs (decay 0.99, weight 1)")
    ap.add_argument("--subthreshold_file", type=str, default="",
                    help="option 1: measured sub-threshold TRAINING pairs (c, p, pChEMBL) added with soft targets "
                         "sigmoid((pChEMBL - 5) / 0.5)")
    ap.add_argument("--l2rw_meta_fraction", type=float, default=0.15)
    ap.add_argument("--learn_pos_weight", action="store_true",
                    help="learn the positive class weight on an inner split of the training interactions "
                         "(meta-objective: pairwise ROC-AUC surrogate); writes learned_pos_weight.json")
    ap.add_argument("--meta_fraction", type=float, default=0.15)
    ap.add_argument("--learn_a2_strength", action="store_true",
                    help="option 2: meta-learn the A2 strength k on an inner training split; writes learned_a2_strength.json")
    ap.add_argument("--pos_weight", type=float, default=0.0, help="positive class weight (0 = neg_ratio, published)")
    ap.add_argument("--pair_d", type=int, default=192)
    ap.add_argument("--pair_hidden", type=int, default=384)
    ap.add_argument("--pair_dropout", type=float, default=0.25)
    ap.add_argument("--test_ratio", type=float, default=0.15)
    ap.add_argument("--val_ratio", type=float, default=0.15)
    ap.add_argument("--graph_path", type=str, default=GRAPH_PATH)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)
    os.makedirs(args.results_dir, exist_ok=True)

    with open(args.graph_path, "rb") as f:
        data = pickle.load(f)
    # Same call sequence as earlier trainers so the train/val/test split is identical.
    with contextlib.redirect_stdout(io.StringIO()):
        analyze_meaningful_validation_categories(data)
        train_pairs, val_pairs, test_pairs = split_interaction_data(data, test_ratio=args.test_ratio, val_ratio=args.val_ratio)
    train_pairs = [tuple(map(int, p)) for p in train_pairs]
    val_pairs = [tuple(map(int, p)) for p in val_pairs]
    # Held-out val+test CP pairs are never present in message passing.
    graph = remove_cp_pairs(data, val_pairs + [tuple(map(int, p)) for p in test_pairs])

    num_compounds, num_proteins = int(data["num_compounds"]), int(data["num_proteins"])
    input_dim = data["node_features"].shape[1]
    edge_dim = data["edge_attr"].shape[1]
    model = GIFT(
        input_dim=input_dim, hidden_dim=128, output_dim=64, num_layers=3, edge_dim=edge_dim,
        use_edge_mask=True, edge_dropout_p=0.15, gate_hidden_dim=16,
        feature_mean=data["node_features"].mean(axis=0), feature_std=data["node_features"].std(axis=0),
        pair_d=args.pair_d, pair_hidden=args.pair_hidden, pair_dropout=args.pair_dropout,
        n_heads=args.n_heads, desc_aux=args.desc_aux_weight > 0, potency=args.potency_weight > 0,
        ban=args.ban, lig_attn=args.lig_attn, soft_ccpp=args.soft_ccpp, nbr_lig=args.nbr_lig,
        soft_real_flag=args.soft_real_flag, struct_enc=args.struct_enc, unc_out=args.unc_out, prop_bias=args.prop_bias, unc_node=args.unc_node, unc_hetero=args.unc_hetero, evidential=args.evidential,         edl_exp=args.edl_exp, descriptor_only=args.descriptor_only,
    ).to(device)

    backbone = {k: v for k, v in torch.load(args.phase2_checkpoint, map_location=device).items()
                if not k.startswith("pair_classifier.")}
    missing, unexpected = model.load_state_dict(backbone, strict=False)
    assert not unexpected and all(k.startswith("pair_classifier.") for k in missing), (missing, unexpected)
    if args.struct_enc:
        from struct_encoders import set_bit_thresholds
        set_bit_thresholds(model, data["node_features"], np.asarray(data["node_types"]))
    for p in model.parameters():
        p.requires_grad_(False)
    for p in model.pair_classifier.parameters():
        p.requires_grad_(True)
    print(f"Loaded backbone from {args.phase2_checkpoint}; head params: "
          f"{sum(p.numel() for p in model.pair_classifier.parameters())}")

    x = torch.FloatTensor(graph["node_features"]).to(device)
    edge_index = torch.LongTensor(graph["edge_index"]).to(device)
    edge_attr = torch.FloatTensor(graph["edge_attr"]).to(device)
    node_types = torch.from_numpy(data["node_types"]).long().to(device)
    compound_mask, protein_mask = node_types == 0, node_types == 1
    compound_node_ids = torch.nonzero(compound_mask).flatten()
    protein_node_ids = torch.nonzero(protein_mask).flatten()

    def embed(isolate_compounds, isolate_proteins=(), cp_mask_proteins=(), drop_pairs=()):
        """Backbone embeddings with all edges of the given compounds (compound masking) and proteins (protein
        masking) masked, and the compound-protein edges of `cp_mask_proteins` masked (STRING edges kept)."""
        ei, ea = edge_index, edge_attr
        if len(drop_pairs) > 0:
            dc = compound_node_ids[torch.tensor([c for c, _ in drop_pairs], device=device)]
            dp = protein_node_ids[torch.tensor([p for _, p in drop_pairs], device=device)]
            nn_ = x.size(0)
            keep = ~torch.isin(ei[0] * nn_ + ei[1], torch.cat([dc * nn_ + dp, dp * nn_ + dc]))
            ei, ea = ei[:, keep], ea[keep]
        if len(isolate_compounds) > 0 or len(isolate_proteins) > 0:
            nodes = [compound_node_ids[torch.as_tensor(np.asarray(isolate_compounds), dtype=torch.long, device=device)]]
            if len(isolate_proteins) > 0:
                nodes.append(protein_node_ids[torch.as_tensor(np.asarray(isolate_proteins), dtype=torch.long, device=device)])
            ei, ea = build_isolated_edge_index(edge_index, edge_attr, torch.cat(nodes))
        if len(cp_mask_proteins) > 0:
            pn = protein_node_ids[torch.as_tensor(np.asarray(cp_mask_proteins), dtype=torch.long, device=device)]
            cn_mask = node_types[ei[0]] != node_types[ei[1]]
            drop = cn_mask & (torch.isin(ei[0], pn) | torch.isin(ei[1], pn))
            ei, ea = ei[:, ~drop], ea[~drop]
        model.eval()
        with torch.no_grad():
            _, z = model(x, ei, ea, node_types=node_types)
        return z[compound_mask], z[protein_mask]

    # Standardized pChEMBL of the training interactions (training labels only), for the potency auxiliary loss.
    pchembl = dict(data["all_known_pchembl"])
    tr_pot = np.array([pchembl[p] for p in train_pairs], dtype=np.float32)
    pot_mu, pot_sd = float(tr_pot.mean()), float(tr_pot.std() + 1e-6)
    pot_of = {p: (v - pot_mu) / pot_sd for p, v in zip(train_pairs, tr_pot)}

    rng = np.random.RandomState(args.seed)
    positives = set(train_pairs) | set(val_pairs) | {tuple(map(int, p)) for p in test_pairs}
    positives |= {tuple(map(int, p)) for p in data.get("all_known_pairs", [])}
    reserved = reserved_eval_mask(num_compounds, num_proteins)
    train_exclude = training_exclusion(train_pairs, reserved)
    nbr_c, nbr_p = (soft_neighbors(model, x, edge_index, edge_attr, node_types) if args.softneg_frac > 0 else ({}, {}))
    if args.softneg_frac > 0:
        print(f"Soft-edge hard negatives: fraction {args.softneg_frac}, {len(nbr_c)} compounds and {len(nbr_p)} proteins with soft neighbours")
    all_compounds = np.arange(num_compounds)
    val_compounds = sorted({c for c, _ in val_pairs})
    val_negs = sample_negatives(rng, np.asarray(val_compounds), num_proteins, VAL_NEG_RATIO * len(val_pairs),
                                evaluation_exclusion(positives, reserved))
    val_all = val_pairs + val_negs
    y_val = np.array([1] * len(val_pairs) + [0] * len(val_negs))
    vc = torch.tensor([p[0] for p in val_all], device=device)
    vp = torch.tensor([p[1] for p in val_all], device=device)

    log_w = None
    learn_k = args.learn_a2_strength
    if args.learn_pos_weight or learn_k:
        # Inner meta split of the TRAINING interactions only: removed from the supervised pairs and from message
        # passing; the official validation, test and external sets are never used to learn the weight.
        meta_rng = np.random.RandomState(args.seed + 1000)
        meta_idx = set(meta_rng.choice(len(train_pairs), size=int(round(args.meta_fraction * len(train_pairs))),
                                       replace=False).tolist())
        meta_pairs = [p for i, p in enumerate(train_pairs) if i in meta_idx]
        train_pairs = [p for i, p in enumerate(train_pairs) if i not in meta_idx]
        mc = compound_node_ids[torch.tensor([c for c, _ in meta_pairs], device=device)]
        mp = protein_node_ids[torch.tensor([p for _, p in meta_pairs], device=device)]
        n_nodes = x.size(0)
        drop = torch.cat([mc * n_nodes + mp, mp * n_nodes + mc])
        keep = ~torch.isin(edge_index[0] * n_nodes + edge_index[1], drop)
        edge_index, edge_attr = edge_index[:, keep], edge_attr[keep]
        meta_negs = sample_negatives(meta_rng, all_compounds, num_proteins, len(meta_pairs), train_exclude)
        meta_c_pos = torch.tensor([c for c, _ in meta_pairs], device=device)
        meta_p_pos = torch.tensor([p for _, p in meta_pairs], device=device)
        meta_c_neg = torch.tensor([c for c, _ in meta_negs], device=device)
        meta_p_neg = torch.tensor([p for _, p in meta_negs], device=device)
        log_w = torch.zeros((), device=device, requires_grad=True)  # log of the positive weight, or of the A2 strength k
        opt_w = torch.optim.Adam([log_w], lr=0.05)
        w_trace = []
        print(f"Learning the positive weight: {len(train_pairs)} supervised / {len(meta_pairs)} meta interactions; "
              f"{int((~keep).sum())} meta edges removed from message passing", flush=True)

        def head_args(zc_, zp_, c_idx, p_idx):
            conn_c = (model._compound_log_degree[c_idx] > 0).float()
            conn_p = (model._protein_log_degree[p_idx] > 0).float()
            return (zc_[c_idx], zp_[p_idx], model._cx[c_idx], model._px[p_idx], conn_c, conn_p)

    head = model.pair_classifier
    opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    teacher_head = None
    if args.mean_teacher:
        import copy
        teacher_head = copy.deepcopy(head)
        for p_ in teacher_head.parameters():
            p_.requires_grad_(False)
        mt_rng = np.random.RandomState(args.seed + 3000)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=args.lr * 0.05)
    # Balanced class weighting, matching RF's class_weight="balanced".
    pos_weight = torch.tensor(float(args.pos_weight if args.pos_weight > 0 else args.neg_ratio), device=device)

    def edl_pw(w):
        """EDL positive weight: only when set or learned; the published EDL loss is unweighted."""
        return w if (args.pos_weight > 0 or log_w is not None) else None

    def score_val(isolate):
        zc, zp = embed(isolate)
        head.eval()
        with torch.no_grad():
            _, pr = model.classify_pairs(zc, zp, vc, vp)
        pr = pr.cpu().numpy()
        return roc_auc_score(y_val, pr), average_precision_score(y_val, pr), pr

    l2rw_rng = np.random.RandomState(args.seed + 2000)
    sub_pairs, sub_t = [], None
    if args.subthreshold_file:
        rows = json.load(open(args.subthreshold_file))
        sub_pairs = [(int(c), int(p)) for c, p, _ in rows]
        sub_t = torch.tensor([1.0 / (1.0 + np.exp(-(v - 5.0) / 0.5)) for _, _, v in rows], dtype=torch.float32, device=device)
        print(f"Option 1: {len(sub_pairs)} sub-threshold training pairs with soft targets "
              f"(mean {float(sub_t.mean()):.3f})", flush=True)
    best, patience = -1.0, 0
    ckpt_path = os.path.join(args.results_dir, "best_classifier.pth")
    for epoch in range(args.epochs):
        iso = rng.choice(num_compounds, size=int(round(args.iso_fraction * num_compounds)), replace=False)
        iso_p = (rng.choice(num_proteins, size=int(round(args.protein_mask_fraction * num_proteins)), replace=False)
                 if args.protein_mask_fraction > 0 else [])
        cp_p = (rng.choice(num_proteins, size=int(round(args.protein_cp_mask_fraction * num_proteins)), replace=False)
                if args.protein_cp_mask_fraction > 0 else [])
        sup_pairs = train_pairs
        if args.l2rw:
            midx = set(l2rw_rng.choice(len(train_pairs), size=int(round(args.l2rw_meta_fraction * len(train_pairs))),
                                       replace=False).tolist())
            meta_pairs = [q for i, q in enumerate(train_pairs) if i in midx]
            sup_pairs = [q for i, q in enumerate(train_pairs) if i not in midx]
            meta_negs = sample_negatives(l2rw_rng, all_compounds, num_proteins, len(meta_pairs), train_exclude)
            meta_pos = (torch.tensor([c for c, _ in meta_pairs], device=device), torch.tensor([q for _, q in meta_pairs], device=device))
            meta_neg = (torch.tensor([c for c, _ in meta_negs], device=device), torch.tensor([q for _, q in meta_negs], device=device))
            zc, zp = embed(iso, iso_p, cp_p, drop_pairs=meta_pairs)
        else:
            if teacher_head is not None:
                # Teacher view: unmasked graph; keep its connectivity flags before the masked student view.
                zc_t, zp_t = embed([])
                conn_c_t = (model._compound_log_degree > 0).float()
                conn_p_t = (model._protein_log_degree > 0).float()
            zc, zp = embed(iso, iso_p, cp_p)
        n_neg = args.neg_ratio * len(sup_pairs)
        soft_negs = soft_neighbor_negatives(rng, train_pairs, nbr_c, nbr_p, int(round(args.softneg_frac * n_neg)),
                                            train_exclude) if args.softneg_frac > 0 else []
        negs = soft_negs + (popularity_negatives(rng, train_pairs, n_neg - len(soft_negs), train_exclude) if args.popneg
                            else liganded_negatives(rng, train_pairs, all_compounds, n_neg - len(soft_negs), train_exclude)
                            if args.liganded_negs else mixed_negatives(rng, train_pairs, all_compounds, num_proteins, n_neg - len(soft_negs),
                                                 args.matched_frac, train_exclude))
        pairs = sup_pairs + negs + sub_pairs
        y = torch.tensor([1.0] * len(sup_pairs) + [0.0] * len(negs), device=device)
        if sub_pairs:
            y = torch.cat([y, sub_t])
        perm = torch.randperm(len(pairs), device=device)
        tc = torch.tensor([p[0] for p in pairs], device=device)[perm]
        tp = torch.tensor([p[1] for p in pairs], device=device)[perm]
        y = y[perm]
        masked_all = torch.isin(tc, torch.as_tensor(np.asarray(iso), device=device, dtype=torch.long))
        if len(iso_p) > 0:
            masked_all |= torch.isin(tp, torch.as_tensor(np.asarray(iso_p), device=device, dtype=torch.long))
        pot = torch.tensor([pot_of.get(p, 0.0) for p in pairs], dtype=torch.float32, device=device)[perm]

        head.train()
        total = 0.0
        for i in range(0, len(pairs), args.batch_size):
            sl = slice(i, i + args.batch_size)
            if log_w is not None:
                # One-step look-ahead hypergradient (first-order meta-learning of the class weight).
                params = dict(head.named_parameters())
                lg = torch.func.functional_call(head, params, head_args(zc, zp, tc[sl], tp[sl]))
                if learn_k:
                    wpos = 1 + torch.exp(log_w) * (1 - torch.sigmoid(lg.detach())) ** 2
                else:
                    wpos = torch.exp(log_w)
                # Same as BCE-with-logits with positive weights, written out so the gradient reaches log_w; with the
                # evidential head, the positive-weighted EDL loss of this functional forward instead.
                if getattr(head, "evidential", False):
                    inner = (edl_mse_loss if args.edl_mse else edl_loss)(*head._last_ab, y[sl],
                                                                         min(1.0, (epoch + 1) / 10.0), pos_weight=wpos)
                else:
                    inner = (wpos * y[sl] * F.softplus(-lg) + (1 - y[sl]) * F.softplus(lg)).mean()
                grads = torch.autograd.grad(inner, list(params.values()), create_graph=True, allow_unused=True)
                fast = {k: (v - args.lr * g if g is not None else v) for (k, v), g in zip(params.items(), grads)}
                bi = torch.randint(0, len(meta_pairs), (512,), device=device)
                bj = torch.randint(0, len(meta_negs), (512,), device=device)
                s_pos = torch.func.functional_call(head, fast, head_args(zc, zp, meta_c_pos[bi], meta_p_pos[bi]))
                s_neg = torch.func.functional_call(head, fast, head_args(zc, zp, meta_c_neg[bj], meta_p_neg[bj]))
                meta = F.softplus(-(s_pos[:, None] - s_neg[None, :])).mean()
                opt_w.zero_grad()
                meta.backward(inputs=[log_w])
                opt_w.step()
                with torch.no_grad():
                    if learn_k:
                        log_w.clamp_(float(np.log(0.01)), float(np.log(50.0)))
                    else:
                        log_w.clamp_(float(np.log(0.1)), float(np.log(20.0)))
                if learn_k:
                    args.hard_pos_focus, args.hard_pos_strength = True, float(torch.exp(log_w))
                else:
                    pos_weight = torch.exp(log_w).detach()
            w_l2rw = (l2rw_weights(model, args.lr, zc, zp, tc[sl], tp[sl], y[sl], meta_pos, meta_neg)
                      if args.l2rw else None)
            logits, _ = model.classify_pairs(zc, zp, tc[sl], tp[sl])
            loss = ((edl_mse_loss if args.edl_mse else edl_loss)(*head._last_ab, y[sl], min(1.0, (epoch + 1) / 10.0),
                                                                 pos_weight=edl_pw(pos_weight))
                    if getattr(head, "evidential", False)
                    else nnpu_loss(logits, y[sl], args.nnpu_eta) if args.nnpu_eta > 0
                    else asl_loss(logits, y[sl]) if args.asl
                    else F.binary_cross_entropy_with_logits(
                        logits, y[sl], pos_weight=pos_weight,
                        weight=(w_l2rw if w_l2rw is not None else
                                positive_sample_weights(logits, y[sl], masked_all[sl], args.masked_pos_weight,
                                                        args.hard_pos_focus, args.hard_pos_strength,
                                                        args.hard_pos_masked_only))))
            if args.rank_weight > 0:
                loss = loss + args.rank_weight * pairwise_rank_loss(logits, y[sl])
            if args.desc_aux_weight > 0:
                aux = model.desc_aux_logits(tc[sl], tp[sl])
                loss = loss + args.desc_aux_weight * F.binary_cross_entropy_with_logits(aux, y[sl], pos_weight=pos_weight)
            if args.potency_weight > 0:
                pos = y[sl] > 0.5
                if pos.any():
                    loss = loss + args.potency_weight * F.mse_loss(model.potency_from_last_pairs()[pos], pot[sl][pos])
            if teacher_head is not None:
                un = sample_negatives(mt_rng, all_compounds, num_proteins, logits.numel(), train_exclude)
                uc = torch.cat([tc[sl], torch.tensor([c for c, _ in un], device=device)])
                up = torch.cat([tp[sl], torch.tensor([q for _, q in un], device=device)])
                s_logit, _ = model.classify_pairs(zc, zp, uc, up)
                with torch.no_grad():
                    t_logit = teacher_head(zc_t[uc], zp_t[up], model._cx[uc], model._px[up], conn_c_t[uc], conn_p_t[up])
                loss = loss + F.mse_loss(torch.sigmoid(s_logit), torch.sigmoid(t_logit))
            opt.zero_grad()
            loss.backward()
            if args.sam:
                # SAM (Foret et al. 2021): ascend to the worst-case neighbour within rho = 0.05, take the gradient there.
                ps = [q for q in head.parameters() if q.grad is not None]
                gn = torch.norm(torch.stack([q.grad.norm() for q in ps])) + 1e-12
                eps_ = [0.05 * q.grad / gn for q in ps]
                with torch.no_grad():
                    for q, ev in zip(ps, eps_):
                        q.add_(ev)
                opt.zero_grad()
                lg2, _ = model.classify_pairs(zc, zp, tc[sl], tp[sl])
                F.binary_cross_entropy_with_logits(lg2, y[sl], pos_weight=pos_weight).backward()
                with torch.no_grad():
                    for q, ev in zip(ps, eps_):
                        q.sub_(ev)
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            opt.step()
            if teacher_head is not None:
                with torch.no_grad():
                    for pt, ps in zip(teacher_head.parameters(), head.parameters()):
                        pt.mul_(0.99).add_(ps.detach(), alpha=0.01)
            total += loss.item() * logits.numel()
        sched.step()
        if log_w is not None:
            w_trace.append(float(torch.exp(log_w)))
            print(f"  epoch {epoch+1}: learned {'A2 strength' if learn_k else 'positive weight'} {w_trace[-1]:.3f}", flush=True)

        auc_c, ap_c, pr_c = score_val([])
        auc_i, ap_i, _ = score_val(val_compounds)
        sel = 0.5 * (auc_c + auc_i)
        if sel > best:
            best, patience = sel, 0
            torch.save(head.state_dict(), ckpt_path)
        else:
            patience += 1
        if (epoch + 1) % 5 == 0 or patience == 0:
            f1c = f1_score(y_val, (pr_c > 0.5).astype(int), zero_division=0)
            print(f"Epoch {epoch+1:3d}: loss={total/len(pairs):.4f} val_auc(conn,iso)=({auc_c:.4f},{auc_i:.4f}) "
                  f"val_ap(conn,iso)=({ap_c:.4f},{ap_i:.4f}) f1@0.5(conn)={f1c:.4f}{' *' if patience == 0 else ''}", flush=True)
        if patience >= args.patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
    if log_w is not None:
        w_final = float(np.mean(w_trace[-5:]))
        name = "learned_a2_strength" if learn_k else "learned_pos_weight"
        json.dump({name: w_final, "trace": w_trace, "meta_fraction": args.meta_fraction,
                   "n_meta": len(meta_pairs)}, open(os.path.join(args.results_dir, name + ".json"), "w"), indent=1)
        print(f"{name} (mean of last 5 epochs): {w_final:.3f}")

    head.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()
    # Test metrics are computed only by evaluate_chembl35.py (reserved negatives).
    torch.save({
        "model_state_dict": model.state_dict(),
        "model_config": {
            "input_dim": input_dim, "hidden_dim": 128, "output_dim": 64, "num_layers": 3, "edge_dim": edge_dim,
            "use_edge_mask": True, "edge_dropout_p": 0.15, "gate_hidden_dim": 16,
            "pair_d": args.pair_d, "pair_hidden": args.pair_hidden, "pair_dropout": args.pair_dropout,
            "iso_fraction": args.iso_fraction, "neg_ratio": args.neg_ratio,
            "protein_mask_fraction": args.protein_mask_fraction, "matched_frac": args.matched_frac,
            "rank_weight": args.rank_weight, "protein_cp_mask_fraction": args.protein_cp_mask_fraction, "nnpu_eta": args.nnpu_eta, "popneg": args.popneg, "liganded_negs": args.liganded_negs, "asl": args.asl, "masked_pos_weight": args.masked_pos_weight, "hard_pos_focus": args.hard_pos_focus, "hard_pos_strength": args.hard_pos_strength, "hard_pos_masked_only": args.hard_pos_masked_only, "l2rw": args.l2rw, "sam": args.sam, "edl_mse": args.edl_mse, "mean_teacher": args.mean_teacher, "subthreshold_file": args.subthreshold_file, "pos_weight": args.pos_weight,
            "n_heads": args.n_heads, "desc_aux": args.desc_aux_weight > 0, "desc_aux_weight": args.desc_aux_weight, "softneg_frac": args.softneg_frac,
            "potency": args.potency_weight > 0, "potency_weight": args.potency_weight, "ban": args.ban,
            "lig_attn": args.lig_attn or args.nbr_lig, "soft_ccpp": args.soft_ccpp, "nbr_lig": args.nbr_lig, "soft_real_flag": args.soft_real_flag, "struct_enc": args.struct_enc, "unc_out": args.unc_out, "prop_bias": args.prop_bias, "unc_node": args.unc_node, "unc_hetero": args.unc_hetero, "evidential": args.evidential, "edl_exp": args.edl_exp,
            "phase2_checkpoint": args.phase2_checkpoint, "graph_path": args.graph_path,
            "descriptor_only": args.descriptor_only,
        },
        "training_completed": True,
    }, os.path.join(args.results_dir, "trained_model.pth"))
    print(f"Model saved to: {args.results_dir}/trained_model.pth")


if __name__ == "__main__":
    main()
