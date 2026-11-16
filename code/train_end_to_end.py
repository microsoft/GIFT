"""Phase 3b: end-to-end fine-tuning of the encoder together with the inductive fusion classifier.
Leakage control: every training positive is itself a CP edge in the message-passing graph, so a trainable encoder
could learn "edge present => positive". Each step therefore removes the CP edges of the step's target positives
(both directions) before message passing and additionally masks a random fraction of compounds and proteins.
Validation uses the same rule.
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
from train_inductive_common import sample_negatives
from train_classifier import mixed_negatives, nnpu_loss, popularity_negatives, liganded_negatives, asl_loss, edl_loss, edl_mse_loss, positive_sample_weights, l2rw_weights, pairwise_rank_loss, soft_neighbor_negatives, soft_neighbors

GRAPH_PATH = "data/graphs/graph_similarity_input.pkl"
# Validation negatives per positive; fixed so every method is selected on the same validation set.
VAL_NEG_RATIO = 5


def main():
    ap = argparse.ArgumentParser(description="GIFT Phase 3b: end-to-end fine-tuning")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--results_dir", type=str, default="legacy_results/seed_42")
    ap.add_argument("--init_checkpoint", type=str, required=True, help="Phase 3a trained_model.pth")
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--eval_every", type=int, default=10)
    ap.add_argument("--patience", type=int, default=12, help="in evaluation rounds")
    ap.add_argument("--backbone_lr", type=float, default=2e-5)
    ap.add_argument("--head_lr", type=float, default=2e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--target_fraction", type=float, default=0.30)
    ap.add_argument("--iso_fraction", type=float, default=0.35)
    ap.add_argument("--protein_mask_fraction", type=float, default=0.0)
    ap.add_argument("--matched_frac", type=float, default=0.0)
    ap.add_argument("--rank_weight", type=float, default=0.0)
    ap.add_argument("--protein_cp_mask_fraction", type=float, default=0.0)
    # The architecture comes from the Phase 3a checkpoint; accepted here so both phases share one argument list.
    ap.add_argument("--n_heads", type=int, default=1)
    ap.add_argument("--desc_aux_weight", type=float, default=0.0)
    ap.add_argument("--softdistill_weight", type=float, default=0.0,
                    help="soft-edge distillation: masked nodes match a frozen teacher that sees their edges and CC/PP soft edges")
    ap.add_argument("--softdistill_teacher_no_soft", action="store_true",
                    help="ablation: teacher without soft edges (same distillation otherwise)")
    ap.add_argument("--potency_weight", type=float, default=0.0)
    ap.add_argument("--ban", action="store_true")
    ap.add_argument("--lig_attn", action="store_true")
    ap.add_argument("--soft_ccpp", action="store_true")
    ap.add_argument("--soft_real_flag", action="store_true")
    ap.add_argument("--softneg_frac", type=float, default=0.0)
    ap.add_argument("--nbr_lig", action="store_true")
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
    ap.add_argument("--pos_weight", type=float, default=0.0, help="positive class weight (0 = neg_ratio, published)")
    ap.add_argument("--neg_ratio", type=int, default=5)
    ap.add_argument("--test_ratio", type=float, default=0.15)
    ap.add_argument("--val_ratio", type=float, default=0.15)
    ap.add_argument("--graph_path", type=str, default=GRAPH_PATH)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)
    os.makedirs(args.results_dir, exist_ok=True)

    with open(args.graph_path, "rb") as f:
        data = pickle.load(f)
    with contextlib.redirect_stdout(io.StringIO()):
        analyze_meaningful_validation_categories(data)
        train_pairs, val_pairs, test_pairs = split_interaction_data(data, test_ratio=args.test_ratio, val_ratio=args.val_ratio)
    train_pairs = [tuple(map(int, p)) for p in train_pairs]
    val_pairs = [tuple(map(int, p)) for p in val_pairs]
    # Held-out val+test CP pairs are never present in message passing.
    graph = remove_cp_pairs(data, val_pairs + [tuple(map(int, p)) for p in test_pairs])

    init = torch.load(args.init_checkpoint, map_location=device, weights_only=False)
    cfg = dict(init["model_config"])
    num_compounds, num_proteins = int(data["num_compounds"]), int(data["num_proteins"])
    model = GIFT(
        input_dim=cfg["input_dim"], hidden_dim=cfg["hidden_dim"], output_dim=cfg["output_dim"],
        num_layers=cfg["num_layers"], edge_dim=cfg["edge_dim"], use_edge_mask=True,
        edge_dropout_p=cfg["edge_dropout_p"], gate_hidden_dim=cfg["gate_hidden_dim"],
        pair_d=cfg["pair_d"], pair_hidden=cfg["pair_hidden"], pair_dropout=cfg["pair_dropout"],
        n_heads=cfg.get("n_heads", 1), desc_aux=cfg.get("desc_aux", False), potency=cfg.get("potency", False),
        ban=cfg.get("ban", False), lig_attn=cfg.get("lig_attn", False), soft_ccpp=cfg.get("soft_ccpp", False), soft_real_flag=cfg.get("soft_real_flag", False),
        nbr_lig=cfg.get("nbr_lig", False), struct_enc=cfg.get("struct_enc", False), unc_out=cfg.get("unc_out", False), prop_bias=cfg.get("prop_bias", False), unc_node=cfg.get("unc_node", False), unc_hetero=cfg.get("unc_hetero", False), evidential=cfg.get("evidential", False), edl_exp=cfg.get("edl_exp", False),
        descriptor_only=cfg.get("descriptor_only", False),
    ).to(device)
    model.load_state_dict(init["model_state_dict"], strict=True)
    print(f"Initialized from {args.init_checkpoint}")
    teacher = None
    if args.softdistill_weight > 0:
        # Frozen copy of the Phase 3a model (its encoder is the Phase 2 encoder) with the label-free CC/PP soft edges
        # of the frozen Phase 2 scorers; used only for training targets, never at validation or inference.
        teacher = GIFT(
            input_dim=cfg["input_dim"], hidden_dim=cfg["hidden_dim"], output_dim=cfg["output_dim"],
            num_layers=cfg["num_layers"], edge_dim=cfg["edge_dim"], use_edge_mask=True,
            edge_dropout_p=cfg["edge_dropout_p"], gate_hidden_dim=cfg["gate_hidden_dim"],
            pair_d=cfg["pair_d"], pair_hidden=cfg["pair_hidden"], pair_dropout=cfg["pair_dropout"],
            n_heads=cfg.get("n_heads", 1), desc_aux=cfg.get("desc_aux", False), potency=cfg.get("potency", False),
            soft_ccpp=not args.softdistill_teacher_no_soft,
        ).to(device)
        teacher.load_state_dict(init["model_state_dict"], strict=True)
        teacher.eval()
        for p_ in teacher.parameters():
            p_.requires_grad_(False)
        print(f"Soft-edge distillation: weight {args.softdistill_weight}, teacher soft edges: {not args.softdistill_teacher_no_soft}")

    x = torch.FloatTensor(graph["node_features"]).to(device)
    edge_index = torch.LongTensor(graph["edge_index"]).to(device)
    edge_attr = torch.FloatTensor(graph["edge_attr"]).to(device)
    node_types = torch.from_numpy(data["node_types"]).long().to(device)
    compound_mask, protein_mask = node_types == 0, node_types == 1
    compound_node_ids = torch.nonzero(compound_mask).flatten()
    protein_node_ids = torch.nonzero(protein_mask).flatten()
    n_nodes = x.size(0)
    edge_keys = edge_index[0] * n_nodes + edge_index[1]

    def graph_without(pairs, isolate_compounds, isolate_proteins=(), cp_mask_proteins=()):
        """Edge set with the CP edges of `pairs` (both directions) and all edges of `isolate_compounds` removed."""
        keep = torch.ones(edge_index.size(1), dtype=torch.bool, device=device)
        if len(pairs) > 0:
            pc = torch.as_tensor([p[0] for p in pairs], device=device)
            pp = torch.as_tensor([p[1] for p in pairs], device=device)
            cn, pn = compound_node_ids[pc], protein_node_ids[pp]
            drop = torch.cat([cn * n_nodes + pn, pn * n_nodes + cn])
            keep &= ~torch.isin(edge_keys, drop)
        if len(isolate_compounds) > 0:
            iso = compound_node_ids[torch.as_tensor(np.asarray(isolate_compounds), dtype=torch.long, device=device)]
            keep &= ~(torch.isin(edge_index[0], iso) | torch.isin(edge_index[1], iso))
        if len(isolate_proteins) > 0:
            isp = protein_node_ids[torch.as_tensor(np.asarray(isolate_proteins), dtype=torch.long, device=device)]
            keep &= ~(torch.isin(edge_index[0], isp) | torch.isin(edge_index[1], isp))
        if len(cp_mask_proteins) > 0:
            cpp = protein_node_ids[torch.as_tensor(np.asarray(cp_mask_proteins), dtype=torch.long, device=device)]
            cross = node_types[edge_index[0]] != node_types[edge_index[1]]
            keep &= ~(cross & (torch.isin(edge_index[0], cpp) | torch.isin(edge_index[1], cpp)))
        return edge_index[:, keep], edge_attr[keep]

    pchembl = dict(data["all_known_pchembl"])
    tr_pot = np.array([pchembl[p] for p in train_pairs], dtype=np.float32)
    pot_mu, pot_sd = float(tr_pot.mean()), float(tr_pot.std() + 1e-6)

    rng = np.random.RandomState(args.seed)
    positives = set(train_pairs) | set(val_pairs) | {tuple(map(int, p)) for p in test_pairs}
    positives |= {tuple(map(int, p)) for p in data.get("all_known_pairs", [])}
    reserved = reserved_eval_mask(num_compounds, num_proteins)
    train_exclude = training_exclusion(train_pairs, reserved)
    nbr_c, nbr_p = (soft_neighbors(model, x, edge_index, edge_attr, node_types) if args.softneg_frac > 0 else ({}, {}))
    all_compounds = np.arange(num_compounds)
    val_compounds = sorted({c for c, _ in val_pairs})
    val_negs = sample_negatives(rng, np.asarray(val_compounds), num_proteins, VAL_NEG_RATIO * len(val_pairs),
                                evaluation_exclusion(positives, reserved))
    val_all = val_pairs + val_negs
    y_val = np.array([1] * len(val_pairs) + [0] * len(val_negs))
    vc = torch.tensor([p[0] for p in val_all], device=device)
    vp = torch.tensor([p[1] for p in val_all], device=device)

    head_params = list(model.pair_classifier.parameters())
    head_ids = {id(p) for p in head_params}
    backbone_params = [p for p in model.parameters() if id(p) not in head_ids]
    opt = torch.optim.AdamW([
        {"params": backbone_params, "lr": args.backbone_lr},
        {"params": head_params, "lr": args.head_lr},
    ], weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps, eta_min=0.0)
    pos_weight = torch.tensor(float(args.pos_weight if args.pos_weight > 0 else args.neg_ratio), device=device)
    teacher_mt = None
    if args.mean_teacher:
        import copy
        teacher_mt = copy.deepcopy(model)
        for p_ in teacher_mt.parameters():
            p_.requires_grad_(False)
        mt_rng = np.random.RandomState(args.seed + 3000)

    def score_val(isolate):
        # Val target edges are always removed, so neither view can read the label from the graph.
        ei, ea = graph_without(val_pairs, isolate)
        model.eval()
        with torch.no_grad():
            _, z = model(x, ei, ea, node_types=node_types)
            _, pr = model.classify_pairs(z[compound_mask], z[protein_mask], vc, vp)
        pr = pr.cpu().numpy()
        return roc_auc_score(y_val, pr), average_precision_score(y_val, pr), pr

    auc_c, ap_c, _ = score_val([])
    auc_i, ap_i, _ = score_val(val_compounds)
    best = 0.5 * (auc_c + auc_i)
    print(f"Init (Phase 3a): val_auc(conn,iso)=({auc_c:.4f},{auc_i:.4f}) val_ap(conn,iso)=({ap_c:.4f},{ap_i:.4f})")
    ckpt_path = os.path.join(args.results_dir, "best_model.pth")
    torch.save(model.state_dict(), ckpt_path)

    patience = 0
    l2rw_rng = np.random.RandomState(args.seed + 2000)
    sub_pairs, sub_t = [], None
    if args.subthreshold_file:
        rows = json.load(open(args.subthreshold_file))
        sub_pairs = [(int(c), int(p)) for c, p, _ in rows]
        sub_t = torch.tensor([1.0 / (1.0 + np.exp(-(v - 5.0) / 0.5)) for _, _, v in rows], dtype=torch.float32, device=device)
        print(f"Option 1: {len(sub_pairs)} sub-threshold training pairs with soft targets", flush=True)
    n_targets = int(round(args.target_fraction * len(train_pairs)))
    for step in range(1, args.steps + 1):
        tgt_idx = rng.choice(len(train_pairs), size=n_targets, replace=False)
        targets = [train_pairs[i] for i in tgt_idx]
        iso = rng.choice(num_compounds, size=int(round(args.iso_fraction * num_compounds)), replace=False)
        iso_p = (rng.choice(num_proteins, size=int(round(args.protein_mask_fraction * num_proteins)), replace=False)
                 if args.protein_mask_fraction > 0 else [])
        cp_p = (rng.choice(num_proteins, size=int(round(args.protein_cp_mask_fraction * num_proteins)), replace=False)
                if args.protein_cp_mask_fraction > 0 else [])
        if args.l2rw:
            tset = set(targets)
            pool = [q for q in train_pairs if q not in tset]
            mi = l2rw_rng.choice(len(pool), size=min(len(pool), int(round(args.l2rw_meta_fraction * len(train_pairs)))),
                                 replace=False)
            meta_pairs = [pool[i] for i in mi]
            meta_negs = sample_negatives(l2rw_rng, all_compounds, num_proteins, len(meta_pairs), train_exclude)
            meta_pos = (torch.tensor([c for c, _ in meta_pairs], device=device), torch.tensor([q for _, q in meta_pairs], device=device))
            meta_neg = (torch.tensor([c for c, _ in meta_negs], device=device), torch.tensor([q for _, q in meta_negs], device=device))
            ei, ea = graph_without(targets + meta_pairs, iso, iso_p, cp_p)
        else:
            ei, ea = graph_without(targets, iso, iso_p, cp_p)
        n_neg = args.neg_ratio * len(targets)
        soft_negs = soft_neighbor_negatives(rng, train_pairs, nbr_c, nbr_p, int(round(args.softneg_frac * n_neg)),
                                            train_exclude) if args.softneg_frac > 0 else []
        negs = soft_negs + (popularity_negatives(rng, train_pairs, n_neg - len(soft_negs), train_exclude) if args.popneg
                            else liganded_negatives(rng, train_pairs, all_compounds, n_neg - len(soft_negs), train_exclude)
                            if args.liganded_negs else mixed_negatives(rng, train_pairs, all_compounds, num_proteins, n_neg - len(soft_negs),
                                                 args.matched_frac, train_exclude))
        pairs = targets + negs + sub_pairs
        tc = torch.tensor([p[0] for p in pairs], device=device)
        tp = torch.tensor([p[1] for p in pairs], device=device)
        y = torch.tensor([1.0] * len(targets) + [0.0] * len(negs), device=device)
        if sub_pairs:
            y = torch.cat([y, sub_t])

        model.train()
        _, z = model(x, ei, ea, node_types=node_types)
        logits, _ = model.classify_pairs(z[compound_mask], z[protein_mask], tc, tp)
        w_l2rw = (l2rw_weights(model, args.head_lr, z[compound_mask], z[protein_mask], tc, tp, y, meta_pos, meta_neg)
                  if args.l2rw else None)
        masked_e2e = torch.isin(tc, torch.as_tensor(np.asarray(iso), device=device, dtype=torch.long))
        if len(iso_p) > 0:
            masked_e2e |= torch.isin(tp, torch.as_tensor(np.asarray(iso_p), device=device, dtype=torch.long))
        edl_pw = pos_weight if args.pos_weight > 0 else None  # published EDL loss is unweighted
        loss = ((edl_mse_loss if args.edl_mse else edl_loss)(*model.pair_classifier._last_ab, y, 1.0, pos_weight=edl_pw)
                if getattr(model.pair_classifier, "evidential", False)
                else nnpu_loss(logits, y, args.nnpu_eta) if args.nnpu_eta > 0
                else asl_loss(logits, y) if args.asl
                else F.binary_cross_entropy_with_logits(
                    logits, y, pos_weight=pos_weight,
                    weight=(w_l2rw if w_l2rw is not None else
                            positive_sample_weights(logits, y, masked_e2e, args.masked_pos_weight, args.hard_pos_focus,
                                                    args.hard_pos_strength, args.hard_pos_masked_only))))
        if teacher is not None:
            # Teacher graph: the step's supervision targets removed (as for the student), no node masking.
            ei_t, ea_t = graph_without(targets, [])
            with torch.no_grad():
                _, z_t = teacher(x, ei_t, ea_t, node_types=node_types)
            masked = [compound_node_ids[torch.as_tensor(np.asarray(iso), dtype=torch.long, device=device)]]
            if len(iso_p) > 0:
                masked.append(protein_node_ids[torch.as_tensor(np.asarray(iso_p), dtype=torch.long, device=device)])
            nodes = torch.cat(masked)
            loss = loss + args.softdistill_weight * (1.0 - F.cosine_similarity(z[nodes], z_t[nodes], dim=1)).mean()
        if args.rank_weight > 0:
            loss = loss + args.rank_weight * pairwise_rank_loss(logits, y)
        if args.desc_aux_weight > 0:
            aux = model.desc_aux_logits(tc, tp)
            loss = loss + args.desc_aux_weight * F.binary_cross_entropy_with_logits(aux, y, pos_weight=pos_weight)
        if args.potency_weight > 0:
            pot = torch.tensor([(pchembl[p] - pot_mu) / pot_sd for p in targets], dtype=torch.float32, device=device)
            loss = loss + args.potency_weight * F.mse_loss(model.potency_from_last_pairs()[:len(targets)], pot)
        if teacher_mt is not None:
            un = sample_negatives(mt_rng, all_compounds, num_proteins, len(pairs), train_exclude)
            uc = torch.cat([tc, torch.tensor([c for c, _ in un], device=device)])
            up = torch.cat([tp, torch.tensor([q for _, q in un], device=device)])
            s_logit, _ = model.classify_pairs(z[compound_mask], z[protein_mask], uc, up)
            ei_t, ea_t = graph_without(targets, [])
            teacher_mt.eval()
            with torch.no_grad():
                _, z_tm = teacher_mt(x, ei_t, ea_t, node_types=node_types)
                t_logit, _ = teacher_mt.classify_pairs(z_tm[compound_mask], z_tm[protein_mask], uc, up)
            loss = loss + F.mse_loss(torch.sigmoid(s_logit), torch.sigmoid(t_logit))
        opt.zero_grad()
        loss.backward()
        if args.sam:
            ps = [q for q in model.parameters() if q.grad is not None]
            gn = torch.norm(torch.stack([q.grad.norm() for q in ps])) + 1e-12
            eps_ = [0.05 * q.grad / gn for q in ps]
            with torch.no_grad():
                for q, ev in zip(ps, eps_):
                    q.add_(ev)
            opt.zero_grad()
            _, z2 = model(x, ei, ea, node_types=node_types)
            lg2, _ = model.classify_pairs(z2[compound_mask], z2[protein_mask], tc, tp)
            (edl_loss(*model.pair_classifier._last_ab, y, 1.0, pos_weight=edl_pw)
             if getattr(model.pair_classifier, "evidential", False)
             else F.binary_cross_entropy_with_logits(lg2, y, pos_weight=pos_weight)).backward()
            with torch.no_grad():
                for q, ev in zip(ps, eps_):
                    q.sub_(ev)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if teacher_mt is not None:
            with torch.no_grad():
                for pt, ps in zip(teacher_mt.parameters(), model.parameters()):
                    pt.mul_(0.99).add_(ps.detach(), alpha=0.01)
                for bt, bs in zip(teacher_mt.buffers(), model.buffers()):
                    bt.copy_(bs)
        sched.step()

        if step % args.eval_every == 0:
            auc_c, ap_c, pr_c = score_val([])
            auc_i, ap_i, _ = score_val(val_compounds)
            sel = 0.5 * (auc_c + auc_i)
            improved = sel > best
            if improved:
                best, patience = sel, 0
                torch.save(model.state_dict(), ckpt_path)
            else:
                patience += 1
            f1c = f1_score(y_val, (pr_c > 0.5).astype(int), zero_division=0)
            print(f"Step {step:4d}: loss={loss.item():.4f} val_auc(conn,iso)=({auc_c:.4f},{auc_i:.4f}) "
                  f"val_ap(conn,iso)=({ap_c:.4f},{ap_i:.4f}) f1@0.5(conn)={f1c:.4f}{' *' if improved else ''}", flush=True)
            if patience >= args.patience:
                print(f"Early stopping at step {step}")
                break

    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()
    # Test metrics are computed only by evaluate_chembl35.py (reserved negatives).
    cfg.update({
        "init_checkpoint": args.init_checkpoint, "backbone_lr": args.backbone_lr, "head_lr": args.head_lr,
        "target_fraction": args.target_fraction, "iso_fraction": args.iso_fraction, "steps": args.steps,
        "protein_mask_fraction": args.protein_mask_fraction, "matched_frac": args.matched_frac,
        "rank_weight": args.rank_weight, "protein_cp_mask_fraction": args.protein_cp_mask_fraction,
        "softdistill_weight": args.softdistill_weight, "softneg_frac": args.softneg_frac, "softdistill_teacher_no_soft": args.softdistill_teacher_no_soft, "nnpu_eta": args.nnpu_eta, "popneg": args.popneg, "liganded_negs": args.liganded_negs, "asl": args.asl, "masked_pos_weight": args.masked_pos_weight, "hard_pos_focus": args.hard_pos_focus, "hard_pos_strength": args.hard_pos_strength, "hard_pos_masked_only": args.hard_pos_masked_only, "l2rw": args.l2rw, "sam": args.sam, "edl_mse": args.edl_mse, "mean_teacher": args.mean_teacher, "subthreshold_file": args.subthreshold_file, "pos_weight": args.pos_weight,
        "graph_path": args.graph_path,
    })
    torch.save({"model_state_dict": model.state_dict(), "model_config": cfg,
                "training_completed": True}, os.path.join(args.results_dir, "trained_model.pth"))
    print(f"Model saved to: {args.results_dir}/trained_model.pth")


if __name__ == "__main__":
    main()
