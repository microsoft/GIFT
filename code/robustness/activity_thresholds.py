"""Point 12: frozen models evaluated at stronger activity thresholds (median pChEMBL >= 5, 6, 7). No retraining.

Per run (seed 42-71) the three benchmarks of the manuscript are rebuilt with the exact RNG streams of
evaluate_chembl35.py / evaluate_external.py / train_rf_baseline.py
(asserted equal to the saved DrugBAN benchmark pairs), every saved model is scored once (CPU), and the scores are
cached in cache/scores_seed_{s}.npz. At threshold T the positives are relabelled from the per-pair median pChEMBL:
  ChEMBL35 test pairs : all_known_pchembl of data/graphs/graph_seed{s}.pkl (build_run_graphs.py)
  external ChEMBL36   : median pChEMBL of data/external_compounds_labels.pkl (build_external_labels.py),
                        exposed by vs.load_external_chembl36_interactions_from_pkls as meta[pair]['max_pchembl']
Variants:  exclude (primary) - positives with 5 <= p < T are dropped, negatives unchanged
           relabel           - positives with 5 <= p < T become negatives
Models: GIFT (results/gift/run_{s}/e2e), descriptor-only GIFT control, RF (saved joblib), GCN/GraphSAGE/GAT
(saved), DeepDTA/GraphDTA (saved), DrugBAN (saved per-pair scores, results/gift/baseline_models/drugban/run_{s}/drugban_dgl/scores.npz).
At T=5 every model's ROC-AUC/PR-AUC is compared with the saved run results (reproduction check).
Usage: python code/robustness/activity_thresholds.py [--seeds ...]
"""
import os as _os, sys as _sys  # release bootstrap: run from any directory as python code/robustness/<script>.py
_REPO = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", ".."))
_sys.path[:0] = [_os.path.join(_REPO, "code"), _os.path.join(_REPO, "code", "robustness")]
_os.chdir(_REPO)
import argparse
import json
import os
import pickle
import sys

import joblib
import numpy as np
import torch
from sklearn.metrics import average_precision_score, matthews_corrcoef, roc_auc_score

sys.path.insert(0, os.getcwd())
import pipeline_utils as vs  # noqa: E402
from graph_cleaning import clean_graph  # noqa: E402
from evaluate_chembl35 import load_gift_model  # noqa: E402
from evaluate_external import external_matched_negatives, per_target_roc_auc  # noqa: E402
from negative_pool import (dti_random_negatives, evaluation_exclusion, reserved_eval_mask,  # noqa: E402
                               sample_compound_matched_negatives, sample_negatives)

D = "results/robustness/measured_activity_thresholds"
CACHE = f"{D}/cache"
OUT = f"{D}/pt12_thresholds.json"
BENCH = ["same_version", "same_version_matched", "external"]
MODELS = ["gift", "descriptor_only", "rf", "gcn", "sage", "gat", "deepdta", "graphdta", "drugban", "dtiam"]
DTIAM = "results/robustness/dtiam/runs/run_{s}"
THR = [5.0, 6.0, 7.0]


def build(s):
    data = pickle.load(open(f"data/graphs/graph_seed{s}.pkl", "rb"))
    nc, npr = int(data["num_compounds"]), int(data["num_proteins"])
    sp = vs.derive_chembl35_splits(data, min_training_pchembl=5.0)
    train, val, test, known = sp["train_pairs"], sp["val_pairs"], sp["test_pairs"], sp["all_positive_pairs"]
    reserved = reserved_eval_mask(nc, npr)
    excl = evaluation_exclusion(known, reserved)
    rng = np.random.RandomState(vs.SEED)
    sample_negatives(rng, np.arange(nc), npr, 5 * len(test), excl)
    matched = sample_compound_matched_negatives(rng, test, npr, 5, excl)
    random_neg = dti_random_negatives(vs.SEED, train + val + test, npr, 5 * len(test), excl)
    aug, cross, meta = vs.load_external_chembl36_interactions_from_pkls(data)
    ext_neg = external_matched_negatives(cross, npr, tested_pairs=aug.get("external_tested_pairs", ()))
    med = {tuple(map(int, k)): float(v) for k, v in dict(data["all_known_pchembl"]).items()}
    benches = {"same_version": (test, random_neg), "same_version_matched": (test, matched), "external": (cross, ext_neg)}
    pch = {"same_version": [med[p] for p in test], "same_version_matched": [med[p] for p in test],
           "external": [meta[p]["max_pchembl"] for p in cross]}
    return data, aug, benches, pch


def score_all(s, dev):
    data, aug, benches, pch = build(s)
    nc, npr = int(data["num_compounds"]), int(data["num_proteins"])
    pairs = {b: pos + neg for b, (pos, neg) in benches.items()}
    out = {}
    for b in BENCH:
        out[f"{b}|pairs"] = np.asarray(pairs[b])
        out[f"{b}|npos"] = np.array(len(benches[b][0]))
        out[f"{b}|pchembl_pos"] = np.asarray(pch[b], float)
    ban = np.load(f"results/gift/baseline_models/drugban/run_{s}/drugban_dgl/scores.npz", allow_pickle=False)
    for b in BENCH:
        assert np.array_equal(ban[f"{b}|pairs"], out[f"{b}|pairs"]), f"bench mismatch {s} {b}"
        out[f"drugban|{b}"] = ban[f"{b}|score"]
    dti = np.load(f"{DTIAM.format(s=s)}/scores.npz", allow_pickle=False)
    for b in BENCH:
        assert np.array_equal(dti[f"{b}|pairs"], out[f"{b}|pairs"]), f"bench mismatch {s} {b}"
        out[f"dtiam|{b}"] = dti[f"{b}|score"]

    gg = clean_graph(data)[0]
    gaug = vs.load_external_chembl36_interactions_from_pkls(gg)[0]
    for name, ck in (("gift", f"results/gift/run_{s}/e2e/trained_model.pth"),
                     ("descriptor_only", f"results/gift/descriptor_only_seed_{s}/e2e/trained_model.pth")):
        model, _ = load_gift_model(data, dev, ck)
        for b in BENCH:
            out[f"{name}|{b}"] = vs.score_gnn_pairs(model, gaug if b == "external" else gg, pairs[b], dev)

    rf = joblib.load(f"results/gift/baseline_models/rf/rf_seed_{s}.joblib")
    rf.set_params(n_jobs=8)
    fs, fx = vs.build_pair_feature_fn(data), vs.build_pair_feature_fn(aug)
    for b in BENCH:
        f = fx if b == "external" else fs
        out[f"rf|{b}"] = rf.predict_proba(np.stack([f(c, p) for c, p in pairs[b]]))[:, 1]

    import train_gnn_baselines as gb
    mu = data["node_features"].mean(axis=0)
    sd = data["node_features"].std(axis=0)
    sd[sd == 0] = 1.0

    def tens(g):
        x = torch.tensor((np.asarray(g["node_features"]) - mu) / sd, dtype=torch.float32)
        nt = np.asarray(g["node_types"])
        return x, torch.tensor(np.asarray(g["edge_index"]), dtype=torch.long), np.nonzero(nt == 0)[0], np.nonzero(nt == 1)[0]

    gt = {"in": tens(gg), "ext": tens(gaug)}
    for kind in ("gcn", "sage", "gat"):
        ck = torch.load(f"results/gift/baseline_models/gnn/{kind}_seed_{s}.pth", map_location="cpu")
        m = gb.StdLinkGNN(kind, gt["in"][0].size(1))
        m.load_state_dict(ck["model_state_dict"])
        m.eval()
        for b in BENCH:
            x, ei, cn, pn = gt["ext" if b == "external" else "in"]
            with torch.no_grad():
                z = m.encode(x, ei)
                c = torch.tensor(cn[[p[0] for p in pairs[b]]])
                p = torch.tensor(pn[[p[1] for p in pairs[b]]])
                out[f"{kind}|{b}"] = torch.sigmoid((z[c] * z[p]).sum(-1)).numpy()

    out.update(score_dta(s, data, aug, pairs, dev))
    return out


def score_dta(s, data, aug, pairs, dev):
    import pandas as pd
    import train_dta_baselines as dt
    from torch_geometric.data import Batch
    nc, npr = int(data["num_compounds"]), int(data["num_proteins"])
    smi_csv = pd.read_csv("data/chembl_lung_cancer_all_compounds.csv").dropna(subset=["smiles"])
    smi_csv = smi_csv.drop_duplicates("compound_id").set_index("compound_id")["smiles"]
    ext_records = pickle.load(open(aug["external_interactions_pkl"], "rb"))["external_compounds"]
    smiles = [smi_csv[c] for c in list(data["compound_ids"])[:nc]] + [ext_records[c]["smiles"] for c in aug["external_compound_ids"]]
    seqs = json.load(open("data/protein_sequences.json"))
    P = torch.tensor(np.stack([dt.label_seq(seqs[p], dt.CHARPROTSET, dt.MAX_SEQ) for p in list(data["protein_ids"])[:npr]]))
    S = torch.tensor(np.stack([dt.label_seq(x, dt.CHARISOSMISET, dt.MAX_SMI) for x in smiles]))
    graphs = [dt.mol_graph(x) for x in smiles]
    out = {}
    for kind in ("deepdta", "graphdta"):
        ck = torch.load(f"results/gift/dta_leadprio/models/{kind}_seed_{s}.pth", map_location="cpu", weights_only=False)
        m = dt.build_model(kind, ck["config"])
        m.load_state_dict(ck["model_state_dict"])
        m.eval()
        for b in BENCH:
            res = []
            with torch.no_grad():
                for i in range(0, len(pairs[b]), 512):
                    pp = pairs[b][i:i + 512]
                    c = [x[0] for x in pp]
                    seq = P[torch.tensor([x[1] for x in pp])]
                    if kind == "deepdta":
                        o = m(S[torch.tensor(c)], seq)
                    else:
                        o = m(Batch.from_data_list([graphs[i] for i in c]), seq)
                    res.append(torch.sigmoid(o).numpy().ravel())
            out[f"{kind}|{b}"] = np.concatenate(res)
    return out


def saved_ref(s):
    ev = json.load(open(f"results/gift/run_{s}/eval_final.json"))["results_by_method"]["gift"]
    ex = json.load(open(f"results/gift/run_{s}/eval_external_matched.json"))["results"]["gift"]
    evd = json.load(open(f"results/gift/descriptor_only_seed_{s}/eval_final.json"))["results_by_method"]["gift"]
    exd = json.load(open(f"results/gift/descriptor_only_seed_{s}/eval_external_matched.json"))["results"]["gift"]
    rf = json.load(open(f"results/gift/baseline_models/rf/rf_seed_{s}.json"))["benchmarks"]
    gnn = json.load(open(f"results/gift/baseline_models/gnn/gnn_baselines_std_seed_{s}.json"))["results"]
    dta = json.load(open(f"results/gift/dta_leadprio/dta_metrics_seed_{s}.json"))["results"]
    ban = json.load(open(f"results/gift/baseline_models/drugban/run_{s}/dta_baselines_drugban.json"))["results"]["drugban"]
    dti = json.load(open(f"{DTIAM.format(s=s)}/dta_baselines_dtiam.json"))["results"]["dtiam"]
    ref = {}
    for b in BENCH:
        gk = "cross_version" if b == "external" else b
        ref[("gift", b)] = ex if b == "external" else ev[b]
        ref[("descriptor_only", b)] = exd if b == "external" else evd[b]
        ref[("rf", b)] = rf[b]
        for k in ("gcn", "sage", "gat"):
            ref[(k, b)] = gnn[k][gk]
        for k in ("deepdta", "graphdta"):
            ref[(k, b)] = dta[k][b]
        ref[("drugban", b)] = ban[b]
        ref[("dtiam", b)] = dti[b]
    return ref


def evaluate(sc, b, model, T, variant):
    pairs = sc[f"{b}|pairs"]
    npos = int(sc[f"{b}|npos"])
    pc = sc[f"{b}|pchembl_pos"]
    s = sc[f"{model}|{b}"]
    y = np.r_[np.ones(npos), np.zeros(len(pairs) - npos)]
    keep = np.ones(len(pairs), bool)
    weak = np.r_[(pc < T), np.zeros(len(pairs) - npos, bool)]
    if variant == "exclude":
        keep &= ~weak
    else:
        y[weak] = 0
    y, s2, pr = y[keep], s[keep], [tuple(x) for x in pairs[keep]]
    pt, nt = per_target_roc_auc(pr, y, s2)
    return {"roc_auc": float(roc_auc_score(y, s2)), "pr_auc": float(average_precision_score(y, s2)),
            "mcc": float(matthews_corrcoef(y, (s2 >= 0.5).astype(int))),
            "per_target_roc_auc": pt, "n_pos": int(y.sum()), "n_neg": int((1 - y).sum()),
            "prevalence": float(y.mean())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(42, 72)))
    ap.add_argument("--score_only", action="store_true")
    a = ap.parse_args()
    torch.set_num_threads(4)
    dev = torch.device("cpu")
    os.makedirs(CACHE, exist_ok=True)
    per, check = {}, {}
    for s in a.seeds:
        f = f"{CACHE}/scores_seed_{s}.npz"
        if not os.path.exists(f):
            np.savez_compressed(f, **score_all(s, dev))
            print("scored", s, flush=True)
        if a.score_only:
            continue
        sc = dict(np.load(f))
        if "dtiam|external" not in sc:  # caches written before DTIAM was added
            dti = np.load(f"{DTIAM.format(s=s)}/scores.npz", allow_pickle=False)
            for b in BENCH:
                assert np.array_equal(dti[f"{b}|pairs"], sc[f"{b}|pairs"]), f"bench mismatch {s} {b}"
                sc[f"dtiam|{b}"] = dti[f"{b}|score"]
            np.savez_compressed(f, **sc)
        ref = saved_ref(s)
        for (m, b), r in ref.items():
            e = evaluate(sc, b, m, 5.0, "exclude")
            check[f"{s}|{m}|{b}"] = max(abs(e["roc_auc"] - r["roc_auc"]), abs(e["pr_auc"] - r["auprc"]), abs(e["mcc"] - r["mcc"]))
        per[s] = {f"{m}|{b}|{T:g}|{v}": evaluate(sc, b, m, T, v)
                  for m in MODELS for b in BENCH for T in THR for v in ("exclude", "relabel")}
    if a.score_only:
        return
    agg = {"repro_max_abs_diff_T5": {m: max(v for k, v in check.items() if k.split("|")[1] == m) for m in MODELS}}
    for key in per[a.seeds[0]]:
        r = {}
        for k in ("roc_auc", "pr_auc", "mcc", "per_target_roc_auc", "prevalence"):
            v = np.array([per[s][key][k] for s in a.seeds])
            r[k] = [float(v.mean()), float(v.std(ddof=1)) if len(v) > 1 else 0.0]
        for k in ("n_pos", "n_neg"):
            v = np.array([per[s][key][k] for s in a.seeds])
            r[k] = [float(v.mean()), int(v.min()), int(v.max())]
        agg[key] = r
    # GIFT wins per run (PR-AUC and ROC-AUC) against each baseline, primary variant.
    wins = {}
    for b in BENCH:
        for T in THR:
            for m in MODELS[1:]:
                for k in ("pr_auc", "roc_auc", "mcc"):
                    wins[f"{m}|{b}|{T:g}|{k}"] = int(sum(per[s][f"gift|{b}|{T:g}|exclude"][k] > per[s][f"{m}|{b}|{T:g}|exclude"][k]
                                                         for s in a.seeds))
    agg["gift_wins_exclude"] = wins
    json.dump({"per_seed": per, "agg": agg, "repro_check": check}, open(OUT, "w"), indent=1)
    print(json.dumps(agg["repro_max_abs_diff_T5"], indent=1))
    for b in BENCH:
        for T in THR:
            for v in ("exclude", "relabel"):
                g = agg[f"gift|{b}|{T:g}|{v}"]
                print(f"{b:22s} T>={T:g} {v:8s} pos {g['n_pos'][0]:.0f} neg {g['n_neg'][0]:.0f} "
                      f"GIFT PR {g['pr_auc'][0]:.3f}±{g['pr_auc'][1]:.3f} ROC {g['roc_auc'][0]:.3f}±{g['roc_auc'][1]:.3f} MCC {g['mcc'][0]:.3f}±{g['mcc'][1]:.3f}")


if __name__ == "__main__":
    main()
