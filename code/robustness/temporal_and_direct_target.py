"""Points 10 (retrospective temporal evaluation) and 11 (direct-target subset) of the REVIEW response.

Inputs: per-pair scores of the frozen main model models and baselines (temporal_predict_pairs.py -> preds/seed_<s>.npz) and the
ChEMBL activity records (extract_chembl_records.py -> records_c36.pkl, records_c35_external.pkl).
Writes results/robustness/temporal_direct_target/results.json and pair tables (release36_only_pairs.csv, direct_target_flags_*.csv).
Usage (repo root): python code/robustness/temporal_and_direct_target.py
"""
import os as _os, sys as _sys  # release bootstrap: run from any directory as python code/robustness/<script>.py
_REPO = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", ".."))
_sys.path[:0] = [_os.path.join(_REPO, "code"), _os.path.join(_REPO, "code", "robustness")]
_os.chdir(_REPO)
import contextlib
import io
import json
import os
import pickle
import sys

import numpy as np
import pandas as pd
from scipy.stats import rankdata, wilcoxon
from sklearn.metrics import average_precision_score, matthews_corrcoef, precision_score, recall_score, roc_auc_score

sys.path.insert(0, os.getcwd())
OUT = "results/robustness/temporal_direct_target"
SEEDS = [int(x) for x in os.environ["PT_SEEDS"].split(",")] if os.environ.get("PT_SEEDS") else list(range(42, 72))
METHODS = ["gift", "desc_only", "rf", "gcn", "sage", "gat", "deepdta", "graphdta", "drugban", "dtiam"]
BASELINES = ["rf", "gcn", "sage", "gat", "deepdta", "graphdta", "drugban", "dtiam"]  # main-table baselines
KEYS = ["roc_auc", "auprc", "precision", "recall", "mcc"]
N_BOOT = int(os.environ.get("PT_NBOOT", 1000))
QUAL = "pchembl_value.notna() & (data_validity_comment.isna() | data_validity_comment == 'Manually validated')"


def metrics(y, s):
    pred = (s >= 0.5).astype(int)
    return {"roc_auc": float(roc_auc_score(y, s)), "auprc": float(average_precision_score(y, s)),
            "precision": float(precision_score(y, pred, zero_division=0)),
            "recall": float(recall_score(y, pred, zero_division=0)), "mcc": float(matthews_corrcoef(y, pred)),
            "n_pos": int(y.sum()), "n_neg": int((y == 0).sum())}


def fast(y, s):
    """Tie-aware ROC-AUC / AP (identical to scikit-learn) and thresholded metrics, for the bootstrap."""
    npos = y.sum()
    nneg = len(y) - npos
    r = rankdata(s)
    auc = (r[y == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg)
    o = np.argsort(-s, kind="stable")
    ss, yy = s[o], y[o]
    tp, fp = np.cumsum(yy), np.cumsum(1 - yy)
    last = np.r_[np.nonzero(np.diff(ss))[0], len(ss) - 1]
    tp, fp = tp[last], fp[last]
    rec = tp / npos
    ap = float(np.sum(np.diff(np.r_[0, rec]) * tp / (tp + fp)))
    pred = s >= 0.5
    TP, FP = float((pred & (y == 1)).sum()), float((pred & (y == 0)).sum())
    FN, TN = npos - TP, nneg - FP
    prec = TP / (TP + FP) if TP + FP else 0.0
    den = np.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
    mcc = (TP * TN - FP * FN) / den if den else 0.0
    return {"roc_auc": auc, "auprc": ap, "precision": prec, "recall": TP / npos, "mcc": mcc}


def summarize(per_seed):
    """per_seed: method -> list of metric dicts (30 runs) -> mean, sd, and paired comparison with GIFT."""
    out = {}
    for m, rows in per_seed.items():
        out[m] = {k: {"mean": float(np.mean([r[k] for r in rows])), "sd": float(np.std([r[k] for r in rows], ddof=1))}
                  for k in KEYS}
        out[m]["n_pos"], out[m]["n_neg"] = rows[0]["n_pos"], rows[0]["n_neg"]
    for k in KEYS:
        g = np.array([r[k] for r in per_seed["gift"]])
        for m in BASELINES:
            v = np.array([r[k] for r in per_seed[m]])
            p = float(wilcoxon(g, v).pvalue) if np.any(g != v) else 1.0
            out[m][k]["gift_better_runs"] = int((g > v).sum())
            out[m][k]["wilcoxon_p_gift_vs"] = p
        best_b = max(BASELINES, key=lambda m: out[m][k]["mean"])
        out.setdefault("_best", {})[k] = {"best_overall": max(METHODS, key=lambda m: out[m][k]["mean"]),
                                          "best_main_table_method": max(["gift"] + BASELINES,
                                                                        key=lambda m: out[m][k]["mean"]),
                                          "best_baseline": best_b,
                                          "gift_beats_best_baseline_runs": int(
                                              (g > np.array([r[k] for r in per_seed[best_b]])).sum())}
    return out


def evaluate(preds, bench, pos_mask_fn, neg_mask_fn=None):
    """Positives restricted by pos_mask_fn(seed, pairs, y); negatives all (or neg_mask_fn)."""
    per_seed = {m: [] for m in METHODS}
    for s in SEEDS:
        z = preds[s]
        y, pairs = z[f"{bench}|y"], z[f"{bench}|pairs"]
        keep = (y == 0) if neg_mask_fn is None else ((y == 0) & neg_mask_fn(s, pairs, y))
        keep |= (y == 1) & pos_mask_fn(s, pairs, y)
        for m in METHODS:
            per_seed[m].append(metrics(y[keep], z[f"{bench}|{m}"][keep]))
    return per_seed


def cluster_bootstrap(preds, bench, keep, groups, rng):
    """95% CI of the 30-run mean metric, resampling compounds (clusters) with replacement; same pairs every seed."""
    y = preds[SEEDS[0]][f"{bench}|y"][keep]
    g = groups[keep]
    ug = np.unique(g)
    idx_of = {u: np.nonzero(g == u)[0] for u in ug}
    sc = {m: np.stack([preds[s][f"{bench}|{m}"][keep] for s in SEEDS]) for m in METHODS}
    boots = {m: {k: [] for k in KEYS} for m in METHODS}
    diff = {m: {k: [] for k in KEYS} for m in BASELINES}
    for _ in range(N_BOOT):
        ix = np.concatenate([idx_of[u] for u in rng.choice(ug, size=len(ug), replace=True)])
        yb = y[ix]
        if yb.sum() == 0 or yb.sum() == len(yb):
            continue
        means = {}
        for m in METHODS:
            r = [fast(yb, sc[m][j][ix]) for j in range(len(SEEDS))]
            means[m] = {k: np.mean([q[k] for q in r]) for k in KEYS}
            for k in KEYS:
                boots[m][k].append(means[m][k])
        for m in BASELINES:
            for k in KEYS:
                diff[m][k].append(means["gift"][k] - means[m][k])
    ci = {m: {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for k, v in d.items()}
          for m, d in boots.items()}
    dci = {m: {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for k, v in d.items()}
           for m, d in diff.items()}
    return {"ci95_mean_metric": ci, "ci95_gift_minus_baseline": dci, "n_boot": len(boots["gift"]["mcc"]),
            "n_clusters": int(len(ug))}


def main():
    preds = {s: dict(np.load(f"{OUT}/preds/seed_{s}.npz")) for s in SEEDS}
    z0 = preds[SEEDS[0]]
    cid, pid, nc = z0["compound_ids"].astype(str), z0["protein_ids"].astype(str), int(z0["nc"])
    for s in SEEDS:  # the external benchmark (pairs, labels) is identical in every run
        assert np.array_equal(preds[s]["external|pairs"], z0["external|pairs"])
        assert np.array_equal(preds[s]["compound_ids"], z0["compound_ids"])
    res = {"seeds": SEEDS, "model_dir": "results/gift (manuscript numbers)",
           "reproduction_max_abs_metric_diff": max(v for s in SEEDS for b in json.loads(str(preds[s]["check"])).values()
                                                   for v in b.values())}

    # ------------------------------------------------------------------ point 10
    P, y = z0["external|pairs"], z0["external|y"]
    key = np.array([f"{cid[c]}|{pid[p]}" for c, p in P])
    ext_ids = set(cid[nc:])
    pos = set(key[y == 1])
    r36 = pd.read_pickle(f"{OUT}/records_c36.pkl")
    r36["key"] = r36.compound + "|" + r36.target
    e36 = r36[(r36.exact == 1) & r36.compound.isin(ext_ids)]
    q36 = e36.query(QUAL, engine="python")
    med = q36.groupby("key").pchembl_value.median()
    assert set(med[med >= 5].index) == pos, "external positives not reproduced from ChEMBL 36"
    q36_35 = q36[q36.doc_release <= 35]
    med35 = q36_35.groupby("key").pchembl_value.median()
    pos35 = set(med35[med35 >= 5].index)
    sets = {
        "manuscript_def_not_positive_with_le35_documents": pos - pos35,
        "c36_no_qualifying_pchembl_record_from_le35_documents": pos - set(q36_35.key),
        "c36_no_record_any_type_from_le35_documents": pos - set(e36[e36.doc_release <= 35].key),
    }
    c35_path = f"{OUT}/records_c35_external.pkl"
    if os.path.exists(c35_path):
        r35 = pd.read_pickle(c35_path)
        r35["key"] = r35.compound + "|" + r35.target
        q35 = r35.query(QUAL, engine="python")
        m35 = q35[q35.exact == 1].groupby("key").pchembl_value.median()
        sets.update({
            "c35db_no_record_any_type_exact_id": pos - set(r35[r35.exact == 1].key),
            "c35db_no_record_any_type_parent_salt_family": pos - set(r35.key),  # primary (reviewer's definition)
            "c35db_no_pchembl_record_parent_salt_family": pos - set(r35[r35.pchembl_value.notna()].key),
            "c35db_not_positive_median_exact_id": pos - set(m35[m35 >= 5].index),
        })
        res["c35_external_compounds_found"] = int(r35[r35.exact == 1].compound.nunique())
        res["c35_family_extra_molregnos"] = int(r35[r35.exact == 0].molregno.nunique())
    PRIMARY = ("c35db_no_record_any_type_parent_salt_family" if os.path.exists(c35_path)
               else "c36_no_record_any_type_from_le35_documents")
    res["point10"] = {
        "n_external_positives": len(pos), "n_external_negatives": int((y == 0).sum()),
        "pct_positive_with_le35_documents (manuscript 96.0%)": 100 * len(pos & pos35) / len(pos),
        "primary_definition": PRIMARY,
        "release36_only_sets": {k: {"n_pairs": len(v), "n_compounds": len({x.split('|')[0] for x in v}),
                                    "n_targets": len({x.split('|')[1] for x in v}),
                                    "pct_of_positives_supported_earlier": 100 * (1 - len(v) / len(pos))}
                                for k, v in sets.items()},
    }
    prim = sets[PRIMARY]
    ms = sets["manuscript_def_not_positive_with_le35_documents"]
    res["point10"]["overlap_primary_vs_manuscript_set"] = len(prim & ms)
    rows = []
    for k in sorted(pos):
        c, t = k.split("|")
        rows.append({"compound": c, "target": t, **{n: int(k in v) for n, v in sets.items()},
                     "c36_median_pchembl": float(med[k]), "c36_n_qualifying": int((q36.key == k).sum()),
                     "c36_first_doc_release_qualifying": float(q36[q36.key == k].doc_release.min())})
    pd.DataFrame(rows).to_csv(f"{OUT}/external_positive_release_flags.csv", index=False)

    # Leakage checks: external compounds are absent from every run's graph and have no edges.
    import pipeline_utils as vs
    from graph_cleaning import clean_graph
    leak = {"external_ids_in_graph": 0, "edges_touching_external_nodes": 0, "external_in_train_pairs": 0,
            "parent_or_salt_form_in_graph": 0}
    graph_ids = set(cid[:nc])
    fam36 = r36[r36.compound.isin(ext_ids)][["compound", "molregno"]].drop_duplicates()
    graph_mol = set(r36[r36.compound.isin(graph_ids) & (r36.exact == 1)].molregno)
    leak["parent_or_salt_form_in_graph"] = int(fam36[fam36.molregno.isin(graph_mol)].compound.nunique())
    for s in SEEDS:
        data = pickle.load(open(f"data/graphs/graph_seed{s}.pkl", "rb"))
        with contextlib.redirect_stdout(io.StringIO()):
            g = clean_graph(data)[0]
            aug, cross, _ = vs.load_external_chembl36_interactions_from_pkls(g)
            sp = vs.derive_chembl35_splits(data, min_training_pchembl=5.0)
        ids = set(map(str, list(data["compound_ids"])[: int(data["num_compounds"])]))
        leak["external_ids_in_graph"] += len(ids & ext_ids)
        ei = np.asarray(aug["edge_index"])
        ext_nodes = np.nonzero(np.asarray(aug["node_types"]) == 0)[0][nc:]
        assert len(ext_nodes) == len(ext_ids)
        leak["edges_touching_external_nodes"] += int(np.isin(ei, ext_nodes).any(0).sum())
        leak["external_in_train_pairs"] += sum(c >= nc for c, _ in sp["train_pairs"] + sp["val_pairs"])
    leak["note"] = "counts summed over the 30 run graphs (GNN graph with the external compounds added as nodes)"
    res["point10"]["leakage_checks"] = leak

    # Evaluation on the release-36-only interactions: positives = set; negatives = the existing matched negatives
    # of the same compounds (all 5:1 negatives drawn for those compounds in the external benchmark).
    comp = np.array([cid[c] for c, _ in P])
    ev10 = {}
    rng = np.random.default_rng(0)
    for name, S in [("primary", prim), ("manuscript_set", ms)]:
        comps = {x.split("|")[0] for x in S}
        in_set = np.isin(key, list(S))
        same_c = np.isin(comp, list(comps))
        per = evaluate(preds, "external", lambda s, pr, yy: in_set, lambda s, pr, yy: same_c)
        keep = ((y == 1) & in_set) | ((y == 0) & same_c)
        ev10[name] = {"definition": PRIMARY if name == "primary" else "manuscript_def_not_positive_with_le35_documents",
                      "negatives": "existing compound-matched external negatives of the same compounds",
                      "summary": summarize(per),
                      "bootstrap": cluster_bootstrap(preds, "external", keep, comp, rng)}
        # Reference: the same compounds' interactions already supported in ChEMBL 35 (same negatives).
        old = (y == 1) & same_c & ~in_set
        per_old = evaluate(preds, "external", lambda s, pr, yy: old, lambda s, pr, yy: same_c)
        ev10[name]["reference_same_compounds_earlier_interactions"] = summarize(per_old)
        # Robustness: release-36-only positives against all 15,646 external negatives.
        ev10[name]["all_external_negatives"] = summarize(evaluate(preds, "external", lambda s, pr, yy: in_set))
    ev10["full_external_benchmark"] = summarize(evaluate(preds, "external", lambda s, pr, yy: np.ones(len(yy), bool)))
    res["point10"]["evaluation"] = ev10

    # ------------------------------------------------------------------ point 11
    # Qualifying records = the records defining the label: unflagged pChEMBL records (median >= 5); ChEMBL 35 test
    # positives use only records from documents released in ChEMBL 35 or earlier (build_run_graphs.py).
    g36 = r36[r36.exact == 1].query(QUAL, engine="python")
    flags = {}
    for scope, df in [("chembl35", g36[g36.doc_release <= 35]), ("external", g36[g36.compound.isin(ext_ids)])]:
        d9b = set(df[(df.confidence_score == 9) & (df.assay_type == "B")].key)
        d9 = set(df[df.confidence_score == 9].key)
        flags[scope] = {"conf9_B": d9b, "conf9_any": d9, "recorded": set(df.key),
                        "assay_type_counts": df.assay_type.value_counts().to_dict(),
                        "confidence_counts": {int(k): int(v) for k, v in df.confidence_score.value_counts().items()}}
    p11 = {"definition": "positive pair whose qualifying (unflagged pChEMBL) records include >= 1 with "
                         "assays.confidence_score = 9 and assay_type = 'B' (strict) or confidence 9 any type (loose)"}
    sizes = {"same_version_test": [], "external": None}
    ev11 = {}
    for bench, scope in [("same_version", "chembl35"), ("same_version_matched", "chembl35"), ("external", "external")]:
        ev11[bench] = {}
        for sub in ["conf9_B", "conf9_any", "not_conf9_B"]:
            def mask(s, pr, yy, sub=sub, scope=scope):
                k = np.array([f"{cid[c]}|{pid[p]}" for c, p in pr])
                base = flags[scope]["conf9_B" if sub == "not_conf9_B" else sub]
                m = np.isin(k, list(base))
                return ~m if sub == "not_conf9_B" else m
            ev11[bench][sub] = summarize(evaluate(preds, bench, mask))
        ev11[bench]["all_positives"] = summarize(evaluate(preds, bench, lambda s, pr, yy: np.ones(len(yy), bool)))
    for s in SEEDS:
        z = preds[s]
        k = np.array([f"{cid[c]}|{pid[p]}" for c, p in z["same_version|pairs"][z["same_version|y"] == 1]])
        assert np.isin(k, list(flags["chembl35"]["recorded"])).all()
        sizes["same_version_test"].append({"n": len(k), "conf9_B": int(np.isin(k, list(flags["chembl35"]["conf9_B"])).sum()),
                                           "conf9_any": int(np.isin(k, list(flags["chembl35"]["conf9_any"])).sum())})
    kx = np.array(sorted(pos))
    sizes["external"] = {"n": len(kx), "conf9_B": int(np.isin(kx, list(flags["external"]["conf9_B"])).sum()),
                         "conf9_any": int(np.isin(kx, list(flags["external"]["conf9_any"])).sum()),
                         "n_compounds_conf9_B": len({x.split('|')[0] for x in kx if x in flags["external"]["conf9_B"]})}
    st = pd.DataFrame(sizes["same_version_test"])
    sizes["same_version_test_summary"] = {c: {"mean": float(st[c].mean()), "sd": float(st[c].std()),
                                              "min": int(st[c].min()), "max": int(st[c].max())} for c in st}
    sizes["same_version_test_frac_conf9_B"] = float((st.conf9_B / st.n).mean())
    sizes["same_version_test_frac_conf9_any"] = float((st.conf9_any / st.n).mean())
    p11["subset_sizes"] = sizes
    p11["record_composition"] = {s: {k: v for k, v in f.items() if k.endswith("counts")} for s, f in flags.items()}
    p11["evaluation"] = ev11
    res["point11"] = p11
    json.dump(res, open(f"{OUT}/results.json", "w"), indent=1, default=str)
    print("saved", f"{OUT}/results.json")


if __name__ == "__main__":
    main()
