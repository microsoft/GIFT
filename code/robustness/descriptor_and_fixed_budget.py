"""Shared helpers for the descriptor-only and fixed-budget analyses of the main model: driver-kinase lead metrics,
paired tests, and the fixed-budget / validation-threshold comparison (point5), run by descriptor_and_fixed_budget_main.py
and fixed_budget_main.py."""
import os as _os, sys as _sys  # release bootstrap: run from any directory as python code/robustness/<script>.py
_REPO = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", ".."))
_sys.path[:0] = [_os.path.join(_REPO, "code"), _os.path.join(_REPO, "code", "robustness")]
_os.chdir(_REPO)
import json
import os

import numpy as np
from scipy.stats import wilcoxon
from sklearn.metrics import roc_auc_score

HERE = "results/robustness/descriptor_fixed_budget"
REPO = os.path.abspath(_REPO)
SC = os.path.join(HERE, "scores")
SEEDS = list(range(42, 72))
DRIVERS = {"CHEMBL203": "EGFR", "CHEMBL1824": "ERBB2", "CHEMBL4247": "ALK", "CHEMBL3717": "MET",
           "CHEMBL2041": "RET", "CHEMBL5568": "ROS1", "CHEMBL5145": "BRAF",
           "CHEMBL2815": "NTRK1", "CHEMBL4898": "NTRK2", "CHEMBL5608": "NTRK3"}
KS = (50, 100, 200, 500, 1000)
MODELS = ("gift", "descriptor_only", "rf", "gcn", "sage", "gat", "deepdta", "graphdta", "drugban", "dtiam")
BENCHES = ("external", "same_version_matched")


def wil(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if np.all(a == b):
        return 1.0
    return float(wilcoxon(a, b).pvalue)


def msd(x):
    x = np.asarray(x, float)
    return {"mean": float(x.mean()), "sd": float(x.std(ddof=1)), "sd_ddof0": float(x.std())}


# ---------------------------------------------------------------- point 4
def lead_metrics(S, labels, is_kinase, pids):
    n_ext = labels.shape[1]
    n_act = labels.sum(1)
    drivers = [p for p in range(len(pids)) if pids[p] in DRIVERS and 5 <= n_act[p] < n_ext]
    fam, glob = S[is_kinase].mean(0), S.mean(0)
    out = {}
    for name, f in (("per_target", lambda p: S[p]), ("family", lambda p: fam), ("global", lambda p: glob)):
        per = {}
        for p in drivers:
            s, y = f(p), labels[p]
            o = np.argsort(-s, kind="stable")
            per[DRIVERS[pids[p]]] = {"roc_auc": float(roc_auc_score(y, s)), "top10": int(y[o[:10]].sum()),
                                     "at1": int(y[o[0]])}
        out[name] = {"per_driver": per, "n_drivers": len(per),
                     "roc_auc_mean": float(np.mean([v["roc_auc"] for v in per.values()])),
                     "top10_sum": int(sum(v["top10"] for v in per.values())),
                     "top10_mean": float(np.mean([v["top10"] for v in per.values()])),
                     "n_at1": int(sum(v["at1"] for v in per.values()))}
    return out, drivers



def hits_at_k(y, s, k):
    o = np.sort(s)[::-1]
    t = o[k - 1]
    above, tied = s > t, s == t
    return float(y[above].sum() + (k - above.sum()) * y[tied].sum() / tied.sum())


def best_threshold(y, s, crit):
    o = np.argsort(-s, kind="stable")
    ss, yy = s[o], y[o]
    last = np.r_[ss[1:] != ss[:-1], True]           # predict positive for s >= ss[i]
    tp = np.cumsum(yy)[last].astype(float)
    fp = np.cumsum(1 - yy)[last].astype(float)
    th = ss[last]
    P, N = y.sum(), len(y) - y.sum()
    fn, tn = P - tp, N - fp
    if crit == "mcc":
        den = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        v = np.where(den > 0, (tp * tn - fp * fn) / np.where(den > 0, den, 1), 0)
    else:
        v = 2 * tp / (2 * tp + fp + fn)
    i = int(np.argmax(v))
    return float(th[i]), float(v[i])


def at_threshold(y, s, t):
    pred = s >= t
    tp = int((pred & (y == 1)).sum())
    fp = int((pred & (y == 0)).sum())
    fn = int(((~pred) & (y == 1)).sum())
    tn = int(len(y) - tp - fp - fn)
    den = np.sqrt(float(tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return {"n_selected": tp + fp, "tp": tp, "precision": tp / (tp + fp) if tp + fp else 0.0,
            "recall": tp / (tp + fn), "f1": 2 * tp / (2 * tp + fp + fn), "mcc": (tp * tn - fp * fn) / den if den else 0.0}


def load_scores(s):
    g = np.load(f"{SC}/run_seed_{s}.npz")
    d = np.load(f"{SC}/descriptor_only_seed_{s}.npz")
    b = np.load(f"{SC}/baselines_seed_{s}.npz")
    ban = np.load(f"{REPO}/results/gift/baseline_models/drugban/run_{s}/drugban_dgl/scores.npz")
    banv = np.load(f"{SC}/drugban_seed_{s}.npz")
    sc, Y = {}, {}
    for bench in ("val",) + BENCHES:
        P = g[f"{bench}|pairs"]
        Y[bench] = g[f"{bench}|y"]
        for z in (d, b):
            assert np.array_equal(z[f"{bench}|pairs"], P) and np.array_equal(z[f"{bench}|y"], Y[bench])
        sc[("gift", bench)] = g[f"{bench}|score"]
        sc[("descriptor_only", bench)] = d[f"{bench}|score"]
        for m in ("rf", "gcn", "sage", "gat", "deepdta", "graphdta"):
            sc[(m, bench)] = b[f"{m}|{bench}|score"]
        z, key = (banv, "drugban|val|score") if bench == "val" else (ban, f"{bench}|score")
        assert np.array_equal(z[f"{bench}|pairs"], P) and np.array_equal(z[f"{bench}|y"], Y[bench])
        sc[("drugban", bench)] = z[key].astype(np.float64)
        # DTIAM: benchmark scores from its run, validation scores from score_dtiam_val.py (saved models, inference only).
        z, key = ((np.load(f"{SC}/dtiam_seed_{s}.npz"), "dtiam|val|score") if bench == "val" else
                  (np.load(f"{REPO}/results/robustness/dtiam/runs/run_{s}/scores.npz"), f"{bench}|score"))
        assert np.array_equal(z[f"{bench}|pairs"], P) and np.array_equal(z[f"{bench}|y"], Y[bench])
        sc[("dtiam", bench)] = z[key].astype(np.float64)
    return sc, Y


def point5():
    per = {m: {} for m in MODELS}
    for s in SEEDS:
        sc, Y = load_scores(s)
        for m in MODELS:
            r = {"val_roc_auc": float(roc_auc_score(Y["val"], sc[(m, "val")]))}
            th = {c: best_threshold(Y["val"], sc[(m, "val")], c) for c in ("mcc", "f1")}
            for bench in BENCHES:
                y, x = Y[bench], sc[(m, bench)]
                r[bench] = {"hits_at_k": {k: hits_at_k(y, x, k) for k in KS},
                            "n_pos": int(y.sum()), "n": int(len(y)), "roc_auc": float(roc_auc_score(y, x)),
                            "at_0.5": at_threshold(y, x, 0.5)}
                for c in ("mcc", "f1"):
                    r[bench][f"val_{c}_threshold"] = {"threshold": th[c][0], "val_value": th[c][1],
                                                      **at_threshold(y, x, th[c][0])}
            per[m][s] = r
    agg, tests = {}, {}
    for bench in BENCHES:
        agg[bench], tests[bench] = {}, {}
        for m in MODELS:
            a = {f"hits@{k}": msd([per[m][s][bench]["hits_at_k"][k] for s in SEEDS]) for k in KS}
            for c in ("mcc", "f1"):
                for q in ("threshold", "n_selected", "tp", "precision", "recall", "mcc", "f1"):
                    a[f"val{c}_{q}"] = msd([per[m][s][bench][f"val_{c}_threshold"][q] for s in SEEDS])
            for q in ("n_selected", "tp", "precision", "mcc"):
                a[f"t0.5_{q}"] = msd([per[m][s][bench]["at_0.5"][q] for s in SEEDS])
            a["roc_auc"] = msd([per[m][s][bench]["roc_auc"] for s in SEEDS])
            agg[bench][m] = a
            if m == "gift":
                continue
            t = {f"hits@{k}": wil([per["gift"][s][bench]["hits_at_k"][k] for s in SEEDS],
                                  [per[m][s][bench]["hits_at_k"][k] for s in SEEDS]) for k in KS}
            for q in ("precision", "mcc", "tp"):
                t[f"valmcc_{q}"] = wil([per["gift"][s][bench]["val_mcc_threshold"][q] for s in SEEDS],
                                       [per[m][s][bench]["val_mcc_threshold"][q] for s in SEEDS])
            tests[bench][m] = t
    return {"per_seed": per, "aggregate": agg, "wilcoxon_gift_vs": tests}
