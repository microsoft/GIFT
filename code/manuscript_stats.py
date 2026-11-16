"""All numbers of the manuscript (protocol with Phase 3 compound and protein masking at 30%;
30 runs, one random split per run), printed as text. Soft-edge ablation sections run once all ablation runs exist."""

import contextlib
import io
import json
import os
import pickle
import re
import sqlite3
from collections import Counter

import numpy as np
from scipy.stats import spearmanr

SEEDS = list(range(42, 72))
R = "results/gift"
S = json.load(open(f"{R}/summary_multiseed.json"))
L = json.load(open(f"{R}/lead_multiseed.json"))
ABL = os.path.exists(f"{R}/lead_multiseed_ablation.json")
LA = json.load(open(f"{R}/lead_multiseed_ablation.json")) if ABL else None
METHODS = ["gift", "random_forest", "gcn", "sage", "gat"]
NAMES = {"roc_auc": "ROC-AUC", "auprc": "PR-AUC", "per_target_roc_auc": "Per-target ROC-AUC", "accuracy": "Accuracy",
         "precision": "Precision", "recall": "Recall", "f1": "F1", "mcc": "MCC"}


def ms(v, d=3):
    return f"{np.mean(v):.{d}f}$\\pm${np.std(v, ddof=1):.{d}f}"


def rng(v, d=3):
    return f"{min(v):.{d}f}--{max(v):.{d}f}"


def section(t):
    print(f"\n######## {t}")


section("BENCHMARK TABLES (LaTeX rows)")
for bench, mets in [("same_version", list(NAMES)), ("same_version_matched", list(NAMES)),
                    ("external", [m for m in NAMES if m != "accuracy"])]:
    print(f"%% {bench}")
    for m in mets:
        v = [S["main"][f"{bench}|{k}"][m] for k in METHODS]
        best = max(range(5), key=lambda i: v[i]["mean"])
        cells = [(("\\textbf{%s}" if i == best else "%s") % f"{x['mean']:.3f}$\\pm${x['sd']:.3f}") for i, x in enumerate(v)]
        print(NAMES[m] + " & " + " & ".join(cells) + " \\\\")
for b in ["same_version_isolated"]:
    x = S["main"][f"{b}|gift"]
    print(b, {m: f"{x[m]['mean']:.3f}±{x[m]['sd']:.3f}" for m in ["roc_auc", "auprc", "f1", "mcc"]})
for bench in ["same_version", "same_version_matched", "external"]:
    for m in ["roc_auc", "mcc", "precision", "recall", "per_target_roc_auc"]:
        base = [S["main"][f"{bench}|{k}"][m]["mean"] for k in METHODS[2:]]
        print(f"{bench} {m}: GNN range {min(base):.3f}-{max(base):.3f}; all baselines {min([S['main'][f'{bench}|{k}'][m]['mean'] for k in METHODS[1:]]):.3f}-{max([S['main'][f'{bench}|{k}'][m]['mean'] for k in METHODS[1:]]):.3f}")
# margin consistency: proposed > best baseline in every run?
for bench in ["same_version", "same_version_matched", "external"]:
    for m in ["roc_auc", "mcc", "per_target_roc_auc"]:
        p = np.array(S["main"][f"{bench}|gift"][m]["values"])
        best = np.max([S["main"][f"{bench}|{k}"][m]["values"] for k in METHODS[1:]], axis=0)
        print(f"{bench} {m}: proposed > every baseline in {(p > best).sum()}/30 runs")

section("ABLATION (with vs without soft edges)")
def picks(run):
    tot, first = [], []
    for s in SEEDS:
        d = json.load(open(f"{R}/{run.format(s=s)}/lead_prioritization.json"))
        a = f = 0
        for v in d["top_candidates"].values():
            for i, r in enumerate(v["rows"][:5]):
                w = r.get("chembl_web") or {}
                act = (r.get("chembl36_pchembl") or 0) >= 5 or (w.get("max_pchembl") or 0) >= 5
                a += act
                f += (i == 0 and act)
        tot.append(a)
        first.append(f)
    return tot, first


def val_roc(run):
    conn, iso, info = [], [], []
    for s in SEEDS:
        d = f"{R}/{run.format(s=s)}"
        h, e = open(f"{d}/head.log").read(), open(f"{d}/e2e.log").read()
        hb = re.findall(r"Epoch +(\d+):.*val_auc\(conn,iso\)=\(([\d.]+),([\d.]+)\).*\*$", h, re.M)
        eb = re.findall(r"Step +(\d+):.*val_auc\(conn,iso\)=\(([\d.]+),([\d.]+)\).*\*$", e, re.M)
        sel = ("e2e", int(eb[-1][0]), float(eb[-1][1]), float(eb[-1][2])) if eb else ("head", int(hb[-1][0]), float(hb[-1][1]), float(hb[-1][2]))
        conn.append(sel[2])
        iso.append(sel[3])
        info.append((int(hb[-1][0]), sel[0], sel[1]))
    return conn, iso, info


pt_m, pf_m = picks("run_{s}")
vc_m, vi_m, info_m = val_roc("run_{s}")
if ABL:
    abl_rows = []
    for b, m in [("same_version", "roc_auc"), ("same_version", "auprc"), ("same_version", "f1"), ("same_version", "mcc"),
                 ("same_version_isolated", "roc_auc"), ("same_version_isolated", "mcc"),
                 ("same_version_matched", "roc_auc"), ("same_version_matched", "auprc"), ("same_version_matched", "per_target_roc_auc"),
                 ("same_version_matched", "f1"), ("same_version_matched", "mcc"),
                 ("external", "roc_auc"), ("external", "per_target_roc_auc"), ("external", "mcc")]:
        a = S["main"][f"{b}|gift"][m]["values"]
        c = S["ablation"][f"{b}|gift"][m]["values"]
        abl_rows.append((f"{b}.{m}", a, c, 3))
    for k, d in [("auc_mean", 3), ("hits_mean", 2), ("above_family", 0), ("kin_auc", 3), ("kin_hits", 2), ("drv_auc", 3),
                 ("drv_hits", 1), ("tested_auc", 2), ("sel_auc", 2)]:
        abl_rows.append((f"lead.{k}", L["agg"][k]["values"], LA["agg"][k]["values"], d))


    pt_a, pf_a = picks("ablation_no_softedge_seed_{s}")
    abl_rows += [("lead.top5_actives", pt_m, pt_a, 1), ("lead.first_active", pf_m, pf_a, 1)]


    vc_a, vi_a, info_a = val_roc("ablation_no_softedge_seed_{s}")
    abl_rows = [("val.conn", vc_m, vc_a, 3), ("val.iso", vi_m, vi_a, 3)] + abl_rows
    higher = 0
    for name, a, c, d in abl_rows:
        ma, mc = np.mean(a), np.mean(c)
        win = "WITH" if round(ma, d) > round(mc, d) else ("WITHOUT" if round(mc, d) > round(ma, d) else "tie")
        higher += win == "WITH"
        print(f"{name:40s} with {ms(a, d)}  without {ms(c, d)}  higher={win}  runs with>without {sum(np.array(a) > np.array(c))}/30")
    print(f"with soft edges higher mean (at reported precision) in {higher} of {len(abl_rows) - 2} metrics (excluding 2 validation rows: "
          f"{sum(1 for n, a, c, d in abl_rows[2:] if round(np.mean(a), d) > round(np.mean(c), d))})")
else:
    print("ablation runs incomplete: skipped")

section("PHASE 3 SELECTION")
for nm, info in [("main", info_m)] + ([("ablation", info_a)] if ABL else []):
    ep = [i[0] for i in info]
    steps = [i[2] for i in info if i[1] == "e2e"]
    print(f"{nm}: head best epochs {min(ep)}-{max(ep)}; e2e selected in {len(steps)}/30, steps {min(steps)}-{max(steps)}; "
          f"head kept in {sum(i[1] == 'head' for i in info)} runs")

section("LEAD PRIORITIZATION (30 runs)")
A = L["agg"]
for k in A:
    print(f"{k:22s} {A[k]['mean']:.4g} ± {A[k]['sd']:.3g}")
ps = L["per_seed"]
pf = [ps[str(s)]["p_family"] for s in SEEDS]
pk = [ps[str(s)]["p_family_kin"] for s in SEEDS]
print(f"p_family<0.05 in {sum(p < 0.05 for p in pf)}/30, <0.01 in {sum(p < 0.01 for p in pf)}/30; "
      f"p_family_kin<0.05 in {sum(p < 0.05 for p in pk)}/30, <0.01 in {sum(p < 0.01 for p in pk)}/30")
print("drivers above random per run:", Counter(int(ps[str(s)]["drv_above_random"]) for s in SEEDS))
drv = {}
for s in SEEDS:
    for g, v in ps[str(s)]["drv_per_target"].items():
        drv.setdefault(g, {"auc": [], "hits": [], "n": v["n"]})
        drv[g]["auc"].append(v["auc"])
        drv[g]["hits"].append(v["hits"])
below = Counter()
for s in SEEDS:
    for g, v in ps[str(s)]["drv_per_target"].items():
        if v["hits"] <= 10 * v["n"] / 95:
            below[g] += 1
print("runs in which a driver is NOT above random:", dict(below))
for g in sorted(drv, key=lambda g: -drv[g]["n"]):
    v = drv[g]
    print(f"{g:6s} & {v['n']} & {ms(v['auc'], 2)} & {ms(v['hits'], 1)} & {10 * v['n'] / 95:.1f}")
for k, lab in [("drv_auc", "per target auc"), ("drv_hits", "per target hits"), ("drv_family_auc", "family auc"),
               ("drv_family_hits", "family hits"), ("drv_global_auc", "global auc"), ("drv_global_hits", "global hits")]:
    print(lab, ms(A[k]["values"], 2 if "auc" in k else 1))
braf = [ps[str(s)]["tested_per_target"].get("BRAF") for s in SEEDS]
print("BRAF tested-only roc range", rng([b for b in braf if b is not None], 2), "runs <0.5:", sum(b < 0.5 for b in braf if b is not None))
print(f"top5 actives {ms(pt_m, 1)}, first-ranked actives {ms(pf_m, 1)}" + (f" (ablation {ms(pt_a, 1)}, {ms(pf_a, 1)})" if ABL else ""))
d42 = json.load(open(f"{R}/run_42/lead_prioritization.json"))
cat, per = Counter(), {}
exp = 0
for t, v in d42["top_candidates"].items():
    exp += 5 * d42["per_target"][t]["gnn"]["n_actives"] / 95
    for i, r in enumerate(v["rows"][:5]):
        w = r.get("chembl_web") or {}
        c = ("known" if (r.get("chembl36_pchembl") or 0) >= 5 else "parent" if (w.get("max_pchembl") or 0) >= 5
             else "tested" if w.get("n_records", 0) > 0 else "none")
        cat[c] += 1
        p = per.setdefault(r["parent_name"], [0, 0])
        p[0] += 1
        p[1] += c in ("known", "parent")
print("seed42 top5 categories", dict(cat), f"expected {exp:.1f}", "per drug", per)
for t, v in d42["top_candidates"].items():
    print("  ", v["gene"], [(r["parent_name"][:12], r.get("chembl36_pchembl"), (r.get("chembl_web") or {}).get("max_pchembl"),
                             (r.get("chembl_web") or {}).get("n_records")) for r in v["rows"][:5]])

section("SOFT-EDGE PROPOSALS (30 runs)")
held, auc_all, auc_t, nt, fc, fp, rho_p, share_c, share_p = [], [], [], [], [], [], [], [], []
for s in SEEDS:
    sp = json.load(open(f"{R}/run_{s}/soft_edge_proposals.json"))
    held.append(sp["global"]["held_out_in_top_k"])
    auc_all.append(sp["global"]["roc_auc_all_candidates"])
    auc_t.append(sp["per_target_summary"]["mean_roc_auc"])
    nt.append(sp["per_target_summary"]["n_targets"])
    g = pickle.load(open(f"data/graphs/graph_seed{s}.pkl", "rb"))
    tr = {tuple(map(int, p)) for p in g["split_override"]["train"]}
    Sm = np.load(f"{R}/run_{s}/soft_edge_proposals_scores.npy")
    C, P = Sm.shape
    cdeg, pdeg = np.zeros(C, int), np.zeros(P, int)
    for c, p in tr:
        cdeg[c] += 1
        pdeg[p] += 1
    flat = Sm.ravel().copy()
    flat[[c * P + p for c, p in tr]] = -np.inf
    top = np.argpartition(-flat, 200000)[:200000]
    fc.append((cdeg[top // P] == 0).mean())
    fp.append((pdeg[top % P] == 0).mean())
    share_c.append((cdeg == 0).mean())
    share_p.append((pdeg == 0).mean())
    rho_p.append(spearmanr(Sm.mean(0), pdeg)[0])
fp, rho_p = np.array(fp), np.array(rho_p)
print(f"held-out in top200k {min(held)}-{max(held)} (median {np.median(held):.0f}); expected ~{200000 * np.mean([845]) / 2166549:.0f}")
print(f"roc all {ms(auc_all, 2)}; per-target {ms(auc_t, 2)} over {min(nt)}-{max(nt)} targets")
print(f"compound deg0 fraction in proposals {rng(fc * 100 if isinstance(fc, np.ndarray) else np.array(fc) * 100, 1)}% (share of compounds without training interactions {np.mean(share_c) * 100:.1f}%)")
pref = fp > 0.9
print(f"protein deg0 fraction >0.9 in {pref.sum()}/30 runs ({rng(fp[pref] * 100, 1)}%), others {rng(fp[~pref] * 100, 1) if (~pref).any() else '-'}%; "
      f"share of proteins w/o training {np.mean(share_p) * 100:.1f}%; rho in pref runs {rng(rho_p[pref], 2)}, others {rng(rho_p[~pref], 2) if (~pref).any() else '-'}")

section("SPLITS (30 runs)")
ntr, nva, nte, ntc, ntc_tr, nheld = [], [], [], [], [], []
for s in SEEDS:
    g = pickle.load(open(f"data/graphs/graph_seed{s}.pkl", "rb"))
    so = g["split_override"]
    trc = {c for c, _ in so["train"]}
    ntr.append(len(so["train"]))
    nva.append(len(so["val"]))
    nte.append(len(so["test"]))
    tc = {c for c, _ in so["test"]}
    ntc.append(len(tc))
    ntc_tr.append(len(tc & trc) / len(tc))
    nheld.append(len(so["val"]) + len(so["test"]))
print(f"train {np.mean(ntr):.0f} ({min(ntr)}-{max(ntr)}), val {np.mean(nva):.0f} ({min(nva)}-{max(nva)}), test {np.mean(nte):.0f} ({min(nte)}-{max(nte)}), "
      f"held-out {min(nheld)}-{max(nheld)}; test compounds {min(ntc)}-{max(ntc)}, with training interactions {min(ntc_tr) * 100:.0f}-{max(ntc_tr) * 100:.0f}%")
ev = [json.load(open(f"{R}/run_{s}/eval_external_matched.json")) for s in SEEDS]
print("external positives/negatives/compounds:", {(e["n_positives"], e["n_negatives"], e["n_compounds"]) for e in ev})
mis = []
for s in SEEDS:
    e = json.load(open(f"{R}/run_{s}/eval_final.json"))
    mis.append(e)

section("EXTERNAL SET RELEASE (median labels)")
ext = pickle.load(open("data/external_compounds_labels.pkl", "rb"))["external_compounds"]
con = sqlite3.connect(f"file:{os.environ['CHEMBL36_DB']}?mode=ro", uri=True)  # needs CHEMBL36_DB
con.execute("CREATE TEMP TABLE c (id TEXT PRIMARY KEY)")
con.executemany("INSERT INTO temp.c VALUES (?)", [(c,) for c in ext])
rows = con.execute("""SELECT md.chembl_id, td.chembl_id, act.pchembl_value, CAST(SUBSTR(r.chembl_release, 8) AS INTEGER)
    FROM activities act JOIN molecule_dictionary md ON act.molregno = md.molregno JOIN assays a ON act.assay_id = a.assay_id
    JOIN target_dictionary td ON a.tid = td.tid JOIN docs d ON act.doc_id = d.doc_id
    JOIN chembl_release r ON d.chembl_release_id = r.chembl_release_id
    WHERE md.chembl_id IN (SELECT id FROM temp.c) AND act.pchembl_value IS NOT NULL
    AND (act.data_validity_comment IS NULL OR act.data_validity_comment = 'Manually validated')""").fetchall()
chembl35_values = {}
for c, t, v, rel in rows:
    if rel <= 35:
        chembl35_values.setdefault((c, t), []).append(v)
pos = [(c, t) for c, r in ext.items() for t, _ in r["interactions"]]
already = [p for p in pos if p in chembl35_values and np.median(chembl35_values[p]) >= 5]
new = [p for p in pos if p not in already]
print(f"positives {len(pos)}; potent already with ChEMBL<=35 records {len(already)} ({100 * len(already) / len(pos):.1f}%); "
      f"new in 36 {len(new)} pairs on {len({c for c, _ in new})} compounds")
npos = Counter(c for c, _ in pos)
print(f"active targets per compound {min(npos.values())}-{max(npos.values())} (median {np.median(list(npos.values())):.0f})")
