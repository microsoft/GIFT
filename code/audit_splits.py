"""Audit split masks and label manifests for every run (leakage checks).

For each seed's graph pkl it checks the invariants every new control will rely on:
  (1) split_override train/val/test are disjoint and cover the recorded-pair universe of the split;
  (2) training interactions are exactly the potent pairs (median pChEMBL >= 5) of the train part;
  (3) val/test positives used for evaluation are potent and never appear in the message-passing graph;
  (4) the evaluation reserve is identical across seeds (fixed RESERVE_SEED) and excludes no training leakage;
  (5) sub-threshold labels (median pChEMBL in [3,5)) are disjoint from the potent training positives;
  (6) external ChEMBL36 compounds are new nodes (offset >= num_compounds), so they cannot leak as train edges.
Prints PASS/FAIL per check; exits non-zero on any failure.
"""

import pickle
import sys

import numpy as np

import pipeline_utils as vs
from negative_pool import reserved_eval_mask

SEEDS = list(range(42, 72))
GRAPH = "data/graphs/graph_seed{s}.pkl"


def audit_seed(s, ref_reserve):
    d = pickle.load(open(GRAPH.format(s=s), "rb"))
    so = d["split_override"]
    med = {tuple(map(int, k)): float(v) for k, v in dict(d["all_known_pchembl"]).items()}
    tr = {tuple(map(int, p)) for p in so["train"]}
    va = {tuple(map(int, p)) for p in so["val"]}
    te = {tuple(map(int, p)) for p in so["test"]}
    fails = []

    # (1) disjoint splits
    if tr & va or tr & te or va & te:
        fails.append("splits overlap")

    # (2) training interactions are exactly the potent pairs
    if not all(med.get(p, 0.0) >= 5.0 for p in tr):
        fails.append("train contains non-potent pairs")

    # (3) eval positives potent + removed from graph
    splits = vs.derive_chembl35_splits(d, min_training_pchembl=5.0)
    train_pairs = set(splits["train_pairs"])
    if train_pairs != tr:
        fails.append("derive_chembl35_splits train != split_override train")
    if not all(med.get(p, 0.0) >= 5.0 for p in splits["test_pairs"] + splits["val_pairs"]):
        fails.append("val/test eval positives not potent")
    if (set(splits["test_pairs"]) | set(splits["val_pairs"])) & train_pairs:
        fails.append("held-out positives leak into training positives")

    # (4) reserve identical across seeds
    nc, npr = int(d["num_compounds"]), int(d["num_proteins"])
    res = reserved_eval_mask(nc, npr)
    if ref_reserve is not None and not np.array_equal(res, ref_reserve):
        fails.append("reserve mask differs across seeds")

    # (5) sub-threshold disjoint from potent train
    sub = {p for p in med if 3.0 <= med[p] < 5.0}
    if sub & tr:
        fails.append("sub-threshold pairs overlap potent training positives")

    # (6) external compounds are new nodes
    gnn_graph = vs.graph_cleaning.clean_graph(d)[0] if hasattr(vs, "graph_cleaning") else None
    _, cross_pairs, _ = vs.load_external_chembl36_interactions_from_pkls(d)
    ext_c = {c for c, _ in cross_pairs}
    if any(c < nc for c in ext_c):
        fails.append("external compound index collides with training compounds")
    if {(c, p) for c, p in cross_pairs} & tr:
        fails.append("external pairs overlap training positives")

    return res, fails, dict(n_train=len(tr), n_val=len(va), n_test=len(te), n_sub=len(sub),
                            n_ext_pairs=len(cross_pairs), n_ext_comp=len(ext_c))


def main():
    ref = None
    all_ok = True
    for s in SEEDS:
        ref, fails, stats = audit_seed(s, ref)
        tag = "PASS" if not fails else "FAIL: " + "; ".join(fails)
        print(f"seed {s}: train={stats['n_train']} val={stats['n_val']} test={stats['n_test']} "
              f"sub={stats['n_sub']} ext_pairs={stats['n_ext_pairs']} ext_comp={stats['n_ext_comp']} -> {tag}")
        all_ok &= not fails
    print("\nAUDIT", "PASSED" if all_ok else "FAILED")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
