"""
Build data/graphs/graph_run.pkl: the base graph relabeled with the standard activity
definition applied to the median pChEMBL of each pair (positive = median pChEMBL >= 5).

- pChEMBL records come from the ChEMBL 36 SQLite release restricted to documents first released
  in ChEMBL 35 or earlier, i.e. the ChEMBL 35 records, with flagged values
  (data_validity_comment other than 'Manually validated') removed.
- The pair universe (14,453 recorded ChEMBL 35 pairs) and its random 70/15/15 split
  (random_state=42) are unchanged; each part is restricted to its potent pairs.
- CP edge attributes of the kept pairs are recomputed from the median: (pChEMBL/12, 1.0 if > 6 else 0.5).
- --split_seeds S1,S2,..: one graph per seed, random split with random_state=S (S=42 reproduces the default).
  With --cold_target the split is protein-disjoint instead: the proteins with a potent pair are split
  70/15/15 (random_state=S) and every potent pair goes to the part of its protein.
"""

import argparse
import os
import pickle
import sqlite3
from collections import defaultdict

import numpy as np

from graph_cleaning import remove_cp_pairs, trainer_split

IN_PATH = "data/graphs/graph_base.pkl"
OUT_PATH = "data/graphs/graph_run.pkl"
THRESHOLD = 5.0
MAX_RELEASE = 35

QUERY = """
SELECT md.chembl_id, td.chembl_id, act.pchembl_value
FROM activities act
JOIN molecule_dictionary md ON act.molregno = md.molregno
JOIN assays a ON act.assay_id = a.assay_id
JOIN target_dictionary td ON a.tid = td.tid
JOIN docs d ON act.doc_id = d.doc_id
JOIN chembl_release r ON d.chembl_release_id = r.chembl_release_id
WHERE act.pchembl_value IS NOT NULL
  AND (act.data_validity_comment IS NULL OR act.data_validity_comment = 'Manually validated')
  AND CAST(SUBSTR(r.chembl_release, 8) AS INTEGER) <= ?
  AND md.chembl_id IN (SELECT id FROM temp.cids)
  AND td.chembl_id IN (SELECT id FROM temp.tids)
"""


def cold_target_split(potent, seed, test_ratio=0.15, val_ratio=0.15):
    """Protein-disjoint split of the potent pairs: no validation or test protein has a training interaction."""
    from sklearn.model_selection import train_test_split
    prots = sorted({p for _, p in potent})
    trv, te = train_test_split(prots, test_size=test_ratio, random_state=seed)
    tr, va = train_test_split(trv, test_size=val_ratio / (1 - test_ratio), random_state=seed)
    part = {**{p: 0 for p in tr}, **{p: 1 for p in va}, **{p: 2 for p in te}}
    out = ([], [], [])
    for pair in sorted(potent):
        out[part[pair[1]]].append(pair)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chembl_db", default=os.environ.get("CHEMBL36_DB"), required="CHEMBL36_DB" not in os.environ,
                    help="ChEMBL 36 SQLite file (default: $CHEMBL36_DB)")
    ap.add_argument("--split_seeds", default=None, help="comma-separated; one graph per seed")
    ap.add_argument("--cold_target", action="store_true")
    ap.add_argument("--out_pattern", default="data/graphs/graph_seed{seed}.pkl")
    args = ap.parse_args()

    d = pickle.load(open(IN_PATH, "rb"))
    nc = int(d["num_compounds"])
    cids, pids = d["compound_ids"], d["protein_ids"]
    ei, ea = np.asarray(d["cp_edge_index"]), np.asarray(d["cp_edge_attr"])
    first = {}
    for i in range(0, ei.shape[1], 2):
        first[(int(ei[0, i]), int(ei[1, i]) - nc)] = round(float(ea[i, 0]) * 12.0, 2)
    recorded = set(first)

    con = sqlite3.connect(args.chembl_db)
    con.execute("CREATE TEMP TABLE cids (id TEXT PRIMARY KEY)")
    con.execute("CREATE TEMP TABLE tids (id TEXT PRIMARY KEY)")
    con.executemany("INSERT INTO temp.cids VALUES (?)", [(c,) for c in cids])
    con.executemany("INSERT INTO temp.tids VALUES (?)", [(p,) for p in pids])
    c_idx = {c: i for i, c in enumerate(cids)}
    p_idx = {p: i for i, p in enumerate(pids)}
    values = defaultdict(list)
    for cid, tid, v in con.execute(QUERY, (MAX_RELEASE,)):
        pair = (c_idx[cid], p_idx[tid])
        if pair in recorded:
            values[pair].append(float(v))
    con.close()
    median = {p: round(float(np.median(v)), 2) for p, v in values.items()}

    potent = {p for p, v in median.items() if v >= THRESHOLD}
    old_potent = {p for p, v in first.items() if v >= THRESHOLD}
    seeds = [int(s) for s in args.split_seeds.split(",")] if args.split_seeds else [42]
    base = build_relabeled(d, recorded, potent, median, values, nc)
    for seed in seeds:
        if args.cold_target:
            tr_p, va_p, te_p = cold_target_split(potent, seed)
            kind = f"cold-target (protein-disjoint) 70/15/15 over proteins, random_state={seed}"
        else:
            tr, va, te = trainer_split(d, random_state=seed)
            assert len(set(tr) | set(va) | set(te)) == len(recorded)
            tr_p, va_p, te_p = ([p for p in s if p in potent] for s in (tr, va, te))
            kind = f"random 70/15/15 over recorded pairs, random_state={seed}"
        out = dict(base)
        out["split_override"] = {"train": tr_p, "val": va_p, "test": te_p}
        out["split_kind"] = kind
        path = OUT_PATH if args.split_seeds is None else args.out_pattern.format(seed=seed)
        pickle.dump(out, open(path, "wb"))
        print(f"{path}: {kind}; train/val/test {len(tr_p)}/{len(va_p)}/{len(te_p)}; proteins "
              f"{len({p for _, p in tr_p})}/{len({p for _, p in va_p})}/{len({p for _, p in te_p})}; compounds "
              f"{len({c for c, _ in tr_p})}/{len({c for c, _ in va_p})}/{len({c for c, _ in te_p})}", flush=True)

    nrec = np.array([len(v) for v in values.values()])
    pv = np.array([median[p] for p in potent])
    print(f"recorded pairs {len(recorded)}; with pChEMBL (median) {len(median)}; records/pair median "
          f"{np.median(nrec):.0f}, max {nrec.max()}")
    print(f"potent: median {len(potent)} vs first-record {len(old_potent)}; "
          f"both {len(potent & old_potent)}, only median {len(potent - old_potent)}, "
          f"only first {len(old_potent - potent)}")
    print(f"compounds with DTI {len({c for c, _ in potent})}, proteins with DTI {len({p for _, p in potent})}; "
          f"median pChEMBL {np.median(pv):.2f} (range {pv.min():.2f}-{pv.max():.2f})")


def build_relabeled(d, recorded, potent, median, values, nc):
    """The graph with only potent CP edges, their attributes recomputed from the median pChEMBL."""
    out = remove_cp_pairs(d, sorted(recorded - potent))
    assert out["cp_edge_index"].shape[1] == 2 * len(potent)
    n = len(out["node_types"])
    attr = {}
    for (c, p) in potent:
        v = median[(c, p)]
        a = np.array([v / 12.0, 1.0 if v > 6 else 0.5], dtype=np.float32)
        attr[c * n + (p + nc)] = a
        attr[(p + nc) * n + c] = a
    for ei_key, ea_key in [("edge_index", "edge_attr"), ("cp_edge_index", "cp_edge_attr")]:
        e = np.asarray(out[ei_key]).astype(np.int64)
        new_attr = np.array(out[ea_key], dtype=np.float32, copy=True)
        keys = e[0] * n + e[1]
        hit = 0
        for j, k in enumerate(keys):
            a = attr.get(int(k))
            if a is not None:
                new_attr[j, :2] = a
                hit += 1
        assert hit == 2 * len(potent), (ei_key, hit)
        out[ea_key] = new_attr

    out["all_known_pairs"] = sorted(recorded)
    out["all_known_pchembl"] = {p: median.get(p, 0.0) for p in recorded}
    out["pchembl_n_records"] = {p: len(v) for p, v in values.items()}
    out["label_definition"] = (f"positive = recorded ChEMBL35 pair with median pChEMBL >= {THRESHOLD} "
                               f"over ChEMBL records from documents released in ChEMBL <= {MAX_RELEASE}")
    return out


if __name__ == "__main__":
    main()
