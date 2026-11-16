"""
External ChEMBL 36 labels with the same activity definition as the ChEMBL 35 graph.

For the external compounds of the original set (data/external_compounds_source.pkl joined with the
external feature PKL), all ChEMBL 36 pChEMBL records at graph proteins are collected, records flagged by
data_validity_comment (other than 'Manually validated') are removed, and a pair is positive if the
median pChEMBL is >= 5. Every compound-protein pair with any ChEMBL 36 activity record is stored as
tested, so that tested pairs are never used as external negatives.
Output: data/external_compounds_labels.pkl (same layout as the original interaction PKL).
"""

import argparse
import os
import contextlib
import io
import pickle
import sqlite3
from collections import defaultdict

import numpy as np

import pipeline_utils as vs

OUT = "data/external_compounds_labels.pkl"
THRESHOLD = 5.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph_path", default="data/graphs/graph_run.pkl")
    ap.add_argument("--chembl_db", default=os.environ.get("CHEMBL36_DB"), required="CHEMBL36_DB" not in os.environ,
                    help="ChEMBL 36 SQLite file (default: $CHEMBL36_DB)")
    args = ap.parse_args()

    g = pickle.load(open(args.graph_path, "rb"))
    with contextlib.redirect_stdout(io.StringIO()):
        aug, old_pairs, old_meta = vs.load_external_chembl36_interactions_from_pkls(g)
    ext_ids = list(aug["external_compound_ids"])
    old = pickle.load(open("data/external_compounds_source.pkl", "rb"))
    graph_targets = old["graph_targets"]
    tset = set(graph_targets)

    con = sqlite3.connect(f"file:{args.chembl_db}?mode=ro", uri=True)
    con.execute("CREATE TEMP TABLE cids (id TEXT PRIMARY KEY)")
    con.executemany("INSERT INTO temp.cids VALUES (?)", [(c,) for c in ext_ids])
    rows = con.execute("""
        SELECT md.chembl_id, td.chembl_id, act.pchembl_value, act.data_validity_comment
        FROM activities act
        JOIN molecule_dictionary md ON act.molregno = md.molregno
        JOIN assays a ON act.assay_id = a.assay_id
        JOIN target_dictionary td ON a.tid = td.tid
        WHERE md.chembl_id IN (SELECT id FROM temp.cids)""").fetchall()
    con.close()

    values, tested = defaultdict(list), defaultdict(set)
    for cid, tid, v, dvc in rows:
        if tid not in tset:
            continue
        tested[cid].add(tid)
        if v is not None and (dvc is None or dvc == "Manually validated"):
            values[(cid, tid)].append(float(v))

    out_compounds, n_pos, n_old_pos_kept = {}, 0, 0
    old_pos = {(old_meta[p]["compound_id"], old_meta[p]["protein_id"]) for p in old_pairs}
    for cid in ext_ids:
        rec = old["external_compounds"][cid]
        inter = [(tid, round(float(np.median(v)), 2)) for (c, tid), v in values.items()
                 if c == cid and np.median(v) >= THRESHOLD]
        if not inter:
            continue
        n_pos += len(inter)
        n_old_pos_kept += sum((cid, t) in old_pos for t, _ in inter)
        out_compounds[cid] = {"chembl_id": cid, "smiles": rec["smiles"], "max_phase": rec.get("max_phase"),
                              "interactions": sorted(inter), "tested_targets": sorted(tested[cid])}
    meta = {"source_db": "ChEMBL 36", "label": f"median pChEMBL >= {THRESHOLD} over unflagged records",
            "num_external_compounds": len(out_compounds), "total_interactions": n_pos}
    pickle.dump({"external_compounds": out_compounds, "graph_targets": graph_targets, "metadata": meta}, open(OUT, "wb"))
    pv = [v for c in out_compounds.values() for _, v in c["interactions"]]
    print(f"old: {len(ext_ids)} compounds, {len(old_pairs)} positives (max pChEMBL >= 5)")
    print(f"new: {len(out_compounds)} compounds, {n_pos} positives (median >= 5); {n_old_pos_kept} also old positives, "
          f"{len(old_pairs) - n_old_pos_kept} old positives dropped; targets {len({t for c in out_compounds.values() for t, _ in c['interactions']})}")
    print(f"median pChEMBL {np.median(pv):.2f} (range {min(pv):.2f}-{max(pv):.2f}); tested pairs "
          f"{sum(len(c['tested_targets']) for c in out_compounds.values())}; saved {OUT}")


if __name__ == "__main__":
    main()
