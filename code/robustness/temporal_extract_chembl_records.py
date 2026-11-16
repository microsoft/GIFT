"""Extract the ChEMBL activity records behind the evaluation pairs (points 10 and 11).

From the ChEMBL 36 SQLite: every activity record (any type, with or without pChEMBL) of the 95 external compounds and
of the 1,559 ChEMBL 35 graph compounds at the graph proteins, with the release of its source document
(docs.chembl_release_id), assay_type, assays.confidence_score, relationship_type, pChEMBL and data_validity_comment.
From the ChEMBL 35 SQLite: every activity record of the external compounds at the graph proteins, matched by exact
ChEMBL ID and, separately, through molecule_hierarchy (all forms sharing the compound's parent: salts/parent).
Writes results/robustness/temporal_direct_target/records_c36.pkl, records_c35_external.pkl.
Usage (repo root): python code/robustness/temporal_extract_chembl_records.py [--c36 DB] [--c35 DB]
"""
import os as _os, sys as _sys  # release bootstrap: run from any directory as python code/robustness/<script>.py
_REPO = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", ".."))
_sys.path[:0] = [_os.path.join(_REPO, "code"), _os.path.join(_REPO, "code", "robustness")]
_os.chdir(_REPO)
import argparse
import os
import sqlite3

import numpy as np
import pandas as pd

OUT = "results/robustness/temporal_direct_target"
C36 = "data/chembl_db/chembl_36/chembl_36_sqlite/chembl_36.db"
C35 = "data/chembl_db/chembl_35/chembl_35_sqlite/chembl_35.db"

COLS = """act.activity_id, act.molregno, td.chembl_id AS target, act.standard_type, act.pchembl_value,
          act.data_validity_comment, a.assay_type, a.confidence_score, a.relationship_type, a.chembl_id AS assay,
          act.doc_id"""


def connect(db, cids, tids):
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    con.execute("CREATE TEMP TABLE cids (id TEXT PRIMARY KEY)")
    con.execute("CREATE TEMP TABLE tids (id TEXT PRIMARY KEY)")
    con.executemany("INSERT INTO temp.cids VALUES (?)", [(c,) for c in cids])
    con.executemany("INSERT INTO temp.tids VALUES (?)", [(t,) for t in tids])
    return con


def family(con):
    """compound chembl_id -> molregnos sharing its parent (molecule_hierarchy), incl. itself."""
    rows = con.execute("""
        SELECT q.id, md.molregno, COALESCE(h.parent_molregno, md.molregno)
        FROM temp.cids q JOIN molecule_dictionary md ON md.chembl_id = q.id
        LEFT JOIN molecule_hierarchy h ON h.molregno = md.molregno""").fetchall()
    fam = []
    for cid, mol, parent in rows:
        kids = con.execute("SELECT molregno FROM molecule_hierarchy WHERE parent_molregno = ? OR molregno = ?",
                           (parent, parent)).fetchall()
        for m in {mol, parent} | {k[0] for k in kids}:
            fam.append((cid, int(m), int(m == mol)))
    return pd.DataFrame(fam, columns=["compound", "molregno", "exact"]).drop_duplicates()


def records(con, fam, release_col):
    con.execute("CREATE TEMP TABLE mols (molregno INTEGER PRIMARY KEY)")
    con.executemany("INSERT OR IGNORE INTO temp.mols VALUES (?)", [(int(m),) for m in fam.molregno.unique()])
    rel = ", d.chembl_release_id AS doc_release" if release_col else ""
    df = pd.read_sql_query(f"""
        SELECT {COLS}{rel}
        FROM activities act
        JOIN assays a ON act.assay_id = a.assay_id
        JOIN target_dictionary td ON a.tid = td.tid
        LEFT JOIN docs d ON act.doc_id = d.doc_id
        WHERE act.molregno IN (SELECT molregno FROM temp.mols) AND td.chembl_id IN (SELECT id FROM temp.tids)""", con)
    return df.merge(fam, on="molregno")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--c36", default=C36)
    ap.add_argument("--c35", default=C35)
    a = ap.parse_args()
    z = np.load(f"{OUT}/preds/seed_42.npz")
    cids, tids = [str(c) for c in z["compound_ids"]], [str(t) for t in z["protein_ids"]]

    if not os.path.exists(f"{OUT}/records_c36.pkl"):
        con = connect(a.c36, cids, tids)
        fam = family(con)
        df = records(con, fam, True)
        df.to_pickle(f"{OUT}/records_c36.pkl")
        print("C36 records", len(df), "exact", int(df.exact.sum()))
    if os.path.exists(a.c35):
        ext = cids[int(z["nc"]):]
        con = connect(a.c35, ext, tids)
        fam = family(con)
        df = records(con, fam, True)
        df.to_pickle(f"{OUT}/records_c35_external.pkl")
        print("C35 external records", len(df), "exact", int(df.exact.sum()),
              "| external compounds found in C35:", fam[fam.exact == 1].compound.nunique(), "/", len(ext))
        print(con.execute("SELECT * FROM chembl_release ORDER BY 1 DESC LIMIT 2").fetchall())


if __name__ == "__main__":
    main()
