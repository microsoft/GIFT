"""Point 6: N-terminal truncation of protein sequences in the DTA baselines (DeepDTA/GraphDTA 1000, DrugBAN 1200).

train_dta_baselines.py: label_seq(s, vocab, n) encodes s[:n] (N-terminal prefix, zero-padded), MAX_SEQ=1000,
DRUGBAN_MAX_SEQ=1200; the official DrugBAN (train_drugban.py) uses its own integer_label_protein, which also
keeps the first 1200 residues. Gene symbols, UniProt accessions and Pfam protein-kinase domain coordinates
(PF00069 Pkinase / PF07714 Pkinase_Tyr) come from the local ChEMBL 36 SQLite (component_* / domains tables).
Writes pt6_out.json and pt6_proteins.csv. CPU only.
"""
import os as _os, sys as _sys  # release bootstrap: run from any directory as python code/robustness/<script>.py
_REPO = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", ".."))
_sys.path[:0] = [_os.path.join(_REPO, "code"), _os.path.join(_REPO, "code", "robustness")]
_os.chdir(_REPO)
import csv
import json
import os
import pickle
import sqlite3
import sys

ROOT = _REPO
sys.path.insert(0, ROOT)
os.chdir(ROOT)
import numpy as np  # noqa: E402

import pipeline_utils as vs  # noqa: E402

OUT = "results/robustness/domain_crop"
DB = os.environ.get("CHEMBL36_DB", "data/chembl_db/chembl_36/chembl_36_sqlite/chembl_36.db")
CUTS = (1000, 1200)
PK_PFAM = ("PF00069", "PF07714")
DRIVERS = {"CHEMBL203": "EGFR", "CHEMBL1824": "ERBB2", "CHEMBL4247": "ALK", "CHEMBL5568": "ROS1",
           "CHEMBL2041": "RET", "CHEMBL3717": "MET", "CHEMBL5145": "BRAF", "CHEMBL2815": "NTRK1",
           "CHEMBL4898": "NTRK2", "CHEMBL5608": "NTRK3"}
APPROX = {"EGFR": (712, 979), "ERBB2": (720, 987), "ALK": (1116, 1392), "ROS1": (1945, 2222), "RET": (724, 1005),
          "MET": (1078, 1345), "BRAF": (457, 717), "NTRK1": (510, 781), "NTRK2": (538, 807), "NTRK3": (538, 809)}


def domain_status(dom, cut):
    """'inside' if the whole kinase domain is within the first `cut` residues, 'partial' if cut inside it,
    'beyond' if it starts after the cut, None if the protein has no PK domain."""
    if not dom:
        return None
    s, e = min(d[0] for d in dom), max(d[1] for d in dom)
    return "inside" if e <= cut else ("beyond" if s > cut else "partial")


def main():
    seqs = json.load(open("data/protein_sequences.json"))
    con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    rows = con.execute("""
        SELECT td.chembl_id, td.target_type, td.pref_name, td.organism, cs.component_id, cs.accession,
               length(cs.sequence)
        FROM target_dictionary td JOIN target_components tc ON td.tid = tc.tid
        JOIN component_sequences cs ON tc.component_id = cs.component_id""").fetchall()
    comp = {}
    for tid, ttype, name, org, cid, acc, ln in rows:
        comp.setdefault(tid, {"type": ttype, "name": name, "organism": org, "components": []})["components"].append((cid, acc, ln))
    sym = {}
    for cid, s in con.execute("SELECT component_id, component_synonym FROM component_synonyms WHERE syn_type='GENE_SYMBOL'"):
        sym.setdefault(cid, []).append(s)
    pk = {}
    for cid, src, s, e in con.execute("""SELECT cd.component_id, d.source_domain_id, cd.start_position, cd.end_position
            FROM component_domains cd JOIN domains d ON cd.domain_id = d.domain_id
            WHERE d.source_domain_id IN ('PF00069','PF07714')"""):
        pk.setdefault(cid, []).append((s, e, src))
    con.close()

    table = []
    for t, s in seqs.items():
        info = comp.get(t, {"type": None, "name": None, "organism": None, "components": []})
        cs = info["components"]
        # Single-component targets map unambiguously; for complexes/families the sequence file holds one sequence,
        # the component whose ChEMBL length matches it is used when available.
        match = [c for c in cs if c[2] == len(s)]
        c0 = match[0] if match else (cs[0] if len(cs) == 1 else None)
        dom = pk.get(c0[0], []) if c0 else []
        table.append({"target": t, "target_type": info["type"], "pref_name": info["name"], "organism": info["organism"],
                      "n_components": len(cs), "accession": c0[1] if c0 else ";".join(c[1] for c in cs),
                      "gene_symbol": ";".join(sorted(set(sym.get(c0[0], [])))) if c0 else
                      ";".join(sorted({x for c in cs for x in sym.get(c[0], [])})),
                      "len_prefix": len(s), "len_chembl36_component": c0[2] if c0 else None,
                      "pk_domain": ";".join(f"{src}:{a}-{b}" for a, b, src in sorted(dom)),
                      "pk_start": min((a for a, _, _ in dom), default=None),
                      "pk_end": max((b for _, b, _ in dom), default=None),
                      "pk_status_1000": domain_status(dom, 1000), "pk_status_1200": domain_status(dom, 1200)})
    by_t = {r["target"]: r for r in table}
    with open(f"{OUT}/pt6_proteins.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(table[0]))
        w.writeheader()
        w.writerows(table)

    lens = np.array([r["len_prefix"] for r in table])
    res = {"n_proteins": len(table), "length_median": float(np.median(lens)), "length_max": int(lens.max()),
           "n_gt_1000": int((lens > 1000).sum()), "n_gt_1200": int((lens > 1200).sum()),
           "n_with_pk_domain": sum(r["pk_start"] is not None for r in table),
           "pk_domain_status_counts": {str(c): {k: sum(r[f"pk_status_{c}"] == k for r in table)
                                                for k in ("inside", "partial", "beyond")} for c in CUTS},
           "len_mismatch_vs_chembl36_component": sum(r["len_chembl36_component"] not in (None, r["len_prefix"]) for r in table)}

    drivers = {}
    for t, g in DRIVERS.items():
        r = by_t[t]
        a = APPROX[g]
        drivers[g] = {"target": t, "accession": r["accession"], "gene_symbol": r["gene_symbol"], "length": r["len_prefix"],
                      "pfam_pk": r["pk_domain"], "approx_uniprot_kd": f"{a[0]}-{a[1]}",
                      "status_1000": r["pk_status_1000"], "status_1200": r["pk_status_1200"],
                      "residues_of_pk_kept_1000": max(0, min(r["pk_end"], 1000) - r["pk_start"] + 1),
                      "residues_of_pk_kept_1200": max(0, min(r["pk_end"], 1200) - r["pk_start"] + 1),
                      "pk_len": r["pk_end"] - r["pk_start"] + 1}
    res["drivers"] = drivers

    def frac(pairs, pids):
        L = np.array([len(seqs[pids[p]]) for _, p in pairs])
        st = [by_t[pids[p]] for _, p in pairs]
        out = {"n": len(pairs)}
        for c in CUTS:
            out[f"truncated_{c}"] = float((L > c).mean())
            out[f"n_truncated_{c}"] = int((L > c).sum())
            out[f"pk_partial_or_beyond_{c}"] = float(np.mean([r[f"pk_status_{c}"] in ("partial", "beyond") for r in st]))
            out[f"pk_beyond_{c}"] = float(np.mean([r[f"pk_status_{c}"] == "beyond" for r in st]))
        return out

    per_seed = {}
    for seed in range(42, 72):
        d = pickle.load(open(f"data/graphs/graph_seed{seed}.pkl", "rb"))
        pids = list(d["protein_ids"])[:int(d["num_proteins"])]
        sp = vs.derive_chembl35_splits(d, min_training_pchembl=5.0)
        per_seed[seed] = {"test": frac(sp["test_pairs"], pids), "train": frac(sp["train_pairs"], pids)}
        if seed == 42:
            aug, cross, _ = vs.load_external_chembl36_interactions_from_pkls(d)
            res["external_positives"] = frac(cross, pids)
            res["all_chembl35_interactions"] = frac(sorted(sp["all_positive_pairs"]), pids)
    res["test_seed42"] = per_seed[42]["test"]
    res["train_seed42"] = per_seed[42]["train"]
    keys = [k for k in per_seed[42]["test"] if k != "n"]
    res["test_30seeds_mean_sd"] = {k: [float(np.mean([per_seed[s]["test"][k] for s in per_seed])),
                                       float(np.std([per_seed[s]["test"][k] for s in per_seed], ddof=1))] for k in keys + ["n"]}
    res["per_seed"] = per_seed
    json.dump(res, open(f"{OUT}/pt6_out.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "per_seed"}, indent=1))


if __name__ == "__main__":
    main()
