"""Exact-structure overlap between DTIAM's compound pre-training corpus (GuacaMol v1 'all' = BerMol pre-training set,
1.59M ChEMBL24 molecules) and our compounds (ChEMBL35 graph compounds and the 95 external ChEMBL36 drugs).
Structures are compared as RDKit canonical SMILES, with and without stereochemistry. BerMol's pre-training
objectives (masked substructure, motif and descriptor prediction) use no bioactivity labels, so overlap means
structure exposure only, never exposure to a compound-target label.
Run: python pretraining_overlap.py
"""
import os as _os, sys as _sys  # release bootstrap: run from any directory as python code/robustness/<script>.py
_REPO = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", ".."))
_sys.path[:0] = [_os.path.join(_REPO, "code"), _os.path.join(_REPO, "code", "robustness")]
_os.chdir(_REPO)
import json
import os
import pickle
import sys
from multiprocessing import Pool

from rdkit import Chem, RDLogger

HERE = "results/robustness/dtiam"
sys.path.insert(0, os.path.abspath(_REPO))  # negative_pool (protocol.pkl objects)


def canon(s):
    RDLogger.DisableLog("rdApp.*")
    m = Chem.MolFromSmiles(s)
    if m is None:
        return None, None
    return Chem.MolToSmiles(m), Chem.MolToSmiles(m, isomericSmiles=False)


def main():
    pr = pickle.load(open(os.path.join(HERE, "runs", "run_42", "protocol.pkl"), "rb"))
    nc = pr["nc"]
    ours = [canon(s) for s in pr["smiles"]]
    with open(os.environ.get("GUACAMOL_SMILES", "data/dtiam/guacamol_v1_all.smiles")) as f:
        guac = [line.strip() for line in f if line.strip()]
    with Pool(16) as pool:
        gc = pool.map(canon, guac, chunksize=5000)
    iso = {a for a, _ in gc if a}
    flat = {b for _, b in gc if b}
    res = {"guacamol_molecules": len(guac)}
    for name, idx in [("chembl35_graph_compounds", range(nc)), ("external_chembl36_drugs", range(nc, len(ours)))]:
        idx = list(idx)
        res[name] = {"n": len(idx), "in_guacamol_exact": sum(ours[i][0] in iso for i in idx),
                     "in_guacamol_ignoring_stereo": sum(ours[i][1] in flat for i in idx)}
    ext_hits = [pr["smiles"][i] for i in range(nc, len(ours)) if ours[i][1] in flat]
    res["external_in_guacamol_smiles"] = ext_hits
    json.dump(res, open(os.path.join(HERE, "pretraining_overlap.json"), "w"), indent=2)
    print(json.dumps({k: v for k, v in res.items() if k != "external_in_guacamol_smiles"}, indent=2))


if __name__ == "__main__":
    main()
