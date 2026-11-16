"""DTIAM frozen representations for every compound and protein of the 30 shared baseline protocols.

Follows third_party/DTIAM/code/data_process/extract_feature.py (commit 61d68b8) exactly:
- compounds: pre-trained BerMol (BerMolModel_base.pkl, official Google-Drive release), pooled output of
  BerMolPreTrainer.transform(smiles) -> 768-d (the official code standardizes SMILES with RDKit internally);
- proteins: ESM-2 esm2_t33_650M_UR50D, last layer (33), sequence truncated to 1,022 residues,
  mean of token representations [1:] (as in the official code, i.e. including the EOS token) -> 1,280-d.
Features are keyed by the SMILES / sequence string so each run's index order is mapped at training time.
Run with: PYTHONPATH=pylib:../../third_party/DTIAM/code/BerMol TORCH_HOME=torch_home \
          python featurize_dtiam.py
"""
import os as _os, sys as _sys  # release bootstrap: run from any directory as python code/robustness/<script>.py
_REPO = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", ".."))
_sys.path[:0] = [_os.path.join(_REPO, "code"), _os.path.join(_REPO, "code", "robustness")]
_os.chdir(_REPO)
import glob
import os
import pickle
import sys
import time

import dill
import esm
import numpy as np
import torch

HERE = "results/robustness/dtiam"
sys.path.insert(0, os.path.abspath(_REPO))  # negative_pool (protocol.pkl objects)
DTIAM_DIR = os.environ.get("DTIAM_DIR", "code/third_party/DTIAM")  # github.com/CSUBioGroup/DTIAM, commit 61d68b8
BERMOL = os.path.join(DTIAM_DIR, "models", "BerMolModel_base.pkl")
OUT = "data/dtiam/dtiam_features.pkl"


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    smiles, seqs = set(), set()
    for f in sorted(glob.glob(os.path.join(HERE, "runs", "run_*", "protocol.pkl"))):
        pr = pickle.load(open(f, "rb"))
        smiles.update(pr["smiles"])
        seqs.update(pr["protein_sequences"])
    smiles, seqs = sorted(smiles), sorted(seqs)
    print(f"{len(smiles)} unique SMILES, {len(seqs)} unique protein sequences", flush=True)
    t0 = time.time()

    with open(BERMOL, "rb") as f:
        comp_model = dill.load(f)
    comp_model.model.to(dev)
    comp_model.model.eval()
    comp = {}
    with torch.no_grad():
        for s in smiles:
            comp[s] = comp_model.transform(s, dev)[1].cpu().numpy().reshape(-1).astype(np.float32)
    del comp_model
    torch.cuda.empty_cache()
    print(f"BerMol done {time.time() - t0:.0f}s", flush=True)

    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    model = model.to(dev).eval()
    conv = alphabet.get_batch_converter()
    layer = model.num_layers
    prot = {}
    with torch.no_grad():
        for i, s in enumerate(seqs):
            _, _, tok = conv([("p", s[:1022])])
            rep = model(tok.to(dev), repr_layers=[layer])["representations"][layer]
            prot[s] = rep[0, 1:].mean(0).cpu().numpy().reshape(-1).astype(np.float32)
            if (i + 1) % 200 == 0:
                print(f"ESM {i + 1}/{len(seqs)} {time.time() - t0:.0f}s", flush=True)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    pickle.dump({"compound": comp, "protein": prot,
                 "compound_model": "BerMol BerMolModel_base.pkl (DTIAM release; pre-trained on GuacaMol/ChEMBL24)",
                 "protein_model": "ESM-2 esm2_t33_650M_UR50D (UniRef50/90 2021_04)"}, open(OUT, "wb"))
    print(f"saved {OUT} ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
