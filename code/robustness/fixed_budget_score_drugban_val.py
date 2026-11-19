"""DrugBAN validation scores using the local PyTorch/PyG graph backend (CPU).

Loads the run's protocol (results/gift/baseline_models/drugban/run_s/drugban_dgl/protocol.pkl: validation set, benchmarks) and the saved
validation-selected weights (results/gift/baseline_models/drugban/drugban_seed_s.pth), featurizes exactly as
train_drugban.py, scores protocol['val_all'], and checks the model by re-scoring the ChEMBL35
compound-matched benchmark against the saved scores (drugban_dgl/scores.npz, which are used for the benchmarks).
Writes scores/drugban_seed_<s>.npz. Run with the main environment plus requirements-drugban.txt:
  python code/robustness/fixed_budget_score_drugban_val.py --seeds 42 ...
"""
import os as _os, sys as _sys  # release bootstrap: run from any directory as python code/robustness/<script>.py
_REPO = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", ".."))
_sys.path[:0] = [_os.path.join(_REPO, "code"), _os.path.join(_REPO, "code", "robustness")]
_os.chdir(_REPO)
import argparse
import json
import os
import pickle
import sys

import numpy as np
import torch

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "code", "third_party", "DrugBAN_9923f8c"))
os.chdir(REPO)
import pandas as pd  # noqa: E402
from configs import get_cfg_defaults  # noqa: E402
from _drugban_graph import DTIDataset, batch_graphs, integer_label_protein  # noqa: E402
from models import DrugBAN  # noqa: E402

OUT = os.path.join(REPO, "results", "robustness", "descriptor_fixed_budget", "scores")


def one(s, threads):
    torch.set_num_threads(threads)
    pr = pickle.load(open(f"results/gift/baseline_models/drugban/run_{s}/drugban_dgl/protocol.pkl", "rb"))
    cfg = get_cfg_defaults()
    cfg.merge_from_file(os.path.join(REPO, "code", "third_party", "DrugBAN_9923f8c", "DrugBAN.yaml"))
    need = sorted({c for c, _ in pr["val_all"]} | {c for c, _ in pr["benches"]["same_version_matched"][0]})
    feat = DTIDataset(list(range(len(pr["smiles"]))),
                      pd.DataFrame({"SMILES": pr["smiles"], "Protein": "A", "Y": 0}), cfg.DRUG.MAX_NODES)
    G = {i: feat[i][0] for i in need}
    P = torch.tensor(np.array([integer_label_protein(x) for x in pr["protein_sequences"]]))
    model = DrugBAN(**cfg)
    ck = torch.load(f"results/gift/baseline_models/drugban/drugban_seed_{s}.pth", map_location="cpu",
                    weights_only=False)
    model.load_state_dict(ck["model_state_dict"])
    model.eval()

    def predict(pairs, bs=256):
        out = []
        with torch.no_grad():
            for i in range(0, len(pairs), bs):
                b = pairs[i:i + bs]
                _, _, _, sc = model(batch_graphs([G[c] for c, _ in b]), P[[p for _, p in b]])
                out.append(torch.sigmoid(sc).squeeze(1).numpy())
        return np.concatenate(out).astype(np.float64)

    from sklearn.metrics import roc_auc_score
    val = predict(pr["val_all"])
    saved = np.load(f"results/gift/baseline_models/drugban/run_{s}/drugban_dgl/scores.npz")
    mp = pr["benches"]["same_version_matched"][0]
    assert np.array_equal(saved["same_version_matched|pairs"], np.asarray(mp))
    re = predict(mp)
    diff = float(np.abs(re - saved["same_version_matched|score"]).max())
    vauc = roc_auc_score(pr["y_val"], val)
    print(f"seed {s}: val ROC-AUC {vauc:.5f} (saved best {ck['val_roc_auc']:.5f}); "
          f"matched re-score max|diff| {diff:.2e}", flush=True)
    np.savez_compressed(f"{OUT}/drugban_seed_{s}.npz", **{"val|pairs": np.asarray(pr["val_all"]),
                        "val|y": np.asarray(pr["y_val"]), "drugban|val|score": val},
                        check=json.dumps({"val_auc": vauc, "saved_val_auc": ck["val_roc_auc"], "matched_diff": diff}))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(42, 72)))
    ap.add_argument("--threads", type=int, default=8)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    for s in a.seeds:
        one(s, a.threads)


if __name__ == "__main__":
    main()
