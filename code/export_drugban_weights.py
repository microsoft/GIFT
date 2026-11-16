"""Export the trained DrugBAN baseline of the 30 runs as standalone weights (results/gift/baseline_models/drugban).

The official-DrugBAN runs (train_drugban.py) keep the validation-selected weights inside their per-epoch
resume checkpoint (results/gift/baseline_models/drugban/run_s/drugban_dgl/scores.npz.ckpt, key st["best_state"]). This script copies them to
drugban_seed_s.pth in the format of the DeepDTA/GraphDTA saves and checks that the best epoch and validation ROC-AUC
match the metadata of the scores behind the reported DrugBAN results (drugban_dgl/scores.npz).
Usage: python export_drugban_weights.py [--out results/gift/baseline_models/drugban] [SEED ...]
"""
import argparse
import json
import os

import numpy as np
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/gift/baseline_models/drugban")
    ap.add_argument("seeds", nargs="*", type=int, default=list(range(42, 72)))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    for s in a.seeds:
        d = f"results/gift/baseline_models/drugban/run_{s}/drugban_dgl"
        st = torch.load(f"{d}/scores.npz.ckpt", weights_only=False, map_location="cpu")["st"]
        meta = json.loads(str(np.load(f"{d}/scores.npz")["meta"]))
        assert st["best_state"] is not None, f"seed {s}: no best_state"
        assert st["best_epoch"] == meta["best_epoch"] and abs(st["best"] - meta["val_roc_auc"]) < 1e-12, \
            f"seed {s}: checkpoint does not match the reported scores"
        out = os.path.join(a.out, f"drugban_seed_{s}.pth")
        torch.save({"seed": s, "kind": "drugban", "implementation": meta["implementation"], "config": meta["config"],
                    "best_epoch": st["best_epoch"], "stopped_at_epoch": st["stop"], "val_roc_auc": st["best"],
                    "source": f"{d}/scores.npz.ckpt", "model_state_dict": st["best_state"]}, out)
        saved = torch.load(out, weights_only=False, map_location="cpu")["model_state_dict"]
        same = saved.keys() == st["best_state"].keys() and all(torch.equal(saved[k], v) for k, v in st["best_state"].items())
        print(f"seed {s}: best_epoch {st['best_epoch']} val_roc_auc {st['best']:.4f} -> {out} identical={same}",
              flush=True)
        assert same, f"seed {s}: saved weights differ"


if __name__ == "__main__":
    main()
