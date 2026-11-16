"""Random-initialisation encoder checkpoint for the no-contrastive-pretraining ablation (reviewer point 15B).

Builds the Phase 1-2 encoder (GNNBackbone) exactly as train_backbone.py:main() does before
Phase 1 (same seed via set_seed, same graph, same constructor arguments = the defaults used for the published
backbones), and saves its untrained state_dict. This file stands in for best_model_phase2.pth (Phases 1 and 2
skipped) or for best_model_phase1.pth (--resume_phase2: Phase 1 skipped, Phase 2 run from random init).

Usage: python make_random_init.py --seed 42 --graph_path data/graphs/graph_seed42.pkl --out DIR/random_init.pth
"""
import os as _os, sys as _sys  # release bootstrap: run from any directory as python code/robustness/<script>.py
_REPO = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", ".."))
_sys.path[:0] = [_os.path.join(_REPO, "code"), _os.path.join(_REPO, "code", "robustness")]
_os.chdir(_REPO)
import argparse
import contextlib
import io
import os
import pickle
import sys

import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

from model_encoder import analyze_meaningful_validation_categories, set_seed  # noqa: E402
from model_backbone import GNNBackbone  # noqa: E402
from graph_cleaning import clean_graph  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--graph_path", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--reference", default="", help="optional published phase-2 checkpoint to check keys/shapes")
    a = ap.parse_args()
    set_seed(a.seed)
    with open(a.graph_path, "rb") as f:
        data = pickle.load(f)
    with contextlib.redirect_stdout(io.StringIO()):
        analyze_meaningful_validation_categories(data)
        clean_graph(data)
    edge_dim = data["edge_attr"].shape[1] if data.get("edge_attr") is not None and data["edge_attr"].size > 0 else 1
    # Constructor arguments of train_backbone.py with its argparse defaults (unchanged in all runs).
    model = GNNBackbone(
        input_dim=data["node_features"].shape[1], hidden_dim=128, output_dim=64, num_layers=3, edge_dim=edge_dim,
        use_edge_mask=True, edge_dropout_p=0.15, gate_hidden_dim=16, pair_hidden_dim=288, pair_geo_dim=112,
        feature_mean=data["node_features"].mean(axis=0), feature_std=data["node_features"].std(axis=0),
    )
    sd = model.state_dict()
    if a.reference:
        ref = torch.load(a.reference, map_location="cpu")
        assert set(ref) == set(sd), (set(ref) ^ set(sd))
        assert all(ref[k].shape == sd[k].shape for k in sd)
        # Non-trainable buffers (e.g. feature normalisation statistics) must match the published backbone.
        params = {n for n, _ in model.named_parameters()}
        diff = [k for k in sd if k not in params and not torch.equal(ref[k].cpu().float(), sd[k].float())]
        print(f"reference check ok: {len(sd)} tensors; buffers differing from reference: {diff}")
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    torch.save(sd, a.out)
    print(f"saved random-init encoder ({sum(v.numel() for v in sd.values())} values) to {a.out}")


if __name__ == "__main__":
    main()
