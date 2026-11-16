"""Checks that the replays in negative_overlap_audit.py follow the original code exactly.

1. Phase 1/2 triplet loss: runs the ORIGINAL TripletLossRandomCandidates (as configured in train_backbone.py)
   on random embeddings for a few epochs (train call + val call per epoch) and the replay for the same epochs, with a
   recorder on random.sample; asserts identical (population size, k) sequences and identical final `random` state.
2. RF: refits the default random forest on the replayed training negatives and compares its benchmark metrics with
   the saved eval_final.json of the paper run (results/gift/run_<seed>).
Usage: python verify_replay.py [--seeds 42 57] [--epochs 3]
"""

import os as _os, sys as _sys  # release bootstrap: run from any directory as python code/robustness/<script>.py
_REPO = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", ".."))
_sys.path[:0] = [_os.path.join(_REPO, "code"), _os.path.join(_REPO, "code", "robustness")]
_os.chdir(_REPO)
import argparse
import contextlib
import io
import json
import os
import pickle
import random
import sys

import numpy as np
import torch

HERE = "results/robustness/negative_overlap"
sys.path.insert(0, HERE)
import negative_overlap_audit as A  # noqa: E402  (chdir to repo root)
import contrastive_loss as tl  # noqa: E402
from model_encoder import split_interaction_data  # noqa: E402
from negative_pool import reserved_eval_mask  # noqa: E402


class Recorder:
    def __init__(self):
        self.log = []
        self._sample = random.sample

    def __enter__(self):
        rec = self

        def sample(population, k, **kw):
            rec.log.append((len(population), k))
            return rec._sample(population, k, **kw)
        random.sample = sample
        return self

    def __exit__(self, *a):
        random.sample = self._sample


def verify_triplet(seed, epochs):
    data = pickle.load(open(f"data/graphs/graph_seed{seed}.pkl", "rb"))
    nc, npr = int(data["num_compounds"]), int(data["num_proteins"])
    with contextlib.redirect_stdout(io.StringIO()):
        tr, va, te = split_interaction_data(data)
    train_o, val_o = list(tr), list(va)            # exactly what the backbone passes to the loss
    reserved = reserved_eval_mask(nc, npr)
    crit = tl.TripletLossRandomCandidates(margin=2.0, max_negatives_per_positive=10, batch_size=64, temperature=1.0,
                                        pos_weight=1.0, neg_weight=1.0, hard_negative_mining=True)
    crit.reserved_mask = reserved
    g = torch.Generator().manual_seed(0)
    zc, zp = torch.randn(nc, 64, generator=g), torch.randn(npr, 64, generator=g)
    random.seed(seed)
    with Recorder() as r1, torch.no_grad():
        for _ in range(epochs):
            crit(zc, zp, train_o, apply_regularization=False)
            crit(zc, zp, val_o, apply_regularization=False)
    st1 = random.getstate()

    train = [tuple(map(int, p)) for p in tr]
    val = [tuple(map(int, p)) for p in va]
    held = np.zeros((nc, npr), bool)
    with Recorder() as r2:
        A.replay_triplet(seed, train, val, reserved, nc, npr, held, held, epochs, 0)
    st2 = random.getstate()
    return {"seed": seed, "epochs": epochs, "n_sample_calls": len(r1.log),
            "sample_sizes_identical": r1.log == r2.log, "final_random_state_identical": st1 == st2}


def verify_rf(seed):
    import pipeline_utils as vs
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import roc_auc_score, average_precision_score
    from negative_pool import (dti_random_negatives, evaluation_exclusion, sample_compound_matched_negatives,
                                   sample_negatives)
    data = pickle.load(open(f"data/graphs/graph_seed{seed}.pkl", "rb"))
    nc, npr = int(data["num_compounds"]), int(data["num_proteins"])
    sp = vs.derive_chembl35_splits(data, min_training_pchembl=5.0)
    train, val, test, known = sp["train_pairs"], sp["val_pairs"], sp["test_pairs"], sp["all_positive_pairs"]
    reserved = reserved_eval_mask(nc, npr)
    _, negs = A.replay_rf(seed, train, reserved, nc, npr, set(), set())
    feat = vs.build_pair_feature_fn(data)
    X = np.stack([feat(c, p) for c, p in train + negs])
    y = np.array([1] * len(train) + [0] * len(negs))
    rf = RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=8).fit(X, y)
    excl = evaluation_exclusion(known, reserved)
    rng = np.random.RandomState(vs.SEED)
    sample_negatives(rng, np.arange(nc), npr, 5 * len(test), excl)
    sample_compound_matched_negatives(rng, test, npr, 5, excl)
    random_neg = dti_random_negatives(vs.SEED, train + val + test, npr, 5 * len(test), excl)
    pairs = test + random_neg
    yb = np.array([1] * len(test) + [0] * len(random_neg))
    s = rf.predict_proba(np.stack([feat(c, p) for c, p in pairs]))[:, 1]
    ref = json.load(open(f"results/gift/run_{seed}/eval_final.json"))["results_by_method"]["random_forest"]["same_version"]
    return {"seed": seed, "roc_auc_replay": float(roc_auc_score(yb, s)), "roc_auc_saved": ref["roc_auc"],
            "auprc_replay": float(average_precision_score(yb, s)), "auprc_saved": ref["auprc"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="*", default=[42, 57])
    ap.add_argument("--epochs", type=int, default=3)
    a = ap.parse_args()
    out = {"triplet": [verify_triplet(s, a.epochs) for s in a.seeds], "rf": [verify_rf(s) for s in a.seeds]}
    print(json.dumps(out, indent=1))
    json.dump(out, open(os.path.join(HERE, "verify_replay.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
