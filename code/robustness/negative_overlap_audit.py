"""Reviewer point 1: were validation/test POSITIVE pairs ever drawn as training NEGATIVES?

Re-runs ONLY the negative-sampling code paths of every training stage / method of the paper (main-model protocol,
30 runs, seeds 42-71) with the exact seeds and RNG call sequence of the original scripts (no model training),
and counts how often held-out (validation + test) positives were drawn as training negatives.

Exact replays (the RNG stream of these samplers does not depend on model outputs):
  GIFT Phase 3a  train_classifier.py   (np.random.RandomState(seed), epochs from head.log)
  GIFT Phase 3b  train_end_to_end.py    (np.random.RandomState(seed), steps from e2e.log)
  RF             evaluate_chembl35.py / train_rf_baseline.py (one draw)
  GCN/SAGE/GAT   train_gnn_baselines.py (stop epoch from the saved json)
  DeepDTA/GraphDTA/DrugBAN  train_dta_baselines.py / train_drugban.py (stop/best from json)
Candidate-exact replay (Python `random` stream reproduced exactly; which candidates are kept depends on embeddings):
  GIFT Phase 1/2 contrastive_loss.TripletLossRandomCandidates (via train_backbone.py)

Usage: python negative_overlap_audit.py [--seeds 42 43 ...] [--procs 30]
Writes per_seed.json next to this script (aggregated by summarize.py into results.json).
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
import re
import sys
from multiprocessing import Pool

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUT = "results/robustness/negative_overlap"
sys.path.insert(0, ROOT)
os.chdir(ROOT)

# Original sampler code (imported, not re-implemented).
from negative_pool import (evaluation_exclusion, reserved_eval_mask, sample_negatives as pool_sample_negatives,
                               training_exclusion)
from model_encoder import split_interaction_data
import pipeline_utils as vs
from graph_cleaning import trainer_split
from train_classifier import mixed_negatives, sample_negatives as head_sample_negatives
from train_inductive_common import sample_negatives as sample_negatives_fn
import contrastive_loss as tl

# Run configuration (run_main model_seed.sh, run_shared baseline protocol_seed.sh, script defaults).
PM_ISO, PM_PROT = 0.30, 0.30            # --iso_fraction 0.30 --protein_mask_fraction 0.30
NEG_RATIO = 1                           # --neg_ratio 1 (Phase 3a and 3b)
VAL_NEG_RATIO = 5
HEAD_MAX_EPOCHS, HEAD_PATIENCE = 150, 20
E2E_MAX_STEPS, E2E_TARGET_FRACTION, E2E_PATIENCE_STEPS = 600, 0.30, 12 * 10
GNN_MAX_EPOCHS, GNN_PATIENCE, GNN_DISJOINT = 500, 50, 0.3
P1_EPOCHS = P2_EPOCHS = 200


def read_log_stop_best(path, unit):
    """(stop, best) from a Phase 3a ('Epoch') or Phase 3b ('Step') log; best = last line marked ' *'."""
    txt = open(path).read()
    m = re.search(rf"Early stopping at {unit.lower()} (\d+)", txt)
    stop = int(m.group(1)) if m else (HEAD_MAX_EPOCHS if unit == "Epoch" else E2E_MAX_STEPS)
    stars = re.findall(rf"{unit}\s+(\d+):.*\*\s*$", txt, flags=re.M)
    best = int(stars[-1]) if stars else 0
    return stop, best


class Tally:
    """Counts held-out positives among a stream of drawn training negatives, by epoch."""

    def __init__(self, val_set, test_set, cutoff=None):
        self.val, self.test, self.cutoff = val_set, test_set, cutoff
        self.draws = self.draws_sel = 0
        self.hits = {"val": 0, "test": 0}
        self.hits_sel = {"val": 0, "test": 0}
        self.distinct = {"val": set(), "test": set()}
        self.distinct_sel = {"val": set(), "test": set()}

    def add(self, epoch, negs):
        sel = self.cutoff is None or epoch <= self.cutoff
        self.draws += len(negs)
        if sel:
            self.draws_sel += len(negs)
        for q in negs:
            for k, s in (("val", self.val), ("test", self.test)):
                if q in s:
                    self.hits[k] += 1
                    self.distinct[k].add(q)
                    if sel:
                        self.hits_sel[k] += 1
                        self.distinct_sel[k].add(q)

    def result(self, **extra):
        h = self.hits["val"] + self.hits["test"]
        hs = self.hits_sel["val"] + self.hits_sel["test"]
        return {"draws": self.draws, "heldout_draws": h, "heldout_draws_val": self.hits["val"],
                "heldout_draws_test": self.hits["test"],
                "distinct_heldout": len(self.distinct["val"]) + len(self.distinct["test"]),
                "distinct_val": len(self.distinct["val"]), "distinct_test": len(self.distinct["test"]),
                "draws_to_selected_ckpt": self.draws_sel, "heldout_draws_to_selected_ckpt": hs,
                "distinct_heldout_to_selected_ckpt": len(self.distinct_sel["val"]) + len(self.distinct_sel["test"]),
                "heldout_pairs": sorted(self.distinct["val"] | self.distinct["test"]), **extra}


# ----------------------------------------------------------------------------------------------- Phase 3a / 3b
def replay_head(seed, train, val, test, positives, reserved, nc, npr, stop, best, vset, tset):
    """train_classifier.py main(): RNG calls in order (lines 415-527)."""
    rng = np.random.RandomState(seed)
    excl = training_exclusion(train, reserved)
    val_compounds = sorted({c for c, _ in val})
    head_sample_negatives(rng, np.asarray(val_compounds), npr, VAL_NEG_RATIO * len(val),
                          evaluation_exclusion(positives, reserved))
    t = Tally(vset, tset, cutoff=best)
    all_c = np.arange(nc)
    for epoch in range(1, stop + 1):
        rng.choice(nc, size=int(round(PM_ISO * nc)), replace=False)
        rng.choice(npr, size=int(round(PM_PROT * npr)), replace=False)
        negs = mixed_negatives(rng, train, all_c, npr, NEG_RATIO * len(train), 0.0, excl)
        t.add(epoch, negs)
    return t.result(stop=stop, best=best)


def replay_e2e(seed, train, val, test, positives, reserved, nc, npr, stop, best, vset, tset):
    """train_end_to_end.py main(): RNG calls in order (lines 187-271)."""
    rng = np.random.RandomState(seed)
    excl = training_exclusion(train, reserved)
    val_compounds = sorted({c for c, _ in val})
    sample_negatives_fn(rng, np.asarray(val_compounds), npr, VAL_NEG_RATIO * len(val),
                         evaluation_exclusion(positives, reserved))
    n_targets = int(round(E2E_TARGET_FRACTION * len(train)))
    t = Tally(vset, tset, cutoff=best)
    all_c = np.arange(nc)
    for step in range(1, stop + 1):
        rng.choice(len(train), size=n_targets, replace=False)
        rng.choice(nc, size=int(round(PM_ISO * nc)), replace=False)
        rng.choice(npr, size=int(round(PM_PROT * npr)), replace=False)
        negs = mixed_negatives(rng, train, all_c, npr, NEG_RATIO * n_targets, 0.0, excl)
        t.add(step, negs)
    return t.result(stop=stop, best=best, n_targets=n_targets)


# ----------------------------------------------------------------------------------------------- baselines
def replay_rf(seed, train, reserved, nc, npr, vset, tset):
    """evaluate_chembl35.py:203-206 (= train_rf_baseline.py:44-45), RF_NEG_RATIO = 1."""
    negs = pool_sample_negatives(np.random.RandomState(seed), np.arange(nc), npr, 1 * len(train),
                                 training_exclusion(train, reserved))
    t = Tally(vset, tset)
    t.add(1, negs)
    return t.result(), negs


def replay_gnn(seed, train_all, reserved, nc, npr, stop, vset, tset):
    """train_gnn_baselines.py:145-166 (per model kind; identical stream for gcn/sage/gat)."""
    rng = np.random.RandomState(seed)
    excl = training_exclusion(train_all, reserved)
    perm = rng.permutation(len(train_all))
    n_sup = int(round(GNN_DISJOINT * len(train_all)))
    best = stop - GNN_PATIENCE if stop < GNN_MAX_EPOCHS else None
    t = Tally(vset, tset, cutoff=best)
    for epoch in range(1, stop + 1):
        negs = pool_sample_negatives(rng, np.arange(nc), npr, n_sup, excl)
        t.add(epoch, negs)
    return t.result(stop=stop, best=best, n_sup=n_sup, perm_head=int(perm[0]))


def replay_dta(seed, train, reserved, nc, npr, stop, best, vset, tset):
    """train_dta_baselines.py:409-418 and train_drugban.py:80-99 (same stream)."""
    rng = np.random.RandomState(seed)
    excl = training_exclusion(train, reserved)
    t = Tally(vset, tset, cutoff=best)
    for epoch in range(1, stop + 1):
        negs = pool_sample_negatives(rng, np.arange(nc), npr, len(train), excl)
        rng.permutation(len(train) + len(negs))
        t.add(epoch, negs)
    return t.result(stop=stop, best=best)


# ----------------------------------------------------------------------------------------------- Phase 1/2
def replay_triplet(seed, train, val, reserved, nc, npr, heldout_mat, test_mat, p1_epochs, p2_epochs):
    """Replays the Python `random` stream of TripletLossRandomCandidates._pair_embedding_loss_advanced
    (contrastive_loss.py:21-88) as called by train_backbone.main(): per epoch one call on the training
    positives (gradient) and one on the validation positives (val loss, checkpoint selection). random.sample /
    randint consume the stream as a function of population sizes only, so candidate sets are reproduced exactly;
    the 7 'hard' negatives of each anchor are the most similar candidates (model-dependent) and are not replayed."""
    random.seed(seed)  # set_seed(args.seed) in main(); nothing else in the backbone process uses `random`
    K, R_, P_ = tl.CANDIDATES_PER_SOURCE, tl.RANDOM_CANDIDATES, tl.POSITIVES_PER_ANCHOR
    MAXNEG, BATCH = 10, 64

    def blocked_matrix(pos):
        b = reserved.copy()
        pc = np.array([c for c, _ in pos]); pp = np.array([p for _, p in pos])
        b[pc, pp] = True
        return b

    mats = {"train": blocked_matrix(train), "val": blocked_matrix(val)}
    counts = {}
    for k_, pos_ in (("train", train), ("val", val)):
        counts[k_] = {}
        for q in pos_:
            counts[k_][q] = counts[k_].get(q, 0) + 1
    free_rows = {k: [np.flatnonzero(~m[c]) for c in range(nc)] for k, m in mats.items()}
    free_cols = {k: [np.flatnonzero(~m[:, p]) for p in range(npr)] for k, m in mats.items()}

    def call(kind, pos, stats):
        B, fr, fc = mats[kind], free_rows[kind], free_cols[kind]
        hmat = heldout_mat if kind == "train" else test_mat
        n = len(pos)
        mult = counts[kind]
        batch = [pos[i] for i in random.sample(range(n), min(BATCH, n))]
        for c, p in batch:
            sc = fr[c]                       # proteins k != p, not blocked (p itself is blocked)
            sp = fc[p]
            ks = sc[random.sample(range(len(sc)), min(K, len(sc)))]
            ls = sp[random.sample(range(len(sp)), min(K, len(sp)))]
            rc, rp = [], []
            for _ in range(R_):
                a, b = random.randint(0, nc - 1), random.randint(0, npr - 1)
                if not B[a, b]:
                    rc.append(a); rp.append(b)
            L = len(ks) + len(ls) + len(rc)
            num_hard = min(int(MAXNEG * 0.7), L)
            rest = L - num_hard
            random.sample(range(rest), min(MAXNEG - num_hard, rest))
            h_sc = int(hmat[c, ks].sum()); h_sp = int(hmat[ls, p].sum())
            h_r = int(hmat[np.array(rc, dtype=int), np.array(rp, dtype=int)].sum()) if rc else 0
            h = h_sc + h_sp + h_r
            nsel = min(MAXNEG, L)
            stats["anchors"] += 1
            stats["candidates"] += L
            stats["neg_slots"] += nsel
            stats["cand_heldout"] += h
            stats["cand_heldout_same_compound"] += h_sc
            stats["cand_heldout_same_protein"] += h_sp
            stats["cand_heldout_random"] += h_r
            stats["anchors_with_heldout_cand"] += int(h > 0)
            stats["exp_selected_uniform"] += h * nsel / L if L else 0.0
            stats["max_selected_bound"] += min(h, nsel)
            if h:
                for k_ in ks[hmat[c, ks]]:
                    stats["_distinct"].add((int(c), int(k_)))
                for l_ in ls[hmat[ls, p]]:
                    stats["_distinct"].add((int(l_), int(p)))
                for a, b in zip(rc, rp):
                    if hmat[a, b]:
                        stats["_distinct"].add((int(a), int(b)))
        for c, p in batch:
            others = n - mult[(c, p)]   # len([q for q in positive_pairs if q != (c, p)])
            if others:
                random.sample(range(others), min(P_, others))

    def new():
        return {"anchors": 0, "candidates": 0, "neg_slots": 0, "cand_heldout": 0, "cand_heldout_same_compound": 0,
                "cand_heldout_same_protein": 0, "cand_heldout_random": 0, "anchors_with_heldout_cand": 0,
                "exp_selected_uniform": 0.0, "max_selected_bound": 0, "_distinct": set()}

    out = {}
    for phase, epochs in (("phase1", p1_epochs), ("phase2", p2_epochs)):
        tr, va = new(), new()
        for _ in range(epochs):
            call("train", train, tr)
            call("val", val, va)
        for s in (tr, va):
            s["distinct_heldout_candidates"] = len(s.pop("_distinct"))
        out[phase] = {"train_call": tr, "val_loss_call_test_positives": va, "epochs": epochs}
    return out


# ----------------------------------------------------------------------------------------------- per seed
def run_seed(seed):
    g = f"data/graphs/graph_seed{seed}.pkl"
    data = pickle.load(open(g, "rb"))
    nc, npr = int(data["num_compounds"]), int(data["num_proteins"])
    with contextlib.redirect_stdout(io.StringIO()):
        tr, va, te = split_interaction_data(data)
    train = [tuple(map(int, p)) for p in tr]
    val = [tuple(map(int, p)) for p in va]
    test = [tuple(map(int, p)) for p in te]
    # Consistency of the split definitions used by the different scripts.
    ctr, cva, cte = trainer_split(data)
    sp = vs.derive_chembl35_splits(data, min_training_pchembl=5.0)
    assert ctr == train and cva == val and cte == test
    assert sorted(sp["train_pairs"]) == sorted(train) and set(sp["val_pairs"]) == set(val) and set(sp["test_pairs"]) == set(test)
    vset, tset = set(val), set(test)
    held = vset | tset
    assert not (held & set(train))
    reserved = reserved_eval_mask(nc, npr)
    trainset = set(train)
    positives = trainset | held | {tuple(map(int, p)) for p in data.get("all_known_pairs", [])}

    held_mat = np.zeros((nc, npr), bool)
    for c, p in held:
        held_mat[c, p] = True
    test_mat = np.zeros((nc, npr), bool)
    for c, p in test:
        test_mat[c, p] = True
    train_mat = np.zeros((nc, npr), bool)
    for c, p in train:
        train_mat[c, p] = True
    excl_mat = reserved | train_mat
    pool_size = int((~excl_mat).sum())
    elig_mat = held_mat & ~excl_mat
    static = {
        "num_compounds": nc, "num_proteins": npr, "all_pairs": nc * npr, "reserved_pairs": int(reserved.sum()),
        "n_train_pos": len(train), "n_val_pos": len(val), "n_test_pos": len(test),
        "n_heldout_pos": len(held),
        "heldout_in_reserve": int((held_mat & reserved).sum()),
        "val_in_reserve": int(sum(bool(reserved[c, p]) for c, p in val)),
        "test_in_reserve": int(sum(bool(reserved[c, p]) for c, p in test)),
        "heldout_eligible": int(elig_mat.sum()),
        "val_eligible": int(sum(not reserved[c, p] for c, p in val)),
        "test_eligible": int(sum(not reserved[c, p] for c, p in test)),
        "training_negative_pool_size": pool_size,
        "heldout_fraction_of_pool": float(elig_mat.sum() / pool_size),
        "heldout_fraction_of_all_pairs": float(len(held) / (nc * npr)),
        "all_known_recorded_pairs": len(data.get("all_known_pairs", [])),
    }

    res = {"seed": seed, "static": static}
    # GIFT Phase 3a / 3b (main run) + controls sharing the same scripts and arguments.
    runs = {"main": f"results/gift/run_{seed}", "ablation_no_softedge": f"results/gift/ablation_no_softedge_seed_{seed}",
            "descriptor_only": f"results/gift/descriptor_only_seed_{seed}", "random_aug": f"results/gift/random_aug_seed_{seed}"}
    for name, d in runs.items():
        hs, hb = read_log_stop_best(f"{d}/head.log", "Epoch")
        es, eb = read_log_stop_best(f"{d}/e2e.log", "Step")
        assert hs - hb == HEAD_PATIENCE or hs == HEAD_MAX_EPOCHS, (d, hs, hb)
        assert es - eb == E2E_PATIENCE_STEPS or es == E2E_MAX_STEPS, (d, es, eb)
        res[f"phase3a_{name}"] = replay_head(seed, train, val, test, positives, reserved, nc, npr, hs, hb, vset, tset)
        res[f"phase3b_{name}"] = replay_e2e(seed, train, val, test, positives, reserved, nc, npr, es, eb, vset, tset)

    rf, rf_negs = replay_rf(seed, train, reserved, nc, npr, vset, tset)
    res["rf"] = rf

    gj = json.load(open(f"results/gift/baseline_models/gnn/gnn_baselines_std_seed_{seed}.json"))["results"]
    with contextlib.redirect_stdout(io.StringIO()):
        _, train_all, _, _ = __import__("graph_cleaning").clean_graph(data)
    train_all = sorted({tuple(map(int, p)) for p in train_all})
    for kind in ("gcn", "sage", "gat"):
        res[f"gnn_{kind}"] = replay_gnn(seed, train_all, reserved, nc, npr, int(gj[kind]["stopped_at_epoch"]), vset, tset)

    dj = json.load(open(f"results/gift/dta_leadprio/dta_metrics_seed_{seed}.json"))["results"]
    bj = json.load(open(f"results/gift/baseline_models/drugban/run_{seed}/dta_baselines_drugban.json"))["results"]["drugban"]
    train_sorted = sorted(sp["train_pairs"])
    for kind, meta in (("deepdta", dj["deepdta"]), ("graphdta", dj["graphdta"]), ("drugban", bj)):
        res[kind] = replay_dta(seed, train_sorted, reserved, nc, npr, int(meta["stopped_at_epoch"]),
                               int(meta["best_epoch"]), vset, tset)

    # GIFT Phase 1/2 backbone (results/backbones/backbone_seed_s, train_backbone.py, no early stopping).
    log = open(f"results/backbones/backbone_seed_{seed}/train.log").read()
    assert "Early stopping" not in log and "backbone saved" in log and "Skipping" not in log
    res["phase12_main"] = replay_triplet(seed, train, val, reserved, nc, npr, held_mat, test_mat, P1_EPOCHS, P2_EPOCHS)
    # Controls: Phase 2 re-run with --resume_phase2 (Python stream starts at Phase 2), same loss class.
    for d in (f"results/backbones/ablation_no_softedge_seed_{seed}/backbone/train.log",
              f"results/gift/random_aug_seed_{seed}/backbone/train.log"):
        lg = open(d).read()
        assert "Early stopping" not in lg and "backbone saved" in lg, d
    res["phase2_controls_resume"] = replay_triplet(seed, train, val, reserved, nc, npr, held_mat, test_mat, 0,
                                                   P2_EPOCHS)["phase2"]
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="*", default=list(range(42, 72)))
    ap.add_argument("--procs", type=int, default=30)
    ap.add_argument("--out", default=os.path.join(OUT, "per_seed.json"))
    a = ap.parse_args()
    with Pool(min(a.procs, len(a.seeds))) as pool:
        rows = pool.map(run_seed, a.seeds)
    json.dump(rows, open(a.out, "w"), indent=1)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
