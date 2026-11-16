"""DTIAM (Lu et al., Nat. Commun. 16, 2548, 2025; github.com/CSUBioGroup/DTIAM, commit 61d68b8) DTI head trained
under the shared DTI-baseline protocol, mirroring train_drugban.py.

Per run (see run_dtiam_queue.sh):
  1. main env:     train_dta_baselines.py --export_protocol P.pkl  (split, negative rule, validation set,
                   benchmarks, lead-prio drivers; identical to DeepDTA/GraphDTA/DrugBAN)
  2. main env:     featurize_dtiam.py (once for all runs): frozen BerMol (768) + ESM-2 650M (1,280) representations
  3. AutoGluon env: this script, P.pkl -> scores .npz (+ lead-prio json)
  4. main env:     train_dta_baselines.py --import_scores S.npz  (identical metrics and output format)

DTIAM's DTI predictor (code/training_validation.py) is an AutoGluon 0.5.2 TabularPredictor(label="y",
eval_metric="roc_auc") with default presets on the concatenated [compound | protein] representation (utils.pack).
Within the shared protocol: one training negative per training positive, drawn with the same sampler, seed and
exclusions as the first epoch of the other baselines (AutoGluon has no epochs to resample over); the protocol's
validation set is passed as AutoGluon tuning_data, so early stopping, model selection and ensemble weighting use
validation ROC-AUC only (it is not used to fit base models). External compounds are only featurized for inference.
"""
import os as _os, sys as _sys  # release bootstrap: run from any directory as python code/robustness/<script>.py
_REPO = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", ".."))
_sys.path[:0] = [_os.path.join(_REPO, "code"), _os.path.join(_REPO, "code", "robustness")]
_os.chdir(_REPO)
import argparse
import json
import os
import pickle
import shutil
import sys
import time

import numpy as np
import pandas as pd

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, REPO)
import numpy as _np  # protocol/feature pickles were written with numpy 2; the AutoGluon 0.5.2 env has numpy 1.x
sys.modules.setdefault("numpy._core", _np.core)
for _m in ("multiarray", "numeric", "_multiarray_umath"):
    if hasattr(_np.core, _m):
        sys.modules.setdefault("numpy._core." + _m, getattr(_np.core, _m))
import autogluon.tabular  # noqa: E402
from autogluon.tabular import TabularPredictor  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

from negative_pool import sample_negatives  # noqa: E402

IMPLEMENTATION = ("DTIAM (github.com/CSUBioGroup/DTIAM, commit 61d68b8): frozen BerMol (BerMolModel_base.pkl) + "
                  "ESM-2 650M representations, AutoGluon " + autogluon.tabular.__version__ + " TabularPredictor head")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocol", required=True)
    ap.add_argument("--features", required=True)
    ap.add_argument("--scores_out", required=True)
    ap.add_argument("--lead_out", default="")
    ap.add_argument("--workdir", required=True, help="AutoGluon model directory")
    ap.add_argument("--num_cpus", type=int, default=8)
    ap.add_argument("--keep_models", action="store_true")
    args = ap.parse_args()
    t0 = time.time()
    pr = pickle.load(open(args.protocol, "rb"))
    feats = pickle.load(open(args.features, "rb"))
    seed, nc, npr, train = pr["seed"], pr["nc"], pr["npr"], pr["train"]
    C = np.stack([feats["compound"][s] for s in pr["smiles"]])
    P = np.stack([feats["protein"][s] for s in pr["protein_sequences"]])

    def pack(pairs, y=None):
        c = np.asarray([a for a, _ in pairs])
        p = np.asarray([b for _, b in pairs])
        df = pd.DataFrame(np.concatenate([C[c], P[p]], axis=1))
        if y is not None:
            df["y"] = np.asarray(y).astype(int)
        return df

    rng = np.random.RandomState(seed)
    negs = sample_negatives(rng, np.arange(nc), npr, len(train), pr["train_exclude"])
    train_df = pack(train + negs, [1] * len(train) + [0] * len(negs))
    val_df = pack(pr["val_all"], pr["y_val"])
    assert max(c for c, _ in train + negs + pr["val_all"]) < nc

    if os.path.exists(args.workdir):
        shutil.rmtree(args.workdir)
    predictor = TabularPredictor(label="y", eval_metric="roc_auc", path=args.workdir, verbosity=2).fit(
        train_data=train_df, tuning_data=val_df, presets=None, excluded_model_types=[],
        ag_args_fit={"num_cpus": args.num_cpus, "num_gpus": 0})
    lb = predictor.leaderboard(silent=True)

    def predict(pairs):
        return predictor.predict_proba(pack(pairs)).iloc[:, 1].to_numpy(dtype=np.float64)

    val_auc = float(roc_auc_score(pr["y_val"], predict(pr["val_all"])))
    out = {}
    for name, (pairs, y) in pr["benches"].items():
        out[f"{name}|score"] = predict(pairs)
        out[f"{name}|pairs"] = np.asarray(pairs)
        out[f"{name}|y"] = y
    meta = {"kind": "dtiam", "implementation": IMPLEMENTATION, "val_roc_auc": val_auc,
            "config": {"head": "AutoGluon TabularPredictor, default presets (medium_quality), eval_metric roc_auc",
                       "best_model": predictor.get_model_best(), "n_train_rows": len(train_df),
                       "n_features": int(C.shape[1] + P.shape[1]), "training_negatives_per_positive": 1,
                       "leaderboard": lb[["model", "score_val", "fit_time"]].round(5).to_dict("records")},
            "stopped_at_epoch": None, "best_epoch": None, "seconds": time.time() - t0, "device": "cpu"}
    np.savez_compressed(args.scores_out, meta=json.dumps(meta), **out)
    print(f"seed {seed} dtiam val_auc={val_auc:.4f} best_model={meta['config']['best_model']} "
          f"{time.time() - t0:.0f}s", flush=True)

    if args.lead_out:
        # Driver-kinase lead prioritization, identical to train_drugban.py --lead_out.
        ext_drug_idx = pr["ext_drug_idx"]
        driver_actives = {int(p): set(v) for p, v in pr["driver_actives"].items()}
        dr = {}
        for p_d, name in pr["lead_drivers"]:
            s = predict([(d, p_d) for d in ext_drug_idx])
            order = np.argsort(-s, kind="stable")
            acts = driver_actives.get(int(p_d), set())
            top10 = [ext_drug_idx[order[j]] for j in range(min(10, len(order)))]
            dr[name] = {"active_at_1": int(ext_drug_idx[order[0]] in acts),
                        "hits_top10": int(sum(d in acts for d in top10)),
                        "n_actives": len(acts & set(ext_drug_idx))}
        lead = {"per_driver": dr, "drivers_active_at_1": int(sum(v["active_at_1"] for v in dr.values())),
                "hits_top10_sum": int(sum(v["hits_top10"] for v in dr.values())), "n_drivers": len(dr)}
        json.dump({"seed": seed, "lead": {"dtiam": lead}}, open(args.lead_out, "w"), indent=2)
        print(f"seed {seed} dtiam lead: top1={lead['drivers_active_at_1']} top10={lead['hits_top10_sum']}",
              flush=True)
    if not args.keep_models:
        shutil.rmtree(args.workdir, ignore_errors=True)


if __name__ == "__main__":
    main()
