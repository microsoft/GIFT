"""Point 5 (validation-chosen threshold): DTIAM scores on the shared validation set, from the AutoGluon predictor saved
by dtiam_train.py --keep_models (inference only; no refitting). The validation set is the protocol's val_all,
identical in order and labels to scores_main_combined/run_seed_S.npz val|pairs. As a reproduction check the external
benchmark is rescored and compared with the saved scores.npz. Writes scores_main_combined/dtiam_seed_S.npz.
Run with the AutoGluon env: $DTIAM_PY code/robustness/fixed_budget_score_dtiam_val.py SEED ...
"""
import os as _os, sys as _sys  # release bootstrap: run from any directory as python code/robustness/<script>.py
_REPO = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", ".."))
_sys.path[:0] = [_os.path.join(_REPO, "code"), _os.path.join(_REPO, "code", "robustness")]
_os.chdir(_REPO)
import os
import pickle
import sys

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, REPO)
sys.modules.setdefault("numpy._core", np.core)  # protocol/feature pickles were written with numpy 2
for _m in ("multiarray", "numeric", "_multiarray_umath"):
    if hasattr(np.core, _m):
        sys.modules.setdefault("numpy._core." + _m, getattr(np.core, _m))
import pandas as pd  # noqa: E402
from autogluon.tabular import TabularPredictor  # noqa: E402

H = os.path.join(REPO, "results", "robustness", "dtiam")
OUT = os.path.join(REPO, "results", "robustness", "descriptor_fixed_budget", "scores_main_combined")


def main():
    feats = pickle.load(open(os.path.join(H, "features", "dtiam_features.pkl"), "rb"))
    for s in map(int, sys.argv[1:]):
        d = os.path.join(H, "runs", f"run_{s}")
        pr = pickle.load(open(os.path.join(d, "protocol.pkl"), "rb"))
        C = np.stack([feats["compound"][x] for x in pr["smiles"]])
        P = np.stack([feats["protein"][x] for x in pr["protein_sequences"]])

        def pack(pairs):
            c = np.asarray([a for a, _ in pairs])
            p = np.asarray([b for _, b in pairs])
            return pd.DataFrame(np.concatenate([C[c], P[p]], axis=1))

        pred = TabularPredictor.load(os.environ.get("AG_DIR", os.path.join(d, "ag")), verbosity=0)
        proba = lambda pairs: pred.predict_proba(pack(pairs)).iloc[:, 1].to_numpy(dtype=np.float64)
        saved = np.load(os.path.join(d, "scores.npz"))
        ext = proba([tuple(x) for x in saved["external|pairs"]])
        diff = float(np.max(np.abs(ext - saved["external|score"])))
        assert diff < 1e-6, f"seed {s}: external rescoring differs from saved scores by {diff}"
        val = proba(pr["val_all"])
        np.savez_compressed(os.path.join(OUT, f"dtiam_seed_{s}.npz"), **{
            "val|pairs": np.asarray(pr["val_all"]), "val|y": np.asarray(pr["y_val"]), "dtiam|val|score": val})
        print(f"seed {s}: val n={len(val)}, external rescoring max |diff|={diff:.1e}", flush=True)


if __name__ == "__main__":
    main()
