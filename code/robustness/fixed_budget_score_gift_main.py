"""Runs fixed_budget_score_gift.py unchanged on the main model checkpoints (manuscript model version) for the point-4 sensitivity
analysis; writes scores_main/<kind>_seed_<s>.npz.
Usage: python code/robustness/fixed_budget_score_gift_main.py --kind descriptor_only --seeds 47 ... [--check]
"""
import os as _os, sys as _sys  # release bootstrap: run from any directory as python code/robustness/<script>.py
_REPO = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", ".."))
_sys.path[:0] = [_os.path.join(_REPO, "code"), _os.path.join(_REPO, "code", "robustness")]
_os.chdir(_REPO)
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fixed_budget_score_gift as sg  # noqa: E402

sg.run_dir = lambda kind, s: (f"results/gift/run_{s}" if kind == "run"
                              else f"results/gift/descriptor_only_seed_{s}")
sg.OUT = os.path.join(sg.REPO, "results", "robustness", "descriptor_fixed_budget", "scores_main")

if __name__ == "__main__":
    sg.main()
