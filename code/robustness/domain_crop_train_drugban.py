"""DrugBAN domain-crop training with the local PyTorch/PyG graph backend.

Per run (see run_drugban_dgl_queue.sh):
  1. main env:  train_dta_baselines.py --export_protocol P.pkl   (split, negatives rule, validation set,
                benchmarks; identical to DeepDTA/GraphDTA)
  2. main env:  this script, P.pkl -> scores .npz (DrugBAN architecture, features, loss)
  3. main env:  train_dta_baselines.py --import_scores S.npz     (identical metrics and output format)

Training follows the official trainer (vanilla DrugBAN, configs/DrugBAN.yaml: Adam, learning rate 5e-5, batch 64,
sigmoid + BCELoss, DTIDataset featurization with 290-node padding, 1,200-residue protein encoding) within the shared
protocol: one training negative per positive resampled every epoch with the same sampler and seed as the other
baselines, checkpoint selection on validation ROC-AUC, at most 300 epochs with patience 20. External compounds are
featurized only for inference. The run checkpoints every epoch and resumes after an interruption.
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
import time

import numpy as np
import torch

# Copy for review point 6: only the repository path changes (the copy lives in results/robustness/domain_crop/).
# Domain crops reach this script through the protocol's protein_sequences (already <= 1,200 residues).
REPO = _REPO
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "code", "third_party", "DrugBAN_9923f8c"))
import pandas as pd  # noqa: E402
from configs import get_cfg_defaults  # noqa: E402
from _drugban_graph import DTIDataset, batch_graphs, integer_label_protein  # noqa: E402
from models import DrugBAN, binary_cross_entropy  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

from negative_pool import sample_negatives  # noqa: E402

IMPLEMENTATION = "DrugBAN (github.com/peizhenbai/DrugBAN, commit 9923f8c), local PyTorch/PyG backend"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocol", required=True)
    ap.add_argument("--scores_out", required=True)
    ap.add_argument("--checkpoint", default="", help="per-epoch checkpoint for resuming (default: scores_out + .ckpt)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--eval_batch", type=int, default=256)
    ap.add_argument("--lead_out", default="", help="also write driver-kinase lead-prio rankings (needs lead fields in protocol)")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    dev = torch.device(args.device)
    t0 = time.time()
    pr = pickle.load(open(args.protocol, "rb"))
    seed, nc, npr, train = pr["seed"], pr["nc"], pr["npr"], pr["train"]
    cfg = get_cfg_defaults()
    cfg.merge_from_file(os.path.join(REPO, "code", "third_party", "DrugBAN_9923f8c", "DrugBAN.yaml"))
    bs, lr = cfg.SOLVER.BATCH_SIZE, cfg.SOLVER.LR

    # Official featurization (DTIDataset.__getitem__), computed once per compound and protein.
    feat = DTIDataset(list(range(len(pr["smiles"]))),
                      pd.DataFrame({"SMILES": pr["smiles"], "Protein": "A", "Y": 0}), cfg.DRUG.MAX_NODES)
    G = [feat[i][0] for i in range(len(pr["smiles"]))]
    P = torch.tensor(np.array([integer_label_protein(s) for s in pr["protein_sequences"]]))

    def batch(pairs):
        d = batch_graphs([G[c] for c, _ in pairs]).to(dev)
        return d, P[[p for _, p in pairs]].to(dev)

    def predict(model, pairs):
        model.eval()
        out = []
        with torch.no_grad():
            for i in range(0, len(pairs), args.eval_batch):
                _, _, _, score = model(*batch(pairs[i:i + args.eval_batch]))
                out.append(torch.sigmoid(score).squeeze(1).cpu().numpy())
        return np.concatenate(out)

    torch.manual_seed(seed)
    rng = np.random.RandomState(seed)
    model = DrugBAN(**cfg).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    st = {"epoch": 0, "best": -1.0, "wait": 0, "best_epoch": 0, "stop": pr["epochs"], "best_state": None}
    ckpt = args.checkpoint or args.scores_out + ".ckpt"
    if os.path.exists(ckpt):
        c = torch.load(ckpt, weights_only=False, map_location=dev)
        model.load_state_dict(c["model"])
        opt.load_state_dict(c["opt"])
        rng.set_state(c["rng"])
        torch.set_rng_state(c["torch_rng"].cpu())
        st = c["st"]
        print(f"resumed after epoch {st['epoch']}", flush=True)
    done = st["wait"] >= pr["patience"]
    while not done and st["epoch"] < pr["epochs"]:
        epoch = st["epoch"] + 1
        negs = sample_negatives(rng, np.arange(nc), npr, len(train), pr["train_exclude"])
        pairs = train + negs
        y = np.array([1.0] * len(train) + [0.0] * len(negs), dtype=np.float32)
        order = rng.permutation(len(pairs))
        model.train()
        te = time.time()
        for i in range(0, len(order), bs):
            ix = order[i:i + bs]
            if len(ix) < 2:
                continue
            opt.zero_grad()
            _, _, _, score = model(*batch([pairs[j] for j in ix]))
            _, loss = binary_cross_entropy(score, torch.tensor(y[ix], device=dev))
            loss.backward()
            opt.step()
        auc = roc_auc_score(pr["y_val"], predict(model, pr["val_all"]))
        st["epoch"] = epoch
        if auc > st["best"]:
            st.update(best=auc, wait=0, best_epoch=epoch,
                      best_state={k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
        else:
            st["wait"] += 1
            if st["wait"] >= pr["patience"]:
                st["stop"], done = epoch, True
        print(f"seed {seed} epoch {epoch} val_auc={auc:.4f} best={st['best']:.4f}@{st['best_epoch']} "
              f"{time.time() - te:.0f}s", flush=True)
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "rng": rng.get_state(),
                    "torch_rng": torch.get_rng_state(), "st": st}, ckpt + ".tmp")
        os.replace(ckpt + ".tmp", ckpt)

    model.load_state_dict(st["best_state"])
    if not (args.lead_out and "lead_drivers" in pr):
        out = {}
        for name, (pairs, y) in pr["benches"].items():
            out[f"{name}|score"] = predict(model, pairs)
            out[f"{name}|pairs"] = np.asarray(pairs)
            out[f"{name}|y"] = y
        meta = {"kind": "drugban", "implementation": IMPLEMENTATION, "val_roc_auc": float(st["best"]),
                "config": {"lr": lr, "batch": bs, "max_drug_nodes": cfg.DRUG.MAX_NODES, "ban_heads": cfg.BCN.HEADS},
                "stopped_at_epoch": st["stop"], "best_epoch": st["best_epoch"], "seconds": time.time() - t0,
                "device": str(dev)}
        np.savez_compressed(args.scores_out, meta=json.dumps(meta), **out)
        print(f"seed {seed} drugban val_auc={st['best']:.4f} best_epoch={st['best_epoch']} stop={st['stop']}",
              flush=True)

    if args.lead_out and "lead_drivers" in pr:
        # Driver-kinase lead prioritization: rank the external drugs for each driver and count known actives
        # at rank one and in the top 10, identical metric to train_dta_baselines.py --lead_out.
        ext_drug_idx = pr["ext_drug_idx"]
        driver_actives = {int(p): set(v) for p, v in pr["driver_actives"].items()}
        dr = {}
        for p_d, name in pr["lead_drivers"]:
            s = predict(model, [(d, p_d) for d in ext_drug_idx])
            order = np.argsort(-s, kind="stable")
            acts = driver_actives.get(int(p_d), set())
            top10 = [ext_drug_idx[order[j]] for j in range(min(10, len(order)))]
            dr[name] = {"active_at_1": int(ext_drug_idx[order[0]] in acts),
                        "hits_top10": int(sum(d in acts for d in top10)),
                        "n_actives": len(acts & set(ext_drug_idx))}
        lead = {"per_driver": dr, "drivers_active_at_1": int(sum(v["active_at_1"] for v in dr.values())),
                "hits_top10_sum": int(sum(v["hits_top10"] for v in dr.values())), "n_drivers": len(dr)}
        json.dump({"seed": seed, "lead": {"drugban": lead}}, open(args.lead_out, "w"), indent=2)
        print(f"seed {seed} drugban lead: top1={lead['drivers_active_at_1']} top10={lead['hits_top10_sum']}",
              flush=True)


if __name__ == "__main__":
    main()
