"""Point 16: build the main-model graph with the protein descriptor block replaced by ESM-2 650M embeddings.

Takes data/graphs/graph_seed{S}.pkl and swaps its 453-dim protein feature block for the 1,280-dim ESM-2
mean embedding (standardized across proteins), leaving the 435-dim compound block, every edge, label, split, and the
external compound features unchanged. Writes data/graphs/esm/graph_esm_seed{S}.pkl so the unchanged main model pipeline
(train_backbone -> train_classifier/e2e -> evaluate_chembl35 / evaluate_external) runs on it by --graph_path alone.

Usage: python code/robustness/esm_build_graphs.py --seeds 42 43 ...
"""
import os as _os, sys as _sys  # release bootstrap: run from any directory as python code/robustness/<script>.py
_REPO = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", ".."))
_sys.path[:0] = [_os.path.join(_REPO, "code"), _os.path.join(_REPO, "code", "robustness")]
_os.chdir(_REPO)
import argparse
import pickle

import numpy as np

ESM = "data/esm2_650m_embeddings.npz"
CDIM = 435  # compound block width (unchanged)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    a = ap.parse_args()
    e = np.load(ESM, allow_pickle=True)
    emb = {i: v for i, v in zip(e["ids"].tolist(), e["embeddings"])}
    edim = e["embeddings"].shape[1]
    for s in a.seeds:
        d = pickle.load(open(f"data/graphs/graph_seed{s}.pkl", "rb"))
        nf = np.asarray(d["node_features"], dtype=np.float32)
        nc, npr = int(d["num_compounds"]), int(d["num_proteins"])
        node_types = np.asarray(d["node_types"])
        pids = list(d["protein_ids"])
        assert all(p in emb for p in pids), f"missing ESM for {sum(p not in emb for p in pids)} proteins"
        # Standardize ESM embeddings across the graph proteins (z-score per dimension).
        P = np.stack([emb[p] for p in pids]).astype(np.float32)
        mean, std = P.mean(0), P.std(0)
        std[std < 1e-6] = 1.0
        Pz = (P - mean) / std
        # New block-diagonal node_features: compound block [0:CDIM] kept, protein block replaced by ESM.
        new = np.zeros((nf.shape[0], CDIM + edim), dtype=np.float32)
        comp_mask = node_types == 0
        prot_mask = node_types == 1
        new[comp_mask, :CDIM] = nf[comp_mask][:, :CDIM]
        # protein rows are in graph order (first npr protein-type nodes map to protein_ids order)
        prot_rows = np.where(prot_mask)[0]
        assert len(prot_rows) == npr
        new[prot_rows, CDIM:] = Pz
        d["node_features"] = new
        d["feature_layout"] = {"compound": [0, CDIM], "protein": [CDIM, CDIM + edim]}
        d["protein_scaler_mean"] = mean.astype(np.float32)
        d["protein_scaler_scale"] = std.astype(np.float32)
        d["protein_raw_columns"] = [f"esm2_650m_{k}" for k in range(edim)]
        d["protein_metadata_source"] = "base compound block (435) + ESM-2 650M mean protein embedding (1280), z-scored"
        out = f"data/graphs/esm/graph_esm_seed{s}.pkl"
        pickle.dump(d, open(out, "wb"))
        print(f"seed {s}: node_features {new.shape}, protein block {CDIM}:{CDIM + edim} -> {out}", flush=True)


if __name__ == "__main__":
    main()
