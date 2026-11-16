"""ESM-2 (650M) mean-pooled embeddings for the 1,391 base proteins, on GPU.

Sequences longer than the window are split into overlapping windows; per-residue embeddings are averaged over the
whole sequence. Reads data/protein_sequences.json (id -> sequence string, aligned to the base graph) and
writes data/esm2_650m_embeddings.npz. Run with an environment with the fair-esm package (pip install fair-esm).
"""
import os as _os, sys as _sys  # release bootstrap: run from any directory as python code/robustness/<script>.py
_REPO = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", ".."))
_sys.path[:0] = [_os.path.join(_REPO, "code"), _os.path.join(_REPO, "code", "robustness")]
_os.chdir(_REPO)
import json
import time

import esm
import numpy as np
import torch

SEQ_PATH = "data/protein_sequences.json"
OUT = "data/esm2_650m_embeddings.npz"
WINDOW, STRIDE = 1000, 500


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seqs = json.load(open(SEQ_PATH))
    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    model.eval().to(device)
    converter = alphabet.get_batch_converter()
    ids, embs = [], []
    t0 = time.time()
    items = list(seqs.items())
    for i, (target, seq) in enumerate(items):
        if not seq:
            continue
        starts = list(range(0, max(len(seq) - WINDOW, 0) + 1, STRIDE))
        if starts[-1] + WINDOW < len(seq):
            starts.append(len(seq) - WINDOW)
        total, count = torch.zeros(1280), 0
        with torch.no_grad():
            for s in starts:
                chunk = seq[s:s + WINDOW]
                _, _, tokens = converter([(target, chunk)])
                rep = model(tokens.to(device), repr_layers=[33])["representations"][33][0, 1:len(chunk) + 1].float().cpu()
                total += rep.sum(0)
                count += rep.shape[0]
        ids.append(target)
        embs.append((total / count).numpy())
        if (i + 1) % 100 == 0:
            print(f"{i + 1}/{len(items)} proteins, {time.time() - t0:.0f}s", flush=True)
    np.savez(OUT, ids=np.array(ids), embeddings=np.stack(embs).astype(np.float32))
    print(f"saved {len(ids)} embeddings -> {OUT} in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
