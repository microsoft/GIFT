"""Clean graph protocol: the message-passing graph contains only TRAINING compound-protein pairs.

Splits are always derived from the original graph (both split functions read cp_edge_index),
then every held-out (val + test) CP pair is removed from edge_index, edge_attr, edge_types,
cp_edge_index and cp_edge_attr before any model sees the graph.
"""

import contextlib
import io

import numpy as np

from model_encoder import split_interaction_data


def trainer_split(data, random_state=42):
    """The train/val/test CP split (default random_state=42) used by every trainer and the evaluator."""
    with contextlib.redirect_stdout(io.StringIO()):
        tr, va, te = split_interaction_data(data, test_ratio=0.15, val_ratio=0.15, random_state=random_state)
    as_pairs = lambda ps: [(int(c), int(p)) for c, p in ps]
    return as_pairs(tr), as_pairs(va), as_pairs(te)


def remove_cp_pairs(graph, pairs):
    """Copy of `graph` with the CP edges of `pairs` (both directions) removed from every edge list."""
    nt = np.asarray(graph["node_types"])
    comp_nodes, prot_nodes = np.where(nt == 0)[0], np.where(nt == 1)[0]
    n = len(nt)
    pc = np.array([c for c, _ in pairs], dtype=np.int64)
    pp = np.array([p for _, p in pairs], dtype=np.int64)
    cn, pn = comp_nodes[pc], prot_nodes[pp]
    drop = np.concatenate([cn * n + pn, pn * n + cn])

    out = dict(graph)
    for ei_key, extra_keys in [("edge_index", ["edge_attr", "edge_types"]), ("cp_edge_index", ["cp_edge_attr"])]:
        ei = np.asarray(graph[ei_key])
        keep = ~np.isin(ei[0].astype(np.int64) * n + ei[1].astype(np.int64), drop)
        out[ei_key] = ei[:, keep]
        for k in extra_keys:
            v = graph.get(k)
            if v is not None and np.asarray(v).shape[0] == ei.shape[1]:
                out[k] = np.asarray(v)[keep]
    out["num_cp_edges"] = int(out["cp_edge_index"].shape[1])
    return out


def clean_graph(data):
    """Return (clean_graph, train_pairs, val_pairs, test_pairs) with only training CP edges kept."""
    tr, va, te = trainer_split(data)
    clean = remove_cp_pairs(data, va + te)
    return clean, tr, va, te


if __name__ == "__main__":
    import pickle

    d = pickle.load(open("data/graphs/graph_similarity_input.pkl", "rb"))
    g, tr, va, te = clean_graph(d)
    nt = np.asarray(d["node_types"])
    ei = np.asarray(g["edge_index"])
    cp_left = int((nt[ei[0]] != nt[ei[1]]).sum())
    print(f"train/val/test pairs: {len(tr)}/{len(va)}/{len(te)}")
    print(f"edge_index: {d['edge_index'].shape[1]} -> {ei.shape[1]}; CP edges left {cp_left} (expected {2 * len(set(tr))})")
    print(f"cp_edge_index: {d['cp_edge_index'].shape[1]} -> {g['cp_edge_index'].shape[1]}")
    a, b, c = trainer_split(g)
    print("note: split on the clean graph would differ (sizes)", len(a), len(b), len(c), "- always split on the original")
