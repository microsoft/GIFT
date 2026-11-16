"""DeepDTA, GraphDTA (GIN), and DrugBAN binary DTI baselines under the shared baseline protocol.

DrugBAN (Bai et al., Nat. Mach. Intell. 2023) is reimplemented in PyTorch/PyG from the official repository
(github.com/peizhenbai/DrugBAN, models.py/ban.py/dataloader.py/configs.py) without its domain adaptation module
(the vanilla DrugBAN of configs/DrugBAN.yaml): dgllife CanonicalAtomFeaturizer (74) plus a virtual-node bit,
graphs padded to 290 nodes with self loops, 3-layer dgllife GCN (128), 3-layer protein CNN (128 filters, kernels
3/6/9, sequences of 1,200 residues), bilinear attention network with 2 heads, MLP decoder 256-512-128-1,
Adam with learning rate 5e-5 and batch size 64. drugban_equivalence/ verifies, with identical weights on 512 real
pairs, that it reproduces the official DGL implementation (outputs, loss, and gradients to float32 precision).

Both models are trained from scratch on the run's ChEMBL35 training interactions only, with the data protocol
shared by every method in the manuscript:
- the per-run graph split (split_override) and its training positives;
- one training negative per positive, resampled every epoch, excluding training positives and the evaluation reserve;
- the validation set of the proposed model's checkpoint selection (validation ROC-AUC, early stopping);
- the identical ChEMBL35 random-negative and compound-matched benchmarks and the external ChEMBL36 benchmark
  with compound-matched negatives, scored at the fixed threshold of 0.5.
The external compounds are not graph compounds, so they never occur in training, validation, or model selection;
they are encoded only at inference. Input encodings are fixed (character vocabularies, RDKit atom features), so no
preprocessing is fitted on any data. Each model uses one standard published configuration, without hyperparameter
search, like every other method in the manuscript.
"""
import os as _os, sys as _sys  # release bootstrap: run from any directory as python code/robustness/<script>.py
_REPO = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", ".."))
_sys.path[:0] = [_os.path.join(_REPO, "code"), _os.path.join(_REPO, "code", "robustness")]
_os.chdir(_REPO)
import argparse
import json
import os
import pickle
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from rdkit import Chem, RDLogger
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix, f1_score,
                             matthews_corrcoef, precision_score, recall_score, roc_auc_score)
from torch_geometric.data import Batch, Data
from torch_geometric.nn import GINConv, global_add_pool

import pipeline_utils as vs
from evaluate_chembl35 import gnn_validation_negatives
from evaluate_external import external_matched_negatives, per_target_roc_auc
from negative_pool import (dti_random_negatives, evaluation_exclusion, reserved_eval_mask,
                               sample_compound_matched_negatives, sample_negatives, training_exclusion)

RDLogger.DisableLog("rdApp.*")

# DeepDTA vocabularies (Ozturk et al., 2018).
CHARISOSMISET = {"#": 29, "%": 30, ")": 31, "(": 1, "+": 32, "-": 33, "/": 34, ".": 2, "1": 35, "0": 3, "3": 36,
                 "2": 4, "5": 37, "4": 5, "7": 38, "6": 6, "9": 39, "8": 7, "=": 40, "A": 41, "@": 8, "C": 42,
                 "B": 9, "E": 43, "D": 10, "G": 44, "F": 11, "I": 45, "H": 12, "K": 46, "M": 47, "L": 13, "O": 48,
                 "N": 14, "P": 15, "S": 49, "R": 16, "U": 50, "T": 17, "W": 51, "V": 18, "Y": 52, "[": 53, "Z": 19,
                 "]": 54, "\\": 20, "a": 55, "c": 56, "b": 21, "e": 57, "d": 22, "g": 58, "f": 23, "i": 59, "h": 24,
                 "m": 60, "l": 25, "o": 61, "n": 26, "s": 62, "r": 27, "u": 63, "t": 28, "y": 64}
CHARPROTSET = {c: i + 1 for i, c in enumerate("ACBEDGFIHKMLONQPSRUTWVYXZ")}
MAX_SMI, MAX_SEQ = 100, 1000
# GraphDTA atom features (Nguyen et al., 2021).
SYMBOLS = ["C", "N", "O", "S", "F", "Si", "P", "Cl", "Br", "Mg", "Na", "Ca", "Fe", "As", "Al", "I", "B", "V", "K",
           "Tl", "Yb", "Sb", "Sn", "Ag", "Pd", "Co", "Se", "Ti", "Zn", "H", "Li", "Ge", "Cu", "Au", "Ni", "Cd", "In",
           "Mn", "Zr", "Cr", "Pt", "Hg", "Pb", "Unknown"]
# Standard configurations: DeepDTA CNNs as in DeepPurpose (filters 32/64/96, kernels 4/6/8 and 4/8/12) with
# DeepDTA's learning rate and dropout; GraphDTA GINConvNet defaults of the official repository.
CONFIG = {"deepdta": {"smi_kernel": [4, 6, 8], "seq_kernel": [4, 8, 12], "lr": 1e-3, "dropout": 0.1},
          "graphdta": {"lr": 5e-4, "dropout": 0.2},
          "drugban": {"lr": 5e-5}}
BATCH = {"deepdta": 256, "graphdta": 512, "drugban": 64}
EVAL_BATCH = {"deepdta": 2048, "graphdta": 2048, "drugban": 256}
DRUGBAN_MAX_NODES, DRUGBAN_MAX_SEQ = 290, 1200
# dgllife CanonicalAtomFeaturizer allowable sets (atom_type_one_hot without the unknown bit).
CANON_SYMBOLS = SYMBOLS[:-1]
CANON_HYB = [Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2, Chem.rdchem.HybridizationType.SP3,
             Chem.rdchem.HybridizationType.SP3D, Chem.rdchem.HybridizationType.SP3D2]


def label_seq(s, vocab, n):
    out = np.zeros(n, dtype=np.int64)
    for i, ch in enumerate(s[:n]):
        out[i] = vocab.get(ch, 0)
    return out


def onehot(x, allowed):
    x = x if x in allowed else allowed[-1]
    return [x == a for a in allowed]


def mol_graph(smiles):
    mol = Chem.MolFromSmiles(smiles)
    feats = []
    for a in mol.GetAtoms():
        f = np.array(onehot(a.GetSymbol(), SYMBOLS) + onehot(a.GetDegree(), list(range(11)))
                     + onehot(a.GetTotalNumHs(), list(range(11))) + onehot(a.GetImplicitValence(), list(range(11)))
                     + [a.GetIsAromatic()], dtype=np.float32)
        feats.append(f / f.sum())
    edges = [[b.GetBeginAtomIdx(), b.GetEndAtomIdx()] for b in mol.GetBonds()]
    edges = edges + [[j, i] for i, j in edges]
    ei = torch.tensor(edges, dtype=torch.long).t() if edges else torch.zeros((2, 0), dtype=torch.long)
    return Data(x=torch.tensor(np.stack(feats)), edge_index=ei)


def drugban_graph(smiles):
    """DrugBAN drug input: dgllife bigraph with self loops, 74 canonical atom features plus a virtual-node bit,
    padded with virtual nodes to 290 nodes, after which a self loop is added to every node (dataloader.py)."""
    mol = Chem.MolFromSmiles(smiles)
    feats = []
    for a in mol.GetAtoms():
        feats.append([a.GetSymbol() == s for s in CANON_SYMBOLS] + onehot_strict(a.GetDegree(), range(11))
                     + onehot_strict(a.GetImplicitValence(), range(7)) + [a.GetFormalCharge(),
                     a.GetNumRadicalElectrons()] + onehot_strict(a.GetHybridization(), CANON_HYB)
                     + [a.GetIsAromatic()] + onehot_strict(a.GetTotalNumHs(), range(5)) + [0])
    n = len(feats)
    assert n <= DRUGBAN_MAX_NODES
    x = np.zeros((DRUGBAN_MAX_NODES, 75), dtype=np.float32)
    x[:n] = np.array(feats, dtype=np.float32)
    x[n:, -1] = 1.0
    edges = [[b.GetBeginAtomIdx(), b.GetEndAtomIdx()] for b in mol.GetBonds()]
    edges = edges + [[j, i] for i, j in edges] + [[i, i] for i in range(n)] + [[i, i] for i in
                                                                              range(DRUGBAN_MAX_NODES)]
    return Data(x=torch.tensor(x), edge_index=torch.tensor(edges, dtype=torch.long).t())


def onehot_strict(x, allowed):
    return [x == a for a in allowed]


class DeepDTA(nn.Module):
    def __init__(self, smi_kernel, seq_kernel, dropout, nf=32, emb=128):
        super().__init__()
        self.se, self.pe = nn.Embedding(65, emb), nn.Embedding(26, emb)
        smi_k = smi_kernel if isinstance(smi_kernel, (list, tuple)) else [smi_kernel] * 3
        seq_k = seq_kernel if isinstance(seq_kernel, (list, tuple)) else [seq_kernel] * 3

        def cnn(k):
            return nn.Sequential(nn.Conv1d(emb, nf, k[0]), nn.ReLU(), nn.Conv1d(nf, 2 * nf, k[1]), nn.ReLU(),
                                 nn.Conv1d(2 * nf, 3 * nf, k[2]), nn.ReLU(), nn.AdaptiveMaxPool1d(1))
        self.sc, self.pc = cnn(smi_k), cnn(seq_k)
        self.head = nn.Sequential(nn.Linear(6 * nf, 1024), nn.ReLU(), nn.Dropout(dropout), nn.Linear(1024, 1024),
                                  nn.ReLU(), nn.Dropout(dropout), nn.Linear(1024, 512), nn.ReLU(), nn.Linear(512, 1))

    def forward(self, smi, seq):
        d = self.sc(self.se(smi).transpose(1, 2)).squeeze(-1)
        p = self.pc(self.pe(seq).transpose(1, 2)).squeeze(-1)
        return self.head(torch.cat([d, p], 1)).squeeze(-1)


class GraphDTA(nn.Module):
    """GINConvNet of GraphDTA with a single-logit classification output."""

    def __init__(self, dropout, nf=32, emb=128, out=128):
        super().__init__()
        self.convs, self.bns = nn.ModuleList(), nn.ModuleList()
        for i in range(5):
            mlp = nn.Sequential(nn.Linear(78 if i == 0 else nf, nf), nn.ReLU(), nn.Linear(nf, nf))
            self.convs.append(GINConv(mlp))
            self.bns.append(nn.BatchNorm1d(nf))
        self.fc_xd = nn.Linear(nf, out)
        self.pe = nn.Embedding(26, emb)
        self.pc = nn.Conv1d(MAX_SEQ, nf, 8)
        self.fc_xt = nn.Linear(nf * (emb - 7), out)
        self.fc1, self.fc2, self.out = nn.Linear(2 * out, 1024), nn.Linear(1024, 256), nn.Linear(256, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, g, seq):
        x = g.x
        for conv, bn in zip(self.convs, self.bns):
            x = bn(F.relu(conv(x, g.edge_index)))
        x = self.dropout(F.relu(self.fc_xd(global_add_pool(x, g.batch))))
        t = self.fc_xt(self.pc(self.pe(seq)).flatten(1))
        h = self.dropout(F.relu(self.fc1(torch.cat([x, t], 1))))
        h = self.dropout(F.relu(self.fc2(h)))
        return self.out(h).squeeze(-1)


class DGLGraphConvNone(nn.Module):
    """dgl.nn.GraphConv with norm='none': sum of in-neighbour features, linear map, bias, activation."""

    def __init__(self, i, o):
        super().__init__()
        self.weight, self.bias = nn.Parameter(torch.empty(i, o)), nn.Parameter(torch.zeros(o))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x, edge_index):
        agg = torch.zeros_like(x).index_add_(0, edge_index[1], x[edge_index[0]])
        return F.relu(agg @ self.weight + self.bias)


class DGLLifeGCNLayer(nn.Module):
    """dgllife GCNLayer defaults: ReLU, residual (activated linear), no dropout, batch norm."""

    def __init__(self, i, o):
        super().__init__()
        self.conv, self.res, self.bn = DGLGraphConvNone(i, o), nn.Linear(i, o), nn.BatchNorm1d(o)

    def forward(self, x, edge_index):
        return self.bn(self.conv(x, edge_index) + F.relu(self.res(x)))


class BANLayer(nn.Module):
    """ban.py BANLayer (h_out <= 32 branch) with weight-normalised FCNets and h_mat."""

    def __init__(self, v_dim, q_dim, h_dim, h_out, dropout=0.2, k=3):
        super().__init__()
        wn = nn.utils.weight_norm
        self.k, self.h_out = k, h_out
        self.v_net = nn.Sequential(nn.Dropout(dropout), wn(nn.Linear(v_dim, h_dim * k), dim=None), nn.ReLU())
        self.q_net = nn.Sequential(nn.Dropout(dropout), wn(nn.Linear(q_dim, h_dim * k), dim=None), nn.ReLU())
        self.p_net = nn.AvgPool1d(k, stride=k)
        self.h_mat = nn.Parameter(torch.Tensor(1, h_out, 1, h_dim * k).normal_())
        self.h_bias = nn.Parameter(torch.Tensor(1, h_out, 1, 1).normal_())
        self.bn = nn.BatchNorm1d(h_dim)

    def pool(self, v, q, att):
        f = torch.einsum("bvk,bvq,bqk->bk", (v, att, q))
        return self.p_net(f.unsqueeze(1)).squeeze(1) * self.k

    def forward(self, v, q):
        v_, q_ = self.v_net(v), self.q_net(q)
        att = torch.einsum("xhyk,bvk,bqk->bhvq", (self.h_mat, v_, q_)) + self.h_bias
        logits = self.pool(v_, q_, att[:, 0])
        for i in range(1, self.h_out):
            logits = logits + self.pool(v_, q_, att[:, i])
        return self.bn(logits)


class DrugBAN(nn.Module):
    """Vanilla DrugBAN (models.py) with its default configuration; returns one logit per pair."""

    def __init__(self, emb=128, hid=128, filters=128, kernels=(3, 6, 9), ban_dim=256, mlp_hid=512, mlp_out=128,
                 heads=2):
        super().__init__()
        self.init_transform = nn.Linear(75, emb, bias=False)
        with torch.no_grad():
            self.init_transform.weight[-1].fill_(0)
        self.gcn = nn.ModuleList([DGLLifeGCNLayer(emb if i == 0 else hid, hid) for i in range(3)])
        self.pe = nn.Embedding(26, emb, padding_idx=0)
        ch = [emb, filters, filters, filters]
        self.convs = nn.ModuleList([nn.Conv1d(ch[i], ch[i + 1], kernels[i]) for i in range(3)])
        self.cbns = nn.ModuleList([nn.BatchNorm1d(ch[i + 1]) for i in range(3)])
        self.bcn = nn.utils.weight_norm(BANLayer(hid, filters, ban_dim, heads), name="h_mat", dim=None)
        self.fc1, self.bn1 = nn.Linear(ban_dim, mlp_hid), nn.BatchNorm1d(mlp_hid)
        self.fc2, self.bn2 = nn.Linear(mlp_hid, mlp_hid), nn.BatchNorm1d(mlp_hid)
        self.fc3, self.bn3 = nn.Linear(mlp_hid, mlp_out), nn.BatchNorm1d(mlp_out)
        self.fc4 = nn.Linear(mlp_out, 1)
        self.hid = hid

    def forward(self, g, seq):
        x = self.init_transform(g.x)
        for layer in self.gcn:
            x = layer(x, g.edge_index)
        v_d = x.view(g.num_graphs, -1, self.hid)
        v = self.pe(seq).transpose(2, 1)
        for conv, bn in zip(self.convs, self.cbns):
            v = bn(F.relu(conv(v)))
        # The official ProteinCNN reshapes with view (not transpose); kept for fidelity.
        v_p = v.view(v.size(0), v.size(2), -1)
        f = self.bcn(v_d, v_p)
        f = self.bn1(F.relu(self.fc1(f)))
        f = self.bn2(F.relu(self.fc2(f)))
        f = self.bn3(F.relu(self.fc3(f)))
        return self.fc4(f).squeeze(-1)


def build_model(kind, cfg):
    if kind == "deepdta":
        return DeepDTA(cfg["smi_kernel"], cfg["seq_kernel"], cfg["dropout"])
    if kind == "graphdta":
        return GraphDTA(cfg["dropout"])
    return DrugBAN()


def metrics(pairs, y, s):
    pred = (s >= 0.5).astype(int)
    pt, n = per_target_roc_auc(pairs, y, s)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {"roc_auc": float(roc_auc_score(y, s)), "auprc": float(average_precision_score(y, s)),
            "per_target_roc_auc": pt, "per_target_n": n, "accuracy": float(accuracy_score(y, pred)),
            "precision": float(precision_score(y, pred, zero_division=0)),
            "recall": float(recall_score(y, pred, zero_division=0)), "f1": float(f1_score(y, pred, zero_division=0)),
            "mcc": float(matthews_corrcoef(y, pred)), "tp": int(tp), "fn": int(fn), "fp": int(fp), "tn": int(tn)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--graph_path", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--models", default="deepdta,graphdta")
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--scores_out", default="", help="optional .npz with the per-pair scores of every benchmark")
    ap.add_argument("--export_protocol", default="", help="write the run's protocol (.pkl) for an external trainer and exit")
    ap.add_argument("--import_scores", default="", help="evaluate benchmark scores (.npz) of an external trainer")
    ap.add_argument("--lead_out", default="", help="also score the 10 driver kinases' lead-prioritization rankings")
    ap.add_argument("--save_models", default="", help="directory to save the trained model state_dicts (reproducibility)")
    ap.add_argument("--domain_crop", default="", help="domain_crops.json; empty = original N-terminal prefix")
    args = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t0 = time.time()

    data = pickle.load(open(args.graph_path, "rb"))
    nc, npr = int(data["num_compounds"]), int(data["num_proteins"])
    splits = vs.derive_chembl35_splits(data, min_training_pchembl=5.0)
    train, val_pairs, test_pairs = splits["train_pairs"], splits["val_pairs"], splits["test_pairs"]
    reserved = reserved_eval_mask(nc, npr)
    eval_exclude = evaluation_exclusion(splits["all_positive_pairs"], reserved)
    # Benchmarks exactly as in train_gnn_baselines.py / evaluate_chembl35.py.
    neg_rng = np.random.RandomState(vs.SEED)
    sample_negatives(neg_rng, np.arange(nc), npr, 5 * len(test_pairs), eval_exclude)
    matched_negs = sample_compound_matched_negatives(neg_rng, test_pairs, npr, 5, eval_exclude)
    random_negs = dti_random_negatives(vs.SEED, train + val_pairs + test_pairs, npr, 5 * len(test_pairs), eval_exclude)
    val_negs = gnn_validation_negatives(args.seed, val_pairs, splits["all_positive_pairs"], nc, npr)
    val_all, y_val = val_pairs + val_negs, np.array([1] * len(val_pairs) + [0] * len(val_negs))
    aug, cross_pairs, _ = vs.load_external_chembl36_interactions_from_pkls(data)
    ext_negs = external_matched_negatives(cross_pairs, npr, tested_pairs=aug.get("external_tested_pairs", ()))
    benches = {
        "same_version": (test_pairs + random_negs, np.array([1] * len(test_pairs) + [0] * len(random_negs))),
        "same_version_matched": (test_pairs + matched_negs, np.array([1] * len(test_pairs) + [0] * len(matched_negs))),
        "external": (cross_pairs + ext_negs, np.array([1] * len(cross_pairs) + [0] * len(ext_negs))),
    }

    # Inputs: compound index < nc are ChEMBL35 graph compounds, >= nc the external compounds (inference only).
    smi_csv = pd.read_csv("data/chembl_lung_cancer_all_compounds.csv").dropna(subset=["smiles"])
    smi_csv = smi_csv.drop_duplicates("compound_id").set_index("compound_id")["smiles"]
    ext_records = pickle.load(open(aug["external_interactions_pkl"], "rb"))["external_compounds"]
    ext_ids = list(aug["external_compound_ids"])
    graph_ids = list(data["compound_ids"])[:nc]
    assert not set(ext_ids) & set(graph_ids), "external compounds must be absent from the ChEMBL35 graph"
    smiles = [smi_csv[c] for c in graph_ids] + [ext_records[c]["smiles"] for c in ext_ids]
    for name, pairs in [("train", train), ("val", val_all)]:
        assert max(c for c, _ in pairs) < nc, f"external compound in {name}"
    assert min(c for c, _ in cross_pairs + ext_negs) >= nc

    # Lead-prioritization driver set (Ashley area 2): the 10 NSCLC driver kinases with >= 5 external actives,
    # ranked over the same 95 external drugs as the proposed model. Labels come from the ChEMBL36 actives (cross_pairs).
    DRIVERS = {"CHEMBL203": "EGFR", "CHEMBL1824": "ERBB2", "CHEMBL4247": "ALK", "CHEMBL3717": "MET",
               "CHEMBL2041": "RET", "CHEMBL5568": "ROS1", "CHEMBL5145": "BRAF",
               "CHEMBL2815": "NTRK1", "CHEMBL4898": "NTRK2", "CHEMBL5608": "NTRK3"}
    protein_ids = list(data["protein_ids"])[:npr]
    pindex = {t: i for i, t in enumerate(protein_ids)}
    ext_drug_idx = [nc + i for i in range(len(ext_ids))]           # the 95 external drug node indices
    driver_actives = {}                                            # protein index -> set of active drug indices
    for c, p in cross_pairs:
        driver_actives.setdefault(p, set()).add(c)
    lead_drivers = [(pindex[t], name) for t, name in DRIVERS.items()
                    if t in pindex and len(driver_actives.get(pindex[t], set())) >= 5]
    lead_store = {}
    canon = [Chem.MolToSmiles(Chem.MolFromSmiles(s)) for s in smiles]
    train_canon = {canon[c] for c, _ in train}
    ext_same_structure = sum(canon[nc + i] in train_canon for i in range(len(ext_ids)))

    seqs = json.load(open("data/protein_sequences.json"))
    # Review point 6: domain-preserving crops (build_domain_crops.py). Proteins longer than the model's limit get the
    # precomputed window of exactly MAX_SEQ (DeepDTA/GraphDTA) or DRUGBAN_MAX_SEQ (DrugBAN) residues instead of the
    # N-terminal prefix; without --domain_crop the sequences are the original ones.
    seqs_ban = seqs
    if args.domain_crop:
        crops = json.load(open(args.domain_crop))
        seqs_ban = {t: crops[str(DRUGBAN_MAX_SEQ)][t]["seq"] if t in crops[str(DRUGBAN_MAX_SEQ)] else s
                    for t, s in seqs.items()}
        seqs = {t: crops[str(MAX_SEQ)][t]["seq"] if t in crops[str(MAX_SEQ)] else s for t, s in seqs.items()}
        assert all(len(s) <= MAX_SEQ for s in seqs.values()) and all(len(s) <= DRUGBAN_MAX_SEQ for s in seqs_ban.values())
        print(f"domain crop: {len(crops[str(MAX_SEQ)])} proteins cropped to {MAX_SEQ}, "
              f"{len(crops[str(DRUGBAN_MAX_SEQ)])} to {DRUGBAN_MAX_SEQ}", flush=True)
    P = torch.tensor(np.stack([label_seq(seqs[p], CHARPROTSET, MAX_SEQ) for p in list(data["protein_ids"])[:npr]]),
                     device=dev)
    S = torch.tensor(np.stack([label_seq(s, CHARISOSMISET, MAX_SMI) for s in smiles]), device=dev)
    graphs = [mol_graph(s) for s in smiles]
    if "drugban" in args.models.split(","):
        P_ban = torch.tensor(np.stack([label_seq(seqs_ban[p], CHARPROTSET, DRUGBAN_MAX_SEQ)
                                       for p in list(data["protein_ids"])[:npr]]), device=dev)
        ban_graphs = [drugban_graph(s) for s in smiles]

    def forward(model, kind, pairs):
        c = [p[0] for p in pairs]
        prot = torch.tensor([p[1] for p in pairs], device=dev)
        if kind == "drugban":
            return model(Batch.from_data_list([ban_graphs[i] for i in c]).to(dev), P_ban[prot])
        seq = P[prot]
        if kind == "deepdta":
            return model(S[torch.tensor(c, device=dev)], seq)
        return model(Batch.from_data_list([graphs[i] for i in c]).to(dev), seq)

    def predict(model, kind, pairs):
        model.eval()
        out, bs = [], EVAL_BATCH[kind]
        with torch.no_grad():
            for i in range(0, len(pairs), bs):
                out.append(torch.sigmoid(forward(model, kind, pairs[i:i + bs])).cpu().numpy())
        return np.concatenate(out)

    train_exclude = training_exclusion(train, reserved)
    if args.export_protocol:
        # Everything an external trainer (train_drugban.py, official DGL code) needs to follow this
        # protocol exactly; its benchmark scores are read back with --import_scores.
        prot_seqs = [seqs_ban[p] for p in list(data["protein_ids"])[:npr]]
        pickle.dump({"seed": args.seed, "graph_path": args.graph_path, "nc": nc, "npr": npr, "smiles": smiles,
                     "protein_sequences": prot_seqs, "train": train, "val_all": val_all, "y_val": y_val,
                     "benches": benches, "train_exclude": train_exclude, "epochs": args.epochs,
                     "patience": args.patience,
                     "lead_drivers": [[int(p), name] for p, name in lead_drivers],
                     "ext_drug_idx": [int(i) for i in ext_drug_idx],
                     "driver_actives": {int(p): sorted(int(c) for c in s) for p, s in driver_actives.items()}},
                    open(args.export_protocol, "wb"))
        print("exported protocol to", args.export_protocol, flush=True)
        return
    results, score_store = {}, {}
    if args.import_scores:
        sc = np.load(args.import_scores, allow_pickle=False)
        meta = json.loads(str(sc["meta"]))
        res = {k: meta[k] for k in ("val_roc_auc", "config", "stopped_at_epoch", "best_epoch", "implementation")}
        for name, (pairs, y) in benches.items():
            assert np.array_equal(sc[f"{name}|pairs"], np.asarray(pairs)) and np.array_equal(sc[f"{name}|y"], y)
            s = sc[f"{name}|score"]
            res[name] = metrics(pairs, y, s)
            score_store[f"{meta['kind']}|{name}|score"] = s
        results[meta["kind"]] = res
        print(meta["kind"], {b: {k: round(res[b][k], 3) for k in ("roc_auc", "auprc", "mcc")} for b in benches},
              flush=True)
    for kind in ([] if args.import_scores else args.models.split(",")):
        best = None
        for cfg in [CONFIG[kind]]:
            torch.manual_seed(args.seed)
            rng = np.random.RandomState(args.seed)
            model = build_model(kind, cfg).to(dev)
            opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
            cfg_best, wait, state, stop, best_epoch = -1.0, 0, None, args.epochs, 0
            for epoch in range(1, args.epochs + 1):
                negs = sample_negatives(rng, np.arange(nc), npr, len(train), train_exclude)
                pairs = train + negs
                y = np.array([1.0] * len(train) + [0.0] * len(negs), dtype=np.float32)
                order = rng.permutation(len(pairs))
                model.train()
                for i in range(0, len(order), BATCH[kind]):
                    ix = order[i:i + BATCH[kind]]
                    if len(ix) < 2:
                        continue
                    loss = F.binary_cross_entropy_with_logits(forward(model, kind, [pairs[j] for j in ix]),
                                                              torch.tensor(y[ix], device=dev))
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                auc = roc_auc_score(y_val, predict(model, kind, val_all))
                if auc > cfg_best:
                    cfg_best, wait, best_epoch = auc, 0, epoch
                    state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                else:
                    wait += 1
                    if wait >= args.patience:
                        stop = epoch
                        break
            print(f"seed {args.seed} {kind} {cfg} val_auc={cfg_best:.4f} best_epoch={best_epoch} stop={stop}", flush=True)
            if best is None or cfg_best > best[0]:
                best = (cfg_best, cfg, state, stop, best_epoch)
        model = build_model(kind, best[1]).to(dev)
        model.load_state_dict(best[2])
        if args.save_models:
            os.makedirs(args.save_models, exist_ok=True)
            torch.save({"seed": args.seed, "kind": kind, "config": best[1],
                        "val_roc_auc": float(best[0]), "stopped_at_epoch": best[3], "best_epoch": best[4],
                        "model_state_dict": best[2]}, os.path.join(args.save_models, f"{kind}_seed_{args.seed}.pth"))
        res = {"val_roc_auc": float(best[0]), "config": best[1], "stopped_at_epoch": best[3], "best_epoch": best[4]}
        for name, (pairs, y) in benches.items():
            s = predict(model, kind, pairs)
            res[name] = metrics(pairs, y, s)
            score_store[f"{kind}|{name}|score"] = s
        results[kind] = res
        if args.lead_out:
            dr = {}
            for p_d, name in lead_drivers:
                s = predict(model, kind, [(d, p_d) for d in ext_drug_idx])
                order = np.argsort(-s, kind="stable")
                acts = driver_actives.get(p_d, set())
                top10 = [ext_drug_idx[order[j]] for j in range(min(10, len(order)))]
                dr[name] = {"active_at_1": int(ext_drug_idx[order[0]] in acts),
                            "hits_top10": int(sum(d in acts for d in top10)),
                            "n_actives": len(acts & set(ext_drug_idx))}
            lead_store[kind] = {"per_driver": dr,
                                "drivers_active_at_1": int(sum(v["active_at_1"] for v in dr.values())),
                                "hits_top10_sum": int(sum(v["hits_top10"] for v in dr.values())),
                                "n_drivers": len(dr)}
        print(kind, {b: {k: round(res[b][k], 3) for k in ("roc_auc", "auprc", "mcc")} for b in benches}, flush=True)

    out = {"seed": args.seed, "graph_path": args.graph_path,
           "protocol": "trained from scratch on run ChEMBL35 training pairs; 1:1 resampled training negatives; "
                       "standard published configuration, no search; up to 300 epochs, patience 20; external compounds inference only; threshold 0.5",
           "n_train_positives": len(train), "n_val": len(val_all), "n_external_compounds": len(ext_ids),
           "external_compounds_structure_in_training": int(ext_same_structure),
           "bench_sizes": {k: [int(v[1].sum()), int((v[1] == 0).sum())] for k, v in benches.items()},
           "seconds": time.time() - t0, "results": results}
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    json.dump(out, open(args.output, "w"), indent=2)
    if args.scores_out:
        for name, (pairs, y) in benches.items():
            score_store[f"{name}|pairs"] = np.asarray(pairs)
            score_store[f"{name}|y"] = y
        np.savez_compressed(args.scores_out, **score_store)
    if args.lead_out:
        json.dump({"seed": args.seed, "lead": lead_store}, open(args.lead_out, "w"), indent=2)


if __name__ == "__main__":
    main()
