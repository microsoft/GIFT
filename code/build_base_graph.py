"""Build data/graphs/graph_base.pkl, the base knowledge graph of all runs, from data/graphs/graph_typed_input.pkl
(same graph, splits, and edges) with label-free node features only:
- compounds (435): 10 ChEMBL physicochemical descriptors (max phase removed) + 256-bit radius-2 Morgan fingerprint
  + 167 MACCS keys + fraction sp3 + number of stereocentres;
- proteins (453): organism (3) + kinase flag + amino-acid composition (20) + full dipeptide composition (400)
  + physicochemical statistics (25) + complexity/length (4);
- per-type scalers and block layout [compound | protein].
Also writes the matching external compound feature file (data/external_compounds_features.pkl).
"""

import json
import pickle

import numpy as np
from rdkit import Chem
from rdkit.Chem import MACCSkeys, rdMolDescriptors

import pipeline_utils as vs

IN_PATH = "data/graphs/graph_typed_input.pkl"
OUT_PATH = "data/graphs/graph_base.pkl"
SEQ_PATH = "data/protein_sequences.json"
EXT_IN = "data/external_compounds_features_raw.pkl"
EXT_OUT = "data/external_compounds_features.pkl"
AA = "ACDEFGHIKLMNPQRSTVWY"
DIPEPTIDES = [a + b for a in AA for b in AA]
# Column indices in the original 267-column protein layout (see extract_graph_for_gnn.py).
P_ORGANISM, P_KINASE = [0, 1, 2], [10]
P_AAC, P_DP30, P_PHYS, P_CPLX = range(11, 31), range(31, 61), range(61, 86), range(86, 90)


def compound_extra(smiles):
    mol = Chem.MolFromSmiles(smiles)
    maccs = np.zeros(167, np.float64)
    for b in MACCSkeys.GenMACCSKeys(mol).GetOnBits():
        maccs[b] = 1.0
    return np.concatenate([maccs, [rdMolDescriptors.CalcFractionCSP3(mol),
                                   rdMolDescriptors.CalcNumAtomStereoCenters(mol)]])


def compound_features(raw267, smiles):
    return np.concatenate([raw267[:10], raw267[11:267], compound_extra(smiles)])


def dipeptide_composition(seq):
    s = "".join(a for a in seq.upper() if a in AA)
    counts = dict.fromkeys(DIPEPTIDES, 0)
    for i in range(len(s) - 1):
        counts[s[i:i + 2]] += 1
    return np.array([counts[dp] for dp in DIPEPTIDES], np.float64) / max(len(s) - 1, 1)


def main():
    d = pickle.load(open(IN_PATH, "rb"))
    nt = np.asarray(d["node_types"])
    comp_rows, prot_rows = np.where(nt == 0)[0], np.where(nt == 1)[0]
    (c0, c1), (p0, p1) = d["feature_layout"]["compound"], d["feature_layout"]["protein"]
    X = np.asarray(d["node_features"], np.float64)
    raw_c = X[comp_rows, c0:c1] * d["compound_scaler_scale"] + d["compound_scaler_mean"]
    raw_p = X[prot_rows, p0:p1] * d["protein_scaler_scale"] + d["protein_scaler_mean"]
    assert list(d["compound_raw_columns"]) == list(range(267))

    md = vs.load_chembl35_compound_metadata(d["compound_ids"])
    fc = np.stack([compound_features(raw_c[i], md[c]["smiles"]) for i, c in enumerate(d["compound_ids"])])

    col = {orig: i for i, orig in enumerate(d["protein_raw_columns"])}
    pick = lambda cols: raw_p[:, [col[c] for c in cols]]
    seqs = json.load(open(SEQ_PATH))
    dpc = np.stack([dipeptide_composition(seqs[p]) for p in d["protein_ids"]])
    aac = np.stack([[s.count(a) / len(s) for a in AA]
                    for s in ("".join(a for a in seqs[p].upper() if a in AA) for p in d["protein_ids"])])
    # The sequences must reproduce the stored input-graph composition and the 30 stored dipeptides exactly.
    common30 = ['LE', 'AL', 'EL', 'LL', 'LA', 'LV', 'VL', 'LI', 'IL', 'VA', 'AV', 'AA', 'GA', 'AG', 'GL',
                'LG', 'GV', 'VG', 'GG', 'LK', 'KL', 'LR', 'RL', 'LS', 'SL', 'LT', 'TL', 'LQ', 'QL', 'LN']
    err_aac = np.abs(aac - pick(P_AAC)).max()
    err_dp = np.abs(dpc[:, [DIPEPTIDES.index(x) for x in common30]] - pick(P_DP30)).max()
    print(f"sequence check vs input graph: max |AAC diff| {err_aac:.2e}, max |30-dipeptide diff| {err_dp:.2e}")
    assert err_aac < 1e-6 and err_dp < 1e-6
    fp = np.hstack([pick(P_ORGANISM), pick(P_KINASE), pick(P_AAC), dpc, pick(P_PHYS), pick(P_CPLX)])
    print(f"compounds: {fc.shape[1]} (10 descriptors + 256 Morgan + 167 MACCS + fsp3 + stereocentres) | "
          f"proteins: {fp.shape[1]} (4 annotations + 20 AAC + 400 DPC + 25 physchem + 4 complexity)")

    def scale(m):
        mean, std = m.mean(0), m.std(0)
        std[std == 0] = 1.0
        return (m - mean) / std, mean, std

    zc, cm, cs = scale(fc)
    zp, pm, ps = scale(fp)
    dc, dp = zc.shape[1], zp.shape[1]
    x = np.zeros((len(nt), dc + dp), np.float32)
    x[comp_rows, :dc], x[prot_rows, dc:] = zc, zp

    out = dict(d)
    out.update({
        "node_features": x,
        "feature_layout": {"compound": [0, dc], "protein": [dc, dc + dp]},
        "compound_raw_columns": list(range(dc)), "compound_raw_dim": dc,
        "compound_scaler_mean": cm.tolist(), "compound_scaler_scale": cs.tolist(),
        "protein_raw_columns": None, "protein_scaler_mean": pm.tolist(), "protein_scaler_scale": ps.tolist(),
        "external_features_pkl": EXT_OUT,
        "protein_metadata_source": "label-free; full dipeptide composition, no 3-mers; compounds + MACCS",
    })
    pickle.dump(out, open(OUT_PATH, "wb"))
    print(f"Saved {OUT_PATH} with node_features {x.shape}")

    ext = pickle.load(open(EXT_IN, "rb"))
    ext_out = {}
    for cid, rec in ext.items():
        new = dict(rec)
        new["features"] = compound_features(np.asarray(rec["features"], np.float64), rec["smiles"]).astype(np.float32)
        ext_out[cid] = new
    pickle.dump(ext_out, open(EXT_OUT, "wb"))
    print(f"Saved {EXT_OUT} ({len(ext_out)} compounds, {dc} raw features each)")


if __name__ == "__main__":
    main()
