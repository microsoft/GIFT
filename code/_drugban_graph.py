"""Local PyTorch/PyG backend preserving the released DrugBAN checkpoint layout.

GCN layer structure is adapted from awslabs/dgl-lifesci (Apache-2.0).
Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
See NOTICE.txt for the license and attribution.
"""

import logging

import numpy as np
import torch
from rdkit import Chem
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset
from torch_geometric.data import Batch, Data

from train_dta_baselines import CANON_HYB, CANON_SYMBOLS, CHARPROTSET, onehot_strict


class _GraphConv(nn.Module):
    def __init__(self, in_feats, out_feats, activation):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(in_feats, out_feats))
        self.bias = nn.Parameter(torch.zeros(out_feats))
        self.activation = activation
        nn.init.xavier_uniform_(self.weight)

    def forward(self, graph, features):
        source, target = graph.edge_index
        aggregate = torch.zeros_like(features).index_add_(0, target, features[source])
        return self.activation(aggregate @ self.weight + self.bias)


class _GCNLayer(nn.Module):
    def __init__(self, in_feats, out_feats, activation):
        super().__init__()
        self.graph_conv = _GraphConv(in_feats, out_feats, activation)
        self.dropout = nn.Dropout(0.0)
        self.res_connection = nn.Linear(in_feats, out_feats)
        self.bn_layer = nn.BatchNorm1d(out_feats)
        self.activation = activation

    def forward(self, graph, features):
        convolved = self.graph_conv(graph, features)
        residual = self.activation(self.res_connection(features))
        return self.bn_layer(self.dropout(convolved + residual))


class GCN(nn.Module):
    """The unnormalized residual GCN configuration used by vanilla DrugBAN."""

    def __init__(self, in_feats, hidden_feats, activation=None):
        super().__init__()
        activations = [F.relu] * len(hidden_feats) if activation is None else activation
        if len(activations) != len(hidden_feats):
            raise ValueError("Expected one activation per DrugBAN GCN layer")
        self.gnn_layers = nn.ModuleList()
        for out_feats, function in zip(hidden_feats, activations):
            self.gnn_layers.append(_GCNLayer(in_feats, out_feats, function))
            in_feats = out_feats

    def forward(self, graph, features):
        for layer in self.gnn_layers:
            features = layer(graph, features)
        return features


def molecule_graph(smiles, max_nodes=290):
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None or molecule.GetNumAtoms() == 0:
        raise ValueError("DrugBAN requires a nonempty valid molecule")
    molecule = Chem.RenumberAtoms(molecule, Chem.CanonicalRankAtoms(molecule))
    count = molecule.GetNumAtoms()
    if count > max_nodes:
        raise ValueError(f"DrugBAN molecule has {count} atoms; maximum is {max_nodes}")
    features = []
    for atom in molecule.GetAtoms():
        features.append(
            [atom.GetSymbol() == symbol for symbol in CANON_SYMBOLS]
            + onehot_strict(atom.GetDegree(), range(11))
            + onehot_strict(atom.GetImplicitValence(), range(7))
            + [atom.GetFormalCharge(), atom.GetNumRadicalElectrons()]
            + onehot_strict(atom.GetHybridization(), CANON_HYB)
            + [atom.GetIsAromatic()]
            + onehot_strict(atom.GetTotalNumHs(), range(5))
            + [0]
        )
    nodes = torch.zeros(max_nodes, 75)
    nodes[:count] = torch.tensor(features, dtype=torch.float32)
    nodes[count:, -1] = 1.0
    edges = []
    for bond in molecule.GetBonds():
        first, second = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        edges.extend([(first, second), (second, first)])
    # The official featurizer adds real-node self-loops before padding, then
    # adds a second set of self-loops over all real and virtual nodes.
    edges.extend((index, index) for index in range(count))
    edges.extend((index, index) for index in range(max_nodes))
    return Data(x=nodes, edge_index=torch.tensor(edges, dtype=torch.long).t().contiguous())


def integer_label_protein(sequence, max_length=1200):
    encoded = np.zeros(max_length)
    for index, letter in enumerate(sequence[:max_length]):
        code = CHARPROTSET.get(letter.upper(), 0)
        if not code:
            logging.warning("Unknown protein character %r treated as padding", letter)
        encoded[index] = code
    return encoded


class DTIDataset(Dataset):
    def __init__(self, list_IDs, df, max_drug_nodes=290):
        self.list_IDs = list_IDs
        self.df = df
        self.max_drug_nodes = max_drug_nodes

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        row = self.df.iloc[self.list_IDs[index]]
        return (
            molecule_graph(row["SMILES"], self.max_drug_nodes),
            integer_label_protein(row["Protein"]),
            row["Y"],
        )


def batch_graphs(graphs):
    return Batch.from_data_list(graphs)
