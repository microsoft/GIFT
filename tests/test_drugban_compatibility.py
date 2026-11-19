import csv
import io
import json
import pickle
import sys
import unittest
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "code"), str(ROOT / "code/third_party/DrugBAN_9923f8c")]

from _drugban_graph import batch_graphs, integer_label_protein, molecule_graph
from configs import get_cfg_defaults
from models import DrugBAN, binary_cross_entropy


class DrugBANCompatibilityTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.checkpoint = torch.load(
            ROOT / "results/gift/baseline_models/drugban/drugban_seed_42.pth",
            map_location="cpu", weights_only=False,
        )

    def setUp(self):
        self.model = DrugBAN(**get_cfg_defaults())
        self.model.load_state_dict(self.checkpoint["model_state_dict"], strict=True)

    def test_graphs_and_logits_match_official_cpu_reference(self):
        # Generated with the committed upstream model, dgllife 0.3.2 and DGL
        # 1.1.3 on torch 2.4 CPU. No DGL package is needed to run this test.
        smiles = ["CCO", "c1ccccc1", "CC(=O)[O-].[Na+]", "N[C@@H](C)C(=O)O", "[NH4+]", "[He]"]
        sequences = ["ACDEFGHIKLMNPQRSTVWY", "mktBXZOU", "A" * 1205, "", "G", "MKT"]
        graphs = [molecule_graph(value) for value in smiles]
        proteins = torch.tensor(np.asarray([integer_label_protein(value) for value in sequences]))
        with np.load(ROOT / "tests/fixtures/drugban_official_cpu.npz", allow_pickle=False) as reference:
            for index, graph in enumerate(graphs):
                np.testing.assert_array_equal(graph.x.numpy(), reference[f"features_{index}"])
                np.testing.assert_array_equal(graph.edge_index.numpy(), reference[f"edges_{index}"])
            np.testing.assert_array_equal(proteins, reference["proteins"])
            self.model.eval()
            batch = batch_graphs(graphs)
            original_features = batch.x.clone()
            with torch.no_grad():
                actual = self.model(batch, proteins)[-1]
                repeated = self.model(batch, proteins)[-1]
            np.testing.assert_allclose(actual, reference["logits"], rtol=1e-5, atol=1e-6)
            torch.testing.assert_close(actual, repeated, rtol=0, atol=0)
            torch.testing.assert_close(batch.x, original_features, rtol=0, atol=0)

    def test_released_checkpoint_reproduces_saved_benchmark(self):
        with (ROOT / "data/graphs/graph_seed42.pkl").open("rb") as stream:
            data = pickle.load(stream)
        with (ROOT / "data/chembl_lung_cancer_all_compounds.csv").open() as stream:
            smiles = {row["compound_id"]: row["smiles"] for row in csv.DictReader(stream)}
        sequences = json.loads((ROOT / "data/protein_sequences.json").read_text())
        with np.load(
            ROOT / "results/gift/baseline_models/drugban/run_42/drugban_dgl/scores.npz",
            allow_pickle=False,
        ) as saved:
            pairs, expected = saved["same_version|pairs"], saved["same_version|score"]
        graphs = {int(c): molecule_graph(smiles[data["compound_ids"][c]]) for c in np.unique(pairs[:, 0])}
        proteins = {
            int(p): integer_label_protein(sequences[data["protein_ids"][p]]) for p in np.unique(pairs[:, 1])
        }
        self.model.eval()
        predictions = []
        with torch.no_grad():
            for start in range(0, len(pairs), 8):
                batch = pairs[start:start + 8]
                score = self.model(
                    batch_graphs([graphs[int(c)] for c, _ in batch]),
                    torch.tensor(np.asarray([proteins[int(p)] for _, p in batch])),
                )[-1]
                predictions.append(torch.sigmoid(score).squeeze(1).numpy())
        actual = np.concatenate(predictions)
        # Original scores were produced on GPU; CPU reduction precision differs.
        np.testing.assert_allclose(actual, expected, rtol=1e-3, atol=5e-4)
        np.testing.assert_array_equal(actual >= 0.5, expected >= 0.5)

    def test_training_and_optimizer_checkpoint_resume(self):
        self.model.train()
        graphs = batch_graphs([molecule_graph("CCO"), molecule_graph("c1ccccc1")])
        proteins = torch.tensor(np.asarray([integer_label_protein("ACDEFG"), integer_label_protein("MKT")]))
        optimizer = torch.optim.Adam(self.model.parameters(), lr=5e-5)
        _, loss = binary_cross_entropy(self.model(graphs, proteins)[-1], torch.tensor([0., 1.]))
        loss.backward()
        gradients = [parameter.grad for parameter in self.model.parameters() if parameter.requires_grad]
        self.assertTrue(torch.isfinite(loss).item())
        self.assertTrue(all(gradient is not None and torch.isfinite(gradient).all().item() for gradient in gradients))
        optimizer.step()
        buffer = io.BytesIO()
        torch.save({"model": self.model.state_dict(), "opt": optimizer.state_dict()}, buffer)
        buffer.seek(0)
        state = torch.load(buffer, map_location="cpu", weights_only=True)
        restored = DrugBAN(**get_cfg_defaults())
        restored.load_state_dict(state["model"], strict=True)
        resumed_optimizer = torch.optim.Adam(restored.parameters(), lr=5e-5)
        resumed_optimizer.load_state_dict(state["opt"])
        self.model.eval()
        restored.eval()
        with torch.no_grad():
            torch.testing.assert_close(self.model(graphs, proteins)[-1], restored(graphs, proteins)[-1])
        restored.train()
        resumed_optimizer.zero_grad()
        _, resumed_loss = binary_cross_entropy(restored(graphs, proteins)[-1], torch.tensor([0., 1.]))
        resumed_loss.backward()
        resumed_optimizer.step()
        self.assertTrue(torch.isfinite(resumed_loss).item())

    def test_invalid_or_oversized_molecules_are_rejected(self):
        for smiles in ("", "not_a_molecule", "C" * 291):
            with self.subTest(smiles=smiles), self.assertRaises(ValueError):
                molecule_graph(smiles)


if __name__ == "__main__":
    unittest.main()
