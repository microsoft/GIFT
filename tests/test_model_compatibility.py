import csv
import io
import pickle
import sys
import unittest
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from evaluate_chembl35 import load_gift_model
from graph_cleaning import clean_graph
from pipeline_utils import score_gnn_pairs


class ModelCompatibilityTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        with (ROOT / "data/graphs/graph_seed42.pkl").open("rb") as stream:
            cls.data = pickle.load(stream)
        cls.graph = clean_graph(cls.data)[0]
        cls.device = torch.device("cpu")
        cls.checkpoint = ROOT / "results/gift/run_42/e2e/trained_model.pth"

    def setUp(self):
        self.model, _ = load_gift_model(self.data, self.device, self.checkpoint)

    def test_released_checkpoint_reproduces_saved_predictions(self):
        with (ROOT / "results/gift/run_42/misclassification/pairs_random.csv").open() as stream:
            rows = list(csv.DictReader(stream))
        compounds = {value: index for index, value in enumerate(self.data["compound_ids"])}
        proteins = {value: index for index, value in enumerate(self.data["protein_ids"])}
        pairs = [(compounds[row["compound_id"]], proteins[row["protein_id"]]) for row in rows]
        expected = np.array([float(row["score"]) for row in rows])

        actual = score_gnn_pairs(self.model, self.graph, pairs, self.device)

        np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-5)
        np.testing.assert_array_equal(actual >= 0.5, expected >= 0.5)

    def test_released_model_supports_training_and_checkpoint_round_trip(self):
        self.model.train()
        graph = self.graph
        node_types = torch.as_tensor(graph["node_types"], dtype=torch.long)
        _, embeddings = self.model(
            torch.as_tensor(graph["node_features"], dtype=torch.float32),
            torch.as_tensor(graph["edge_index"], dtype=torch.long),
            edge_attr=torch.as_tensor(graph["edge_attr"], dtype=torch.float32),
            node_types=node_types,
        )
        indices = torch.arange(8)
        logits, _ = self.model.classify_pairs(
            embeddings[node_types == 0], embeddings[node_types == 1], indices, indices
        )
        loss = F.binary_cross_entropy_with_logits(logits, (indices % 2).float())
        loss.backward()
        gradients = [parameter.grad for parameter in self.model.parameters() if parameter.grad is not None]
        self.assertTrue(torch.isfinite(loss).item())
        self.assertTrue(gradients)
        self.assertTrue(all(torch.isfinite(gradient).all().item() for gradient in gradients))

        buffer = io.BytesIO()
        torch.save(self.model.state_dict(), buffer)
        buffer.seek(0)
        state = torch.load(buffer, map_location="cpu", weights_only=True)
        restored, _ = load_gift_model(self.data, self.device, self.checkpoint)
        restored.load_state_dict(state, strict=True)
        self.model.eval()
        pairs = [(index, index) for index in range(8)]
        np.testing.assert_allclose(
            score_gnn_pairs(restored, graph, pairs, self.device),
            score_gnn_pairs(self.model, graph, pairs, self.device),
            rtol=1e-5,
            atol=1e-6,
        )


if __name__ == "__main__":
    unittest.main()
