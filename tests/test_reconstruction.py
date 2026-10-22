import json
import os
import pickle
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import dgl
import networkx as nx
import numpy as np
import torch
import torch.nn.functional as F

from model.dataset import MyDataset, one2num
from model.loss import GeometricStructureLoss, SequenceLoss
from model.predictor import ReconstructDecoder
from train import create_optimizer


class ReconstructionDataTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.addCleanup(self.temporary.cleanup)
        previous = os.getcwd()
        os.chdir(self.root)
        self.addCleanup(os.chdir, previous)
        self.protein_info = {
            "P0": {"seq": "ARNDC", "go": ["GO:0000001"], "domain": []},
            "P1": {"seq": "QEGHILK", "go": ["GO:0000001"], "domain": []},
        }
        self.coords_path = self.root / "processed_data/TEST_protein_coords.pkl"
        self.cache_dir = self.root / "cache"
        self.cache_dir.mkdir()
        (self.root / "processed_data").mkdir()
        (self.root / "dataset").mkdir()
        self.write_json("processed_data/TEST_protein_info.json", self.protein_info)
        self.write_json("processed_data/interpro_id.json", [])
        self.write_json("processed_data/go_id.json", {"GO:0000001": 0, "GO:0000002": 1})
        self.write_json("processed_data/go_terms.json", [
            {"id": "GO:0000001", "namespace": "biological_process"},
            {"id": "GO:0000002", "namespace": "biological_process"},
        ])
        go_graph = nx.DiGraph([(0, 1)])
        with open(self.root / "dataset/go_graph_nx.pkl", "wb") as stream:
            pickle.dump(go_graph, stream)
        for pid, info in self.protein_info.items():
            directory = self.root / "dataset/TEST" / pid
            directory.mkdir(parents=True)
            length = len(info["seq"])
            with open(directory / "seq_emb.pkl", "wb") as stream:
                pickle.dump(np.zeros((length + 2, 8), dtype=np.float32), stream)
            graph = dgl.graph(([], []), num_nodes=length)
            dgl.save_graphs(str(directory / "struct_gvp.dgl"), [graph])

    def write_json(self, path, value):
        with open(self.root / path, "w") as stream:
            json.dump(value, stream)

    def build_dataset(self, use_cache=False):
        # 保持构造函数在隔离的测试目录内读取数据。
        with patch("model.dataset.os.chdir"):
            return MyDataset("TEST", "bp", self.cache_dir,
                             target_device=torch.device("cpu"), use_cache=use_cache)

    def test_disabled_reconstruction_needs_no_coordinate_file_or_sequence_cache(self):
        dataset = self.build_dataset()
        self.assertEqual(len(dataset), 2)
        self.assertIsNone(dataset.protein_coords)
        self.assertFalse(hasattr(dataset, "re_seq"))
        self.assertFalse(self.coords_path.exists())
        self.assertFalse((self.cache_dir / "re_seq.pkl").exists())
        self.assertTrue(dataset.has_cache())

    def test_legacy_sequence_cache_is_ignored_when_reusing_dataset_cache(self):
        self.build_dataset()
        (self.cache_dir / "re_seq.pkl").write_bytes(b"unused legacy cache")
        cached = self.build_dataset(use_cache=True)
        self.assertEqual(len(cached), 2)
        self.assertIsNone(cached.protein_coords)
        self.assertFalse(hasattr(cached, "re_seq"))
        (self.cache_dir / "re_seq.pkl").unlink()
        self.assertTrue(cached.has_cache())

    def test_sequence_targets_are_built_for_current_batch_with_padding(self):
        dataset = self.build_dataset()
        targets = dataset.generate_re_seq(["P0", "P1"], 7, torch.device("cpu"))
        self.assertEqual(targets.shape, (2, 7, 21))
        self.assertEqual(targets.dtype, torch.float32)
        self.assertEqual(targets.argmax(-1).tolist(), [
            [one2num[aa] for aa in "ARNDC"] + [20, 20],
            [one2num[aa] for aa in "QEGHILK"],
        ])
        torch.testing.assert_close(targets.sum(-1), torch.ones(2, 7))
        self.assertFalse(hasattr(dataset, "re_seq"))
        self.assertIsNone(dataset.protein_coords)
        with self.assertRaisesRegex(ValueError, "exceeds padded length"):
            dataset.generate_re_seq(["P0"], 3, torch.device("cpu"))

    def test_coordinates_are_loaded_only_on_first_structure_target_request(self):
        dataset = self.build_dataset()
        coords = {pid: np.arange(len(info["seq"]) * 3, dtype=np.float32).reshape(-1, 3)
                  for pid, info in self.protein_info.items()}
        with open(self.coords_path, "wb") as stream:
            pickle.dump(coords, stream)
        targets = dataset.generate_re_struct(["P0", "P1"], 7, torch.device("cpu"))
        self.assertEqual([len(target) for target in targets], [5, 7])
        torch.testing.assert_close(targets[0], torch.from_numpy(coords["P0"]))
        self.coords_path.unlink()
        second = dataset.generate_re_struct(["P1"], 7, torch.device("cpu"))
        torch.testing.assert_close(second[0], targets[1])


class ReconstructionTrainingTests(unittest.TestCase):
    def test_decoder_parameters_update_and_gradients_are_cleared_by_shared_optimizer(self):
        torch.manual_seed(0)
        pg_model = torch.nn.Linear(8, 16)
        decoder = ReconstructDecoder(16, 12)
        optimizer = create_optimizer(pg_model, decoder, 0.001)
        global_features = pg_model(torch.randn(2, 8))
        token_features = torch.randn(2, 7, 12)
        seq_logits, coords_logits = decoder(global_features, token_features, [5, 7])
        targets = F.one_hot(torch.randint(0, 20, (2, 7)), 21).float()
        mask = torch.ones(2, 7, dtype=torch.bool)
        mask[0, 5:] = False
        loss = SequenceLoss()(seq_logits, targets, mask)
        for predicted in coords_logits:
            loss = loss + GeometricStructureLoss()(predicted, torch.randn_like(predicted))
        before_encoder = pg_model.weight.detach().clone()
        before_token_head = decoder.token_head.weight.detach().clone()
        before_coord_head = decoder.coord_head[-2].weight.detach().clone()
        loss.backward()
        self.assertTrue(bool(loss.isfinite()))
        self.assertTrue(all(parameter.grad is None or bool(parameter.grad.isfinite().all())
                            for parameter in decoder.parameters()))
        optimizer.step()
        self.assertFalse(torch.equal(pg_model.weight, before_encoder))
        self.assertFalse(torch.equal(decoder.token_head.weight, before_token_head))
        self.assertFalse(torch.equal(decoder.coord_head[-2].weight, before_coord_head))
        optimizer.zero_grad(set_to_none=True)
        self.assertTrue(all(parameter.grad is None for parameter in decoder.parameters()))
        self.assertTrue(all(parameter.grad is None for parameter in pg_model.parameters()))

    def test_optimizer_without_reconstruction_updates_only_main_model(self):
        model = torch.nn.Linear(3, 2)
        optimizer = create_optimizer(model, None, 0.001)
        self.assertEqual(len(optimizer.param_groups), 1)
        before = model.weight.detach().clone()
        model(torch.ones(1, 3)).sum().backward()
        optimizer.step()
        self.assertFalse(torch.equal(model.weight, before))

    def test_sequence_loss_selects_masked_positions_in_single_item_batch(self):
        torch.manual_seed(0)
        logits = torch.randn(1, 5, 21, requires_grad=True)
        targets = F.one_hot(torch.tensor([[0, 1, 2, 3, 4]]), 21).float()
        mask = torch.tensor([[False, True, False, True, False]])
        criterion = SequenceLoss(label_smoothing=0)
        actual = criterion(logits, targets, mask)
        expected = F.cross_entropy(logits[0, [1, 3]], torch.tensor([1, 3]))
        torch.testing.assert_close(actual, expected)
        actual.backward()
        self.assertEqual(logits.grad[0, [0, 2, 4]].abs().sum().item(), 0.)

    def test_empty_sequence_mask_returns_zero_loss_with_finite_gradients(self):
        logits = torch.randn(1, 5, 21, requires_grad=True)
        targets = F.one_hot(torch.zeros(1, 5, dtype=torch.long), 21).float()
        loss = SequenceLoss()(logits, targets, torch.zeros(1, 5, dtype=torch.bool))
        self.assertEqual(loss.item(), 0.)
        loss.backward()
        self.assertEqual(logits.grad.abs().sum().item(), 0.)


if __name__ == "__main__":
    unittest.main()
