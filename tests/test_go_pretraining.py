import json
import math
import os
import pickle
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import networkx as nx
import numpy as np
import torch
import torch.nn.functional as F

from model.go_dataset import GoDataset
from model.loss import GoContrastLoss


@contextmanager
def temporary_cache_directory():
    previous = os.getcwd()
    with tempfile.TemporaryDirectory() as directory:
        Path(directory, "dataset").mkdir()
        Path(directory, "processed_data").mkdir()
        os.chdir(directory)
        try:
            yield
        finally:
            os.chdir(previous)


class GoSamplePoolTests(unittest.TestCase):
    def setUp(self):
        np.random.seed(0)
        self.dataset = GoDataset.__new__(GoDataset)
        self.dataset.pos_k = 1
        self.dataset.pos_max = 1
        self.dataset.neg_k = 3
        self.dataset.neg_max = 5
        self.dataset.pos_rate = 0.5
        self.dataset.go_graph_nx = nx.DiGraph()
        self.dataset.go_graph_nx.add_nodes_from(range(7))
        self.dataset.go_graph_nx.add_edges_from([
            (0, 1), (0, 2), (1, 2), (3, 1), (4, 2), (5, 6),
        ])
        self.sp_gos = {"p0": [0, 1, 2], "p1": [3, 1, 2], "p2": [5, 6]}
        self.go2prot = {}
        for protein, terms in self.sp_gos.items():
            for go in terms:
                self.go2prot.setdefault(go, set()).add(protein)

    def assert_valid_pools(self, dataset):
        for go, (pos, neg) in enumerate(dataset.data_info):
            with self.subTest(go=go):
                self.assertTrue(pos)
                self.assertTrue(neg)
                self.assertNotIn(go, neg)
                self.assertTrue(set(pos).isdisjoint(neg))
                # 图的边指向父节点，NetworkX descendants 在这里是 GO 祖先。
                self.assertTrue(nx.descendants(dataset.go_graph_nx, go).isdisjoint(neg))
                self.assertEqual(len(neg), len(set(neg)))
                for candidate in neg:
                    self.assertTrue(self.go2prot.get(go, set()).isdisjoint(
                        self.go2prot.get(candidate, set())))

    def test_pools_exclude_self_all_ancestors_positives_and_cooccurrences(self):
        self.dataset.data_info = [
            self.dataset.build_sample_pools(go, self.go2prot) for go in range(7)
        ]
        self.assert_valid_pools(self.dataset)
        # 祖先 2 被 pos_max 截断，仍必须从负样本候选中排除。
        self.assertNotIn(2, self.dataset.data_info[0][0])
        self.assertNotIn(2, self.dataset.data_info[0][1])

    def test_positive_children_are_selected_before_negative_pool(self):
        self.dataset.pos_max = 2
        self.dataset.pos_rate = 0.4
        pos, neg = self.dataset.build_sample_pools(1, self.go2prot)
        self.assertEqual(pos, [2, 0])
        self.assertNotIn(0, neg)

    def test_repeated_invalid_draws_fall_back_and_small_pools_remain_sampleable(self):
        with patch("model.go_dataset.np.random.randint", return_value=0):
            pos, neg = self.dataset.build_sample_pools(0, self.go2prot)
        self.assertEqual(set(neg), {3, 4, 5, 6})
        self.dataset.data_info = [[pos, neg]]
        self.dataset.neg_k = 80
        idx, sampled_pos, sampled_neg = self.dataset[0]
        self.assertEqual(idx.item(), 0)
        self.assertEqual(sampled_pos.shape, (1,))
        self.assertEqual(sampled_neg.shape, (80,))
        self.assertTrue(set(sampled_neg.tolist()).issubset(neg))

    def test_no_valid_negatives_raises_instead_of_looping_forever(self):
        go2prot = {go: {"same_protein"} for go in range(7)}
        with self.assertRaisesRegex(ValueError, "no valid negative"):
            self.dataset.build_sample_pools(0, go2prot)

    def test_no_positive_candidates_reports_term(self):
        with self.assertRaisesRegex(ValueError, "GO term 6 has no positive"):
            self.dataset.build_sample_pools(6, {})

    def test_cache_round_trip_and_sampling_parameter_changes(self):
        self.dataset.data_info = [
            self.dataset.build_sample_pools(go, self.go2prot) for go in range(7)
        ]
        expected = self.dataset.data_info
        with temporary_cache_directory():
            self.dataset.save_dataset()
            self.dataset.data_info = None
            self.assertTrue(self.dataset.load_dataset())
            self.assertEqual(self.dataset.data_info, expected)
            self.dataset.pos_rate = 0.4
            self.assertFalse(self.dataset.load_dataset())

    def test_constructor_rebuilds_legacy_cache_then_reuses_new_cache(self):
        with temporary_cache_directory():
            for filename, value in [
                ("dataset/go_graph_nx.pkl", self.dataset.go_graph_nx),
                ("dataset/swissprot_go_cache.pkl", self.sp_gos),
                ("dataset/go_cache.pkl", [[[go], [go]] for go in range(7)]),
            ]:
                with open(filename, "wb") as stream:
                    pickle.dump(value, stream)
            with open("processed_data/go_id.json", "w") as stream:
                json.dump({f"GO:{go:07d}": go for go in range(7)}, stream)
            # 构造函数原本切到项目目录；此处让它读取隔离的测试数据。
            with patch("model.go_dataset.os.chdir"):
                rebuilt = GoDataset(1, 1, 3, 5, 0.5)
            self.assert_valid_pools(rebuilt)
            with patch("model.go_dataset.os.chdir"), patch.object(
                GoDataset, "load_sp_data", side_effect=AssertionError("cache was not reused")
            ):
                cached = GoDataset(1, 1, 3, 5, 0.5)
            self.assertEqual(cached.data_info, rebuilt.data_info)


class GoCosineLossTests(unittest.TestCase):
    def test_single_positive_matches_cross_entropy_of_cosine_scores(self):
        x = torch.tensor([[[1., 0.], [2., 0.], [0., 3.]]])
        actual = GoContrastLoss(1, 1)(x, 0.7)
        expected = F.cross_entropy(torch.tensor([[1. / 0.7, 0.]]), torch.tensor([0]))
        torch.testing.assert_close(actual, expected)

    def test_independent_positive_scaling_of_embeddings_does_not_change_loss(self):
        torch.manual_seed(0)
        x = torch.randn(3, 6, 8, dtype=torch.float64)
        scales = torch.tensor([1., 2., 0.25, 10., 0.5, 3.]).view(1, 6, 1)
        criterion = GoContrastLoss(2, 3)
        torch.testing.assert_close(criterion(x), criterion(x * scales))

    def test_multiple_equally_similar_positives_have_expected_probability_mass(self):
        x = torch.ones(2, 6, 4)
        loss = GoContrastLoss(2, 3)(x)
        self.assertAlmostEqual(loss.item(), math.log(5 / 2), places=5)

    def test_gradient_step_increases_positive_cosine_and_decreases_negative_cosine(self):
        x = torch.tensor([[[1., 0.], [0., 1.], [0., -1.]]], requires_grad=True)
        GoContrastLoss(1, 1)(x).backward()
        updated = x.detach().clone()
        updated[:, 1:] -= 0.01 * x.grad[:, 1:]
        similarities = F.cosine_similarity(updated[:, :1], updated[:, 1:], dim=-1)
        self.assertGreater(similarities[0, 0].item(), 0.)
        self.assertLess(similarities[0, 1].item(), 0.)

    def test_default_sized_embeddings_and_zero_vectors_have_finite_loss_and_gradients(self):
        torch.manual_seed(0)
        for x in [torch.randn(2, 82, 1024), torch.zeros(2, 82, 1024)]:
            with self.subTest(zero_vectors=not bool(x.any())):
                x.requires_grad_(True)
                loss = GoContrastLoss(1, 80)(x)
                loss.backward()
                self.assertTrue(bool(loss.isfinite()))
                self.assertTrue(bool(x.grad.isfinite().all()))

    def test_invalid_temperature_is_rejected(self):
        for temperature in [0., -0.07]:
            with self.assertRaisesRegex(ValueError, "temperature"):
                GoContrastLoss(1, 1)(torch.ones(1, 3, 2), temperature)


if __name__ == "__main__":
    unittest.main()
