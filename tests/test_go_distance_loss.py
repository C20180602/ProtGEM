import math
import unittest

import torch

from model.loss import GoContrastLoss2


class GoDistanceLossTests(unittest.TestCase):
    def test_multiple_positives_match_distance_probability_mass(self):
        # 相对锚点的欧氏距离依次为 5、0、10。
        x = torch.tensor([[[0., 0.], [3., 4.], [0., 0.], [6., 8.]]],
                         dtype=torch.float64)
        expected = -math.log((math.exp(-5.) + 1.) /
                             (math.exp(-5.) + 1. + math.exp(-10.)))
        self.assertAlmostEqual(GoContrastLoss2(2, 1)(x, 1.).item(), expected)

    def test_gradient_moves_positive_closer_and_negative_farther(self):
        x = torch.tensor([[[0., 0.], [1., 0.], [0., 1.]]], requires_grad=True)
        GoContrastLoss2(1, 1)(x, 1.).backward()
        updated = x.detach()[:, 1:] - 0.1 * x.grad[:, 1:]
        distances = torch.linalg.vector_norm(updated, dim=-1)
        self.assertLess(distances[0, 0].item(), 1.)
        self.assertGreater(distances[0, 1].item(), 1.)

    def test_zero_and_large_distances_have_finite_loss_and_gradients(self):
        for values in [[[0., 0.], [0., 0.], [0., 0.]],
                       [[0., 0.], [10000., 0.], [20000., 0.]]]:
            with self.subTest(values=values):
                x = torch.tensor([values], requires_grad=True)
                loss = GoContrastLoss2(1, 1)(x)
                loss.backward()
                self.assertTrue(bool(loss.isfinite()))
                self.assertTrue(bool(x.grad.isfinite().all()))

    def test_invalid_inputs_are_rejected(self):
        for temperature in [0., -0.07]:
            with self.assertRaisesRegex(ValueError, "temperature"):
                GoContrastLoss2(1, 1)(torch.ones(1, 3, 2), temperature)
        for counts in [(0, 1), (1, 0)]:
            with self.assertRaisesRegex(ValueError, "samples"):
                GoContrastLoss2(*counts)(torch.ones(1, 2, 2))
        for shape in [(1, 2), (1, 4, 2)]:
            with self.assertRaisesRegex(ValueError, "shape"):
                GoContrastLoss2(1, 1)(torch.ones(shape))


if __name__ == "__main__":
    unittest.main()
