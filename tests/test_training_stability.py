import logging
import unittest
from unittest.mock import patch

import dgl
import torch

from model.loss import ContrastLoss
from model.struct_model import StructEncoder, VectorLayerNorm
from model.training_stability import backward_and_step, numerical_error_context, require_finite


class ContrastStabilityTests(unittest.TestCase):
    def test_ppi_vector_matches_original_value_and_gradient(self):
        from train_ppi import ContrastLoss as PPIContrastLoss
        pred = torch.tensor([.1, .6, .9], dtype=torch.float64, requires_grad=True)
        labels = torch.tensor([1., 0., 1.], dtype=torch.float64)
        exp = torch.exp(pred / .2)
        original = -((exp * labels).sum() / exp.sum()).log()
        stable = PPIContrastLoss()(pred, labels, .2)
        torch.testing.assert_close(stable, original)
        old_grad, = torch.autograd.grad(original, pred, retain_graph=True)
        new_grad, = torch.autograd.grad(stable, pred)
        torch.testing.assert_close(new_grad, old_grad)

    def test_matches_original_value_and_gradient(self):
        torch.manual_seed(7)
        for labels in [torch.tensor([[1., 0., 1.], [0., 1., 0.]]),
                       torch.tensor([[.2, 0., 1.], [0., .7, .4]])]:
            pred = torch.rand(2, 3, dtype=torch.float64, requires_grad=True)
            weights = labels.double()
            exp = torch.exp(pred / .05)
            pos = (exp * weights).sum(-1)
            neg = (exp * (1 - weights)).sum(-1)
            original = -(pos / (pos + neg)).log().mean()
            stable = ContrastLoss()(pred, weights)
            torch.testing.assert_close(stable, original)
            old_grad, = torch.autograd.grad(original, pred, retain_graph=True)
            new_grad, = torch.autograd.grad(stable, pred)
            torch.testing.assert_close(new_grad, old_grad)

    def test_small_temperature_and_large_scores_remain_finite(self):
        for values, temp in [([0., 1., .5], .001),
                             ([-1000., 1000., 0.], .05),
                             ([1000., 1000., 1000.], .05)]:
            with self.subTest(values=values, temp=temp):
                pred = torch.tensor([values], requires_grad=True)
                loss = ContrastLoss()(pred, torch.tensor([[1., 0., 0.]]), temp)
                loss.backward()
                self.assertTrue(loss.isfinite())
                self.assertTrue(pred.grad.isfinite().all())

    def test_half_precision_reduction_remains_finite(self):
        pred = torch.tensor([[0., 1., .5]], dtype=torch.float16, requires_grad=True)
        loss = ContrastLoss()(pred, torch.tensor([[1., 0., 0.]]))
        loss.backward()
        self.assertTrue(loss.isfinite())
        self.assertTrue(pred.grad.isfinite().all())

    def test_empty_positive_rows_raise_before_backward(self):
        with self.assertRaisesRegex(ValueError, r"empty rows: \[1\]"):
            ContrastLoss()(torch.zeros(2, 3), torch.tensor([[1., 0., 0.], [0., 0., 0.]]))

    def test_invalid_temperature_and_labels_are_rejected(self):
        for temp in [0., -.1, float('nan'), float('inf')]:
            with self.assertRaisesRegex(ValueError, 'temperature'):
                ContrastLoss()(torch.zeros(1, 2), torch.ones(1, 2), temp)
        for value in [float('nan'), -1., 2.]:
            with self.assertRaisesRegex(ValueError, 'labels'):
                ContrastLoss()(torch.zeros(1, 2), torch.tensor([[1., value]]))


class GuardedUpdateTests(unittest.TestCase):
    def test_nonfinite_loss_does_not_update_parameters_or_adam(self):
        param = torch.nn.Parameter(torch.ones(2))
        optimizer = torch.optim.Adam([param])
        with patch.object(optimizer, 'step', wraps=optimizer.step) as step:
            with self.assertRaisesRegex(FloatingPointError, 'total loss'):
                backward_and_step((param * torch.nan).sum(), optimizer, [('weight', param)])
            step.assert_not_called()
        self.assertFalse(optimizer.state)
        torch.testing.assert_close(param, torch.ones(2))

    def test_finite_loss_with_invalid_backward_never_reaches_adam(self):
        param = torch.nn.Parameter(torch.zeros(2))
        optimizer = torch.optim.Adam([param])
        loss = param.sqrt().sum()  # finite forward, infinite derivative at zero
        with patch.object(optimizer, 'step', wraps=optimizer.step) as step:
            with self.assertRaisesRegex(FloatingPointError, 'encoder.weight'):
                backward_and_step(loss, optimizer, [('encoder.weight', param)])
            step.assert_not_called()
        self.assertFalse(optimizer.state)
        torch.testing.assert_close(param, torch.zeros(2))

    def test_large_gradients_are_clipped_across_encoder_and_decoder(self):
        encoder = torch.nn.Parameter(torch.ones(2))
        decoder = torch.nn.Parameter(torch.ones(2))
        optimizer = torch.optim.Adam([{'params': [encoder]}, {'params': [decoder]}])
        norm = backward_and_step(10000 * (encoder.sum() + decoder.sum()), optimizer,
                                 [('encoder', encoder), ('decoder', decoder)], .5)
        self.assertAlmostEqual(norm.item(), 20000.)
        self.assertLessEqual(torch.cat([encoder.grad, decoder.grad]).norm().item(), .500001)
        self.assertTrue(encoder.isfinite().all() and decoder.isfinite().all())
        self.assertEqual(len(optimizer.state), 2)

    def test_normal_update_matches_unclipped_adam(self):
        guarded = torch.nn.Parameter(torch.tensor([.1, .2]))
        reference = torch.nn.Parameter(guarded.detach().clone())
        opt = torch.optim.Adam([guarded], lr=.001)
        ref_opt = torch.optim.Adam([reference], lr=.001)
        backward_and_step(guarded.square().sum(), opt, [('weight', guarded)])
        reference.square().sum().backward()
        ref_opt.step()
        torch.testing.assert_close(guarded, reference)
        torch.testing.assert_close(opt.state[guarded]['exp_avg'], ref_opt.state[reference]['exp_avg'])

    def test_finite_gradients_with_overflowing_norm_match_float64_clipping(self):
        # 1e19 overflows the combined norm; larger values overflow each tensor norm.
        for magnitude in (1e19, 1e30, 3e38):
            with self.subTest(magnitude=magnitude):
                params = [torch.nn.Parameter(torch.zeros(2)) for _ in range(2)]
                reference = [torch.nn.Parameter(torch.zeros(2, dtype=torch.float64)) for _ in range(2)]
                gradients = [torch.tensor([magnitude, -magnitude]),
                             torch.tensor([-magnitude, magnitude])]
                optimizer = torch.optim.Adam([{'params': [p]} for p in params], lr=.001)
                ref_optimizer = torch.optim.Adam(reference, lr=.001)
                for param, grad in zip(reference, gradients):
                    param.grad = grad.double()
                expected_norm = torch.nn.utils.clip_grad_norm_(reference, .5, error_if_nonfinite=True)
                ref_optimizer.step()
                loss = sum((param * grad).sum() for param, grad in zip(params, gradients))
                logger = logging.getLogger('test.training_stability')
                with self.assertLogs(logger, 'WARNING') as logs:
                    norm = backward_and_step(loss, optimizer, [('encoder', params[0]), ('decoder', params[1])], .5,
                                             logger=logger, context='train epoch=6 batch=4683')
                self.assertIn('train epoch=6 batch=4683', '\n'.join(logs.output))
                self.assertIn('encoder', '\n'.join(logs.output))
                self.assertIn('decoder', '\n'.join(logs.output))
                torch.testing.assert_close(norm, expected_norm)
                self.assertLessEqual(torch.cat([p.grad.double() for p in params]).norm().item(), .500001)
                for actual, expected in zip(params, reference):
                    torch.testing.assert_close(actual.grad, expected.grad.float())
                    torch.testing.assert_close(actual, expected.float())
                    self.assertTrue(optimizer.state[actual]['exp_avg_sq'].isfinite().all())
                    self.assertGreater(optimizer.state[actual]['exp_avg_sq'].sum().item(), 0.)

    def test_nan_gradients_are_rejected_without_clipping_other_parameters(self):
        invalid = torch.nn.Parameter(torch.zeros(2))
        finite = torch.nn.Parameter(torch.zeros(2))
        optimizer = torch.optim.Adam([invalid, finite])
        # sqrt(0) multiplied by zero has a finite forward value and NaN backward.
        loss = (invalid.sqrt() * 0).sum() + (finite * 1e30).sum()
        with patch.object(optimizer, 'step', wraps=optimizer.step) as step:
            with self.assertRaisesRegex(FloatingPointError, 'invalid'):
                backward_and_step(loss, optimizer, [('invalid', invalid), ('finite', finite)])
            step.assert_not_called()
        torch.testing.assert_close(finite.grad, torch.full_like(finite, 1e30))
        self.assertFalse(optimizer.state)

    def test_unrelated_clipping_errors_are_not_treated_as_norm_overflow(self):
        param = torch.nn.Parameter(torch.zeros(2))
        optimizer = torch.optim.Adam([param])
        with patch('torch.nn.utils.clip_grad_norm_', side_effect=RuntimeError('unsupported operation')):
            with self.assertRaisesRegex(RuntimeError, 'unsupported operation'):
                backward_and_step(param.sum(), optimizer, [('weight', param)])
        self.assertFalse(optimizer.state)

    def test_predictions_are_checked_and_batch_context_is_logged(self):
        logger = logging.getLogger('test.training_stability')
        with self.assertLogs(logger, 'ERROR') as logs:
            with self.assertRaisesRegex(FloatingPointError, 'predictions'):
                with numerical_error_context(logger, 'train epoch=9 batch=123 pids=P0'):
                    require_finite({'predictions': torch.tensor([torch.nan])})
        self.assertIn('epoch=9 batch=123 pids=P0', '\n'.join(logs.output))


class VectorNormRegressionTests(unittest.TestCase):
    def test_zero_and_small_vectors_have_finite_gradients(self):
        for magnitude in [0., 1e-12, 1e-5, 1.]:
            vectors = torch.full((4, 3, 3), magnitude, requires_grad=True)
            output = VectorLayerNorm(3)(vectors)
            output.sum().backward()
            self.assertTrue(output.isfinite().all())
            self.assertTrue(vectors.grad.isfinite().all())

    def test_fifteen_layer_structure_encoder_with_masked_nodes(self):
        torch.manual_seed(0)
        graph = dgl.graph((torch.arange(6).repeat_interleave(6), torch.arange(6).repeat(6)))
        graph.ndata['s'] = torch.randn(6, 29)
        graph.ndata['v'] = torch.randn(6, 3, 3)
        graph.edata['s'] = torch.randn(36, 32)
        graph.edata['v'] = torch.randn(36, 1, 3)
        encoder = StructEncoder(global_dims=8, part_dims=4, in_dims=(29, 3),
                                out_dims=(29, 3), edge_dims=(32, 1),
                                global_refer_nodes=3, num_layers=15, dropout=.5)
        glob, parts = encoder(graph, torch.tensor([0., 0., 1., 1., 0., 1.]))
        loss = (glob * torch.randn_like(glob)).sum() + (parts[0] * torch.randn_like(parts[0])).sum()
        optimizer = torch.optim.Adam(encoder.parameters())
        norm = backward_and_step(loss, optimizer, encoder.named_parameters())
        self.assertTrue(norm.isfinite())
        self.assertTrue(all(p.isfinite().all() for p in encoder.parameters()))


if __name__ == '__main__':
    unittest.main()
