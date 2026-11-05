"""Finite-value checks and guarded optimizer updates for training."""

from contextlib import contextmanager
import math

import torch


@contextmanager
def numerical_error_context(logger, context):
    """Record the first failing batch without swallowing the original exception."""
    try:
        yield
    except (ValueError, RuntimeError, FloatingPointError):
        logger.exception("Training/evaluation failed: %s", context)
        raise


def require_finite(tensors):
    for name, tensor in tensors.items():
        if not torch.isfinite(tensor).all():
            raise FloatingPointError(f"Non-finite {name}; stopping before optimizer update/metrics")


def backward_and_step(loss, optimizer, named_parameters, max_grad_norm=1.0, *, logger=None, context=""):
    """Reject non-finite gradients before Adam can corrupt parameters or moments.

    Returns the gradient norm before clipping. ``named_parameters`` must include
    every optimizer parameter (including the reconstruction decoder, if enabled).
    If finite gradients overflow the native norm reduction, recompute the norm
    in float64 before clipping. Actual NaN/Inf gradients still abort the update.
    """
    if not math.isfinite(max_grad_norm) or max_grad_norm <= 0:
        raise ValueError("max_grad_norm must be finite and positive")
    require_finite({"total loss": loss})
    loss.backward()
    named_parameters = list(named_parameters)
    parameters = [param for _, param in named_parameters if param.grad is not None]
    try:
        grad_norm = torch.nn.utils.clip_grad_norm_(
            parameters, max_grad_norm, error_if_nonfinite=True)
    except RuntimeError as exc:
        if "is non-finite, so it cannot be clipped" not in str(exc):
            raise
        bad_names = [name for name, param in named_parameters
                     if param.grad is not None and not torch.isfinite(param.grad).all()]
        if bad_names:
            raise FloatingPointError(
                "Non-finite gradient norm; optimizer.step() was not called. "
                f"Parameters: {', '.join(bad_names[:20])}"
            ) from exc
        # error_if_nonfinite=True raises before modifying any gradients. Keep
        # that fast path normally; only pay for float64 reductions on overflow.
        first_device = parameters[0].grad.device
        norms = [torch.linalg.vector_norm(param.grad.detach(), dtype=torch.float64).to(first_device)
                 for param in parameters]
        grad_norm = torch.linalg.vector_norm(torch.stack(norms))
        if not torch.isfinite(grad_norm):
            raise FloatingPointError(
                "Gradient norm overflowed even in float64; optimizer.step() was not called"
            ) from exc
        clip_coef = (max_grad_norm / (grad_norm + 1e-6)).clamp(max=1.0)
        with torch.no_grad():
            for param in parameters:
                param.grad.mul_(clip_coef.to(param.grad.device))
        if logger is not None:
            names = [name for name, param in named_parameters if param.grad is not None]
            largest = sorted(zip(names, [norm.item() for norm in norms]),
                             key=lambda item: item[1], reverse=True)[:5]
            logger.warning(
                "Gradient norm overflow recovered using float64: %s; norm=%g; clip=%g; "
                "largest parameter gradient norms=%s", context, grad_norm.item(), max_grad_norm, largest)
    optimizer.step()
    return grad_norm.detach()
