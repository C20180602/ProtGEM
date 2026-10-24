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


def backward_and_step(loss, optimizer, named_parameters, max_grad_norm=1.0):
    """Reject non-finite gradients before Adam can corrupt parameters or moments.

    Returns the gradient norm before clipping. ``named_parameters`` must include
    every optimizer parameter (including the reconstruction decoder, if enabled).
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
        bad_names = [name for name, param in named_parameters
                     if param.grad is not None and not torch.isfinite(param.grad).all()]
        detail = ", ".join(bad_names[:20]) or "individual gradients finite, but total norm overflowed"
        raise FloatingPointError(
            f"Non-finite gradient norm; optimizer.step() was not called. Parameters: {detail}"
        ) from exc
    optimizer.step()
    return grad_norm.detach()
