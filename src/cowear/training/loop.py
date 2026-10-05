"""Reusable supervised training and validation loop.

Dataset layouts, model outputs, and losses remain model-specific callbacks;
epoch accounting, early stopping, best-state capture, and history are shared.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Callable

import torch


Step = Callable[[torch.nn.Module, Any, torch.device, torch.optim.Optimizer | None, int], float]


@dataclass
class FitResult:
    best_state: dict[str, torch.Tensor]
    best_epoch: int
    best_val_loss: float
    history: list[dict[str, float | int]]


def fit_supervised(
    model: torch.nn.Module,
    train_loader,
    val_loader,
    device: torch.device,
    optimizer: torch.optim.Optimizer,
    train_step: Step,
    val_step: Step,
    epochs: int,
    patience: int,
    log_prefix: str,
    min_delta: float = 0.0,
) -> FitResult:
    best_loss = float("inf")
    best_epoch = 0
    stale = 0
    best_state: dict[str, torch.Tensor] | None = None
    history: list[dict[str, float | int]] = []
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = float(train_step(model, train_loader, device, optimizer, epoch))
        model.eval()
        with torch.inference_mode():
            val_loss = float(val_step(model, val_loader, device, None, epoch))
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        print(f"[{log_prefix}] epoch={epoch:02d} train={train_loss:.6f} val={val_loss:.6f}", flush=True)
        if val_loss < best_loss - min_delta:
            best_loss = val_loss
            best_epoch = epoch
            stale = 0
            best_state = {key: copy.deepcopy(value.detach().cpu()) for key, value in model.state_dict().items()}
        else:
            stale += 1
            if stale >= patience:
                break
    if best_state is None:
        raise RuntimeError("training produced no validation checkpoint")
    return FitResult(best_state, best_epoch, best_loss, history)
