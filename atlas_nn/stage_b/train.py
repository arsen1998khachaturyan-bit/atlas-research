from __future__ import annotations

import copy

import numpy as np
import torch
from torch import nn


def train_mlp(
    model: nn.Module,
    x_train: np.ndarray,
    y_train: np.ndarray,
    *,
    epochs: int = 200,
    lr: float = 1e-2,
) -> dict:
    """Trains in place. Returns a small history dict (final train loss/acc)."""
    x = torch.as_tensor(x_train)
    y = torch.as_tensor(y_train)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        logits = model(x)
        loss = loss_fn(logits, y)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        final_logits = model(x)
        final_loss = float(loss_fn(final_logits, y))
        final_acc = float((final_logits.argmax(dim=1) == y).float().mean())

    return {"epochs": epochs, "final_train_loss": final_loss, "final_train_accuracy": final_acc}


def evaluate(model: nn.Module, x: np.ndarray, y: np.ndarray) -> dict:
    model.eval()
    with torch.no_grad():
        logits = model(torch.as_tensor(x))
        labels = torch.as_tensor(y)
        acc = float((logits.argmax(dim=1) == labels).float().mean())
    return {"accuracy": acc, "logits": logits.numpy()}


def snapshot(model: nn.Module) -> dict:
    return copy.deepcopy(model.state_dict())


def load_snapshot(model: nn.Module, state: dict) -> None:
    model.load_state_dict(state)
