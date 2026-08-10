from __future__ import annotations

import torch
from torch import nn


def build_mlp(
    seed: int,
    input_dim: int = 32,
    hidden_dim: int = 64,
    output_dim: int = 2,
) -> nn.Module:
    """3-layer MLP: Linear -> ReLU -> Linear -> ReLU -> Linear. Small enough
    to train on CPU in seconds, per the mission's Stage B guidance."""
    torch.manual_seed(seed)
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, output_dim),
    )


def linear_layer_names(model: nn.Module) -> list[str]:
    return [name for name, module in model.named_modules() if isinstance(module, nn.Linear)]


def get_weight(model: nn.Module, layer_name: str) -> torch.Tensor:
    return dict(model.named_modules())[layer_name].weight


def set_weight(model: nn.Module, layer_name: str, weight) -> None:
    layer = dict(model.named_modules())[layer_name]
    with torch.no_grad():
        layer.weight.copy_(torch.as_tensor(weight, dtype=layer.weight.dtype))
