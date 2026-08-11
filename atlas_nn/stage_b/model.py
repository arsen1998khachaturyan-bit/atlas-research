from __future__ import annotations

import torch
from torch import nn


def build_mlp(
    seed: int,
    input_dim: int = 32,
    hidden_dim: int = 64,
    output_dim: int = 2,
    n_hidden_layers: int = 2,
) -> nn.Module:
    """(n_hidden_layers + 1)-Linear-layer MLP: [Linear -> ReLU] * n_hidden_layers
    -> Linear. Default n_hidden_layers=2 reproduces the original 3-Linear-layer
    Stage B architecture. Small enough to train on CPU in seconds, per the
    mission's Stage B guidance, even at n_hidden_layers up to ~5-6.

    Caution (found empirically, see docs/RESEARCH_LOG.md Experiment 5): with
    plain PyTorch default init and no normalization, n_hidden_layers=5 on
    this scale produced a *degenerate* random-init network (near-zero output
    variance, single-class prediction for every input) -- vanishing-signal
    collapse from stacking unnormalized ReLU layers. Any experiment that
    compares "random init" vs. "trained" behavior at this depth should
    verify the random-init network isn't already degenerate before drawing
    conclusions from the comparison (e.g. check output logit std/variance).
    """
    if n_hidden_layers < 1:
        raise ValueError("n_hidden_layers must be >= 1")

    torch.manual_seed(seed)
    layers: list[nn.Module] = []
    in_dim = input_dim
    for _ in range(n_hidden_layers):
        layers.append(nn.Linear(in_dim, hidden_dim))
        layers.append(nn.ReLU())
        in_dim = hidden_dim
    layers.append(nn.Linear(in_dim, output_dim))
    return nn.Sequential(*layers)


def linear_layer_names(model: nn.Module) -> list[str]:
    return [name for name, module in model.named_modules() if isinstance(module, nn.Linear)]


def get_weight(model: nn.Module, layer_name: str) -> torch.Tensor:
    return dict(model.named_modules())[layer_name].weight


def set_weight(model: nn.Module, layer_name: str, weight) -> None:
    layer = dict(model.named_modules())[layer_name]
    with torch.no_grad():
        layer.weight.copy_(torch.as_tensor(weight, dtype=layer.weight.dtype))
