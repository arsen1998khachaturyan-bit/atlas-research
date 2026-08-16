from __future__ import annotations

import torch
from torch import nn


def build_mlp(
    seed: int,
    input_dim: int = 32,
    hidden_dim: int = 64,
    output_dim: int = 2,
    n_hidden_layers: int = 2,
    use_layernorm: bool = False,
    use_frozen_input_projection: bool = False,
) -> nn.Module:
    """(n_hidden_layers + 1)-Linear-layer MLP: [Linear -> ReLU (-> LayerNorm)]
    * n_hidden_layers -> Linear. Default n_hidden_layers=2 reproduces the
    original 3-Linear-layer Stage B architecture. Small enough to train on
    CPU in seconds, per the mission's Stage B guidance, even at
    n_hidden_layers up to ~5-6.

    `use_layernorm=True` inserts `nn.LayerNorm(hidden_dim)` after each
    hidden block's ReLU -- added to test whether Experiment 11's
    Transformer finding (LayerNorm amplifies the post-training
    behavioral-robustness gain and decouples it from the layer's own
    effective-rank shrinkage) is a property of LayerNorm specifically, or
    something particular to attention/Transformer architectures (see
    docs/RESEARCH_LOG.md Experiment 13). `linear_layer_names` below only
    picks up `nn.Linear` submodules, so LayerNorm's own parameters are
    never treated as a compressible "layer" here -- consistent with how
    the Transformer experiments handle LayerNorm.

    `use_frozen_input_projection=True` prepends a frozen (non-trainable,
    orthogonally initialized, information-preserving) `nn.Linear(input_dim,
    input_dim, bias=False)` before the trainable stack -- added to test
    whether the input layer's distinctive, unexplained flat-to-negative
    post-training compressibility (open since Experiment 4; noise-fraction
    and effective-rank hypotheses both ruled out, Experiments 7, 12, 15) is
    about literally being the first layer to see *raw, untransformed* task
    input, rather than about capacity/slack. With this on, the first
    *trainable* layer (index 1 in `linear_layer_names`, not index 0 -- the
    frozen projection is index 0 and never changes, so its own
    random-init/trained comparison is trivially flat and should be
    excluded from analysis) sees a fixed linear transform of the raw input
    instead of the raw input itself, while everything else about the
    architecture is unchanged. See docs/RESEARCH_LOG.md Experiment 16.

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

    if use_frozen_input_projection:
        projection = nn.Linear(input_dim, input_dim, bias=False)
        nn.init.orthogonal_(projection.weight)
        for param in projection.parameters():
            param.requires_grad_(False)
        layers.append(projection)

    in_dim = input_dim
    for _ in range(n_hidden_layers):
        layers.append(nn.Linear(in_dim, hidden_dim))
        layers.append(nn.ReLU())
        if use_layernorm:
            layers.append(nn.LayerNorm(hidden_dim))
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
