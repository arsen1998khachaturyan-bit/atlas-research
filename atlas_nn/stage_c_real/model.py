from __future__ import annotations

import numpy as np
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers.pytorch_utils import Conv1D

# Stage C (real): a genuinely pretrained, openly-licensed small model, now
# that huggingface.co is reachable from this environment (previously
# blocked -- see docs/RESEARCH_LOG.md Experiment 8's network-policy note).
# distilgpt2 (82M params, 6 transformer blocks, GPT-2 architecture) was
# chosen for being small enough to run many compression sweeps on CPU in a
# single session while still being a real, non-toy pretrained checkpoint.
MODEL_NAME = "distilgpt2"

# GPT-2's Linear-equivalent sublayers are `transformers.pytorch_utils.Conv1D`
# (weight shape (in_features, out_features), the opposite orientation of
# nn.Linear) -- compression/reconstruction here treats weights as generic
# 2D arrays, so the orientation doesn't matter.
#
# A real 82M-parameter model cannot be swept layer-by-layer at the same
# statistical power as the tiny Stage B/C-lite models on CPU in one session
# (full budget search is a ~31-config sweep per layer per method family).
# Instead we test a fixed, depth-and-type-balanced subset: the first,
# middle, and last of distilgpt2's 6 blocks, times all 4 sublayer types --
# 12 layers total, chosen for coverage across depth rather than exhaustive
# coverage of every layer. This scope choice is deliberate and documented,
# not hidden.
TESTED_BLOCKS = (0, 3, 5)
SUBLAYER_SUFFIXES = ("attn.c_attn", "attn.c_proj", "mlp.c_fc", "mlp.c_proj")


def tested_layer_names() -> list[str]:
    return [
        f"transformer.h.{block}.{suffix}"
        for block in TESTED_BLOCKS
        for suffix in SUBLAYER_SUFFIXES
    ]


def linear_layer_names(model) -> list[str]:
    del model  # interface parity with atlas_nn.stage_b.model.linear_layer_names
    return tested_layer_names()


def load_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_pretrained():
    """The real, trained distilgpt2 checkpoint -- exactly one instance
    exists (there is no 'seed' for a fixed pretrained checkpoint)."""
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    model.eval()
    return model


def load_random_init(seed: int):
    """Same architecture and config as distilgpt2, freshly initialized
    (no training at all) -- the random-init comparison arm, analogous to
    Stage B/C-lite's snapshot-before-training but here there is no
    training step to snapshot before, so this directly instantiates the
    untrained network."""
    config = AutoConfig.from_pretrained(MODEL_NAME)
    torch.manual_seed(seed)
    model = AutoModelForCausalLM.from_config(config)
    # from_config's own init already consumed the seeded RNG state above;
    # re-apply it explicitly so successive seeds are guaranteed to diverge
    # rather than depending on how much RNG state construction consumed.
    model.apply(model._init_weights)
    model.eval()
    return model


def get_weight(model, layer_name: str) -> torch.Tensor:
    return dict(model.named_modules())[layer_name].weight


def set_weight(model, layer_name: str, weight: np.ndarray) -> None:
    module = dict(model.named_modules())[layer_name]
    with torch.no_grad():
        module.weight.copy_(torch.as_tensor(weight, dtype=module.weight.dtype))


def snapshot(model) -> dict:
    """Clones only the tested Conv1D layers' weights, not the whole 82M-
    parameter model (which includes a 38.6M-parameter tied embedding
    table that `set_weight` never touches) -- a full `state_dict()` deep
    copy on every one of the ~700+ rows in a budget search would dominate
    runtime for no benefit, since only one tested layer's weight ever
    changes between a `snapshot` and its matching `load_snapshot`."""
    modules = dict(model.named_modules())
    return {name: modules[name].weight.detach().clone() for name in tested_layer_names()}


def load_snapshot(model, state: dict) -> None:
    modules = dict(model.named_modules())
    with torch.no_grad():
        for name, weight in state.items():
            modules[name].weight.copy_(weight)


def evaluate(model, x_eval: np.ndarray, y_eval) -> dict:
    """Behavioral eval: teacher-forced next-token prediction over a fixed
    held-out batch of self-authored English sentences. `y_eval` is unused
    (kept for interface parity with atlas_nn.stage_b/stage_c_lite, whose
    `evaluate(model, x_eval, y_eval)` signature `run_layer_experiment` and
    `run_budget_search` call directly) -- targets are derived by shifting
    `x_eval` itself, since this is causal language modeling, not
    classification."""
    del y_eval
    model.eval()
    with torch.no_grad():
        input_ids = torch.as_tensor(x_eval, dtype=torch.long)
        logits = model(input_ids).logits
        predictions = logits[:, :-1, :].argmax(dim=-1)
        targets = input_ids[:, 1:]
        accuracy = float((predictions == targets).float().mean())
    return {"accuracy": accuracy, "logits": logits.numpy()}


def conv1d_module(model, layer_name: str) -> Conv1D:
    return dict(model.named_modules())[layer_name]
