from __future__ import annotations

import torch
from torch import nn

# get_weight/set_weight/linear_layer_names are pure named_modules()/isinstance
# logic with no MLP-specific assumptions, so they work unchanged on this
# Transformer -- re-exported here rather than duplicated.
from atlas_nn.stage_b.model import get_weight, linear_layer_names, set_weight  # noqa: F401


class TransformerClassifier(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        d_ff: int = 128,
        max_len: int = 16,
        n_classes: int = 2,
        pad_idx: int = 0,
    ):
        super().__init__()
        self.pad_idx = pad_idx
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.pos_embedding = nn.Embedding(max_len, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.classifier = nn.Linear(d_model, n_classes)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        positions = torch.arange(input_ids.shape[1], device=input_ids.device).unsqueeze(0)
        x = self.embedding(input_ids) + self.pos_embedding(positions)
        pad_mask = input_ids == self.pad_idx
        x = self.encoder(x, src_key_padding_mask=pad_mask)

        keep_mask = (~pad_mask).unsqueeze(-1).to(x.dtype)
        pooled = (x * keep_mask).sum(dim=1) / keep_mask.sum(dim=1).clamp(min=1.0)
        return self.classifier(pooled)


def build_transformer_classifier(
    seed: int,
    vocab_size: int,
    d_model: int = 64,
    n_heads: int = 4,
    n_layers: int = 2,
    d_ff: int = 128,
    max_len: int = 16,
    n_classes: int = 2,
) -> TransformerClassifier:
    torch.manual_seed(seed)
    return TransformerClassifier(
        vocab_size=vocab_size,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        d_ff=d_ff,
        max_len=max_len,
        n_classes=n_classes,
    )


class NoMixingAttention(nn.Module):
    """Drop-in replacement for nn.MultiheadAttention with no cross-token
    mixing: applies a single per-token linear projection instead of
    attention-weighted averaging over other tokens' values. Keeps the same
    `out_proj` attribute name (and therefore the same
    `self_attn.out_proj` layer-discovery naming) as the real attention
    module, so it's directly comparable under the existing analysis
    scripts. Built to test whether attention itself (rather than
    LayerNorm/residual/depth) is what determines the *sign* of LayerNorm's
    effect on gain magnitude (docs/RESEARCH_LOG.md Experiment 13's open
    question). Necessarily has fewer parameters than real attention (no
    in_proj_weight) -- an inherent, expected consequence of removing
    attention, not a bug to correct for."""

    def __init__(self, d_model: int):
        super().__init__()
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, query, key, value, key_padding_mask=None, need_weights=False):
        del key, value, key_padding_mask, need_weights
        return self.out_proj(query), None


class MatchedParamNoMixingAttention(nn.Module):
    """Disentangles Experiment 14's `NoMixingAttention` ablation, which
    simultaneously removed cross-token mixing AND ~3/4 of attention's
    parameters (no in_proj_weight). This variant keeps the same total
    parameter count as real `nn.MultiheadAttention` (in_proj_weight/bias
    stored as raw Parameters, exactly like `nn.MultiheadAttention` itself
    does -- not as an `nn.Linear` submodule, so `linear_layer_names`
    still only picks up `out_proj` here, keeping the tracked-layer set
    identical across all attention variants) but skips the actual
    softmax(QK^T)V cross-token mixing step: the value projection is
    passed straight to `out_proj`, per token, with no mixing across the
    sequence. Isolates "no mixing" from "fewer parameters" (docs/
    RESEARCH_LOG.md Experiment 17)."""

    def __init__(self, d_model: int):
        super().__init__()
        self.in_proj_weight = nn.Parameter(torch.empty(3 * d_model, d_model))
        self.in_proj_bias = nn.Parameter(torch.zeros(3 * d_model))
        nn.init.xavier_uniform_(self.in_proj_weight)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, query, key, value, key_padding_mask=None, need_weights=False):
        del key, value, key_padding_mask, need_weights
        d_model = query.shape[-1]
        qkv = nn.functional.linear(query, self.in_proj_weight, self.in_proj_bias)
        v = qkv[..., 2 * d_model:]
        return self.out_proj(v), None


class AblationEncoderBlock(nn.Module):
    """Same shape as one nn.TransformerEncoderLayer block, but with
    attention, residual connections, and LayerNorm all independently
    toggleable -- needed because nn.TransformerEncoderLayer hardcodes all
    three. Built to test hypotheses about which architectural feature
    drives the Transformer's behavioral-robustness effect
    (docs/RESEARCH_LOG.md Experiments 10, 11, 13, 14)."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        use_residual: bool,
        use_layernorm: bool,
        use_attention: bool = True,
    ):
        super().__init__()
        self.use_residual = use_residual
        self.use_layernorm = use_layernorm
        if use_attention:
            self.self_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True, dropout=0.0)
        else:
            self.self_attn = NoMixingAttention(d_model)
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.activation = nn.ReLU()
        if use_layernorm:
            self.norm1 = nn.LayerNorm(d_model)
            self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.self_attn(x, x, x, key_padding_mask=key_padding_mask, need_weights=False)
        x = x + attn_out if self.use_residual else attn_out
        if self.use_layernorm:
            x = self.norm1(x)

        ff_out = self.linear2(self.activation(self.linear1(x)))
        x = x + ff_out if self.use_residual else ff_out
        if self.use_layernorm:
            x = self.norm2(x)
        return x


class AblationTransformerClassifier(nn.Module):
    """Drop-in analog of TransformerClassifier with the same Linear-layer
    naming (`encoder.layers.{i}.self_attn.out_proj` /
    `.linear1` / `.linear2` / `classifier`), for direct comparison with the
    baseline model under the existing analysis scripts."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        d_ff: int = 128,
        max_len: int = 16,
        n_classes: int = 2,
        pad_idx: int = 0,
        use_residual: bool = True,
        use_layernorm: bool = True,
        use_attention: bool = True,
    ):
        super().__init__()
        self.pad_idx = pad_idx
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.pos_embedding = nn.Embedding(max_len, d_model)
        self.encoder = nn.ModuleDict({
            "layers": nn.ModuleList([
                AblationEncoderBlock(d_model, n_heads, d_ff, use_residual, use_layernorm, use_attention)
                for _ in range(n_layers)
            ])
        })
        self.classifier = nn.Linear(d_model, n_classes)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        positions = torch.arange(input_ids.shape[1], device=input_ids.device).unsqueeze(0)
        x = self.embedding(input_ids) + self.pos_embedding(positions)
        pad_mask = input_ids == self.pad_idx

        for layer in self.encoder["layers"]:
            x = layer(x, key_padding_mask=pad_mask)

        keep_mask = (~pad_mask).unsqueeze(-1).to(x.dtype)
        pooled = (x * keep_mask).sum(dim=1) / keep_mask.sum(dim=1).clamp(min=1.0)
        return self.classifier(pooled)


def build_ablation_transformer_classifier(
    seed: int,
    vocab_size: int,
    d_model: int = 64,
    n_heads: int = 4,
    n_layers: int = 2,
    d_ff: int = 128,
    max_len: int = 16,
    n_classes: int = 2,
    use_residual: bool = True,
    use_layernorm: bool = True,
    use_attention: bool = True,
) -> AblationTransformerClassifier:
    torch.manual_seed(seed)
    return AblationTransformerClassifier(
        vocab_size=vocab_size,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        d_ff=d_ff,
        max_len=max_len,
        n_classes=n_classes,
        use_residual=use_residual,
        use_layernorm=use_layernorm,
        use_attention=use_attention,
    )
