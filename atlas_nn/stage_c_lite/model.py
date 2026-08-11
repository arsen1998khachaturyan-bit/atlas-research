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


class AblationEncoderBlock(nn.Module):
    """Same shape as one nn.TransformerEncoderLayer block, but with residual
    connections and/or LayerNorm independently toggleable -- needed because
    nn.TransformerEncoderLayer hardcodes both. Built to test the Experiment
    10 hypothesis that the Transformer's behavioral-robustness effect comes
    from residual/LayerNorm-mediated error absorption rather than from any
    single layer's own weight-matrix rank structure (docs/RESEARCH_LOG.md
    Experiment 10)."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        use_residual: bool,
        use_layernorm: bool,
    ):
        super().__init__()
        self.use_residual = use_residual
        self.use_layernorm = use_layernorm
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True, dropout=0.0)
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
    ):
        super().__init__()
        self.pad_idx = pad_idx
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.pos_embedding = nn.Embedding(max_len, d_model)
        self.encoder = nn.ModuleDict({
            "layers": nn.ModuleList([
                AblationEncoderBlock(d_model, n_heads, d_ff, use_residual, use_layernorm)
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
    )
