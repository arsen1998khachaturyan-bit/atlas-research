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
