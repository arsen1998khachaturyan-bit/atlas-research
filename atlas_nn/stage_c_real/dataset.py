from __future__ import annotations

import numpy as np

# Self-authored, not fetched from any external corpus (same choice as
# atlas_nn.stage_c_lite.dataset, for the same reason: no copyright/licensing
# question about the eval text, and it's used only to drive forward passes,
# never to train anything here).
EVAL_SENTENCES = [
    "The train left the station a few minutes after sunrise.",
    "She placed the old photograph carefully back into its frame.",
    "A soft rain fell over the empty parking lot all night.",
    "He finally finished reading the book he started last winter.",
    "The committee will announce its decision sometime next week.",
    "Children played in the park until the streetlights came on.",
    "The recipe called for two cups of flour and a pinch of salt.",
    "Engineers spent months testing the bridge before it opened to traffic.",
]


def build_eval_batch(tokenizer, max_length: int = 24) -> np.ndarray:
    encoded = tokenizer(
        EVAL_SENTENCES,
        truncation=True,
        max_length=max_length,
        padding="max_length",
        return_tensors="np",
    )
    return encoded["input_ids"].astype(np.int64)
