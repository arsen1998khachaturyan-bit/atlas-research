from __future__ import annotations

from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from atlas_nn.stage_c_lite.dataset import generate_sentiment_dataset

# Experiment 31: does the depth-gradient reversal's magnitude track how
# much post-initialization training a model underwent? Fine-tunes a base
# model on a fixed, self-authored corpus (reusing Stage C-lite's sentiment
# sentence generator -- no external dataset, same licensing rationale as
# every other text used in this project) for a chosen number of optimizer
# steps, then saves a checkpoint via save_pretrained so the existing
# atlas_nn.stage_c_real infra (which takes any HF model name OR local
# directory path) can load it unmodified.


def build_finetune_corpus(n_examples: int = 4000, seed: int = 0) -> list[str]:
    examples = generate_sentiment_dataset(n_examples, seed=seed)
    return [text for text, _label in examples]


def finetune_checkpoints(
    base_model_name: str,
    output_dirs: dict[int, str],
    *,
    batch_size: int = 4,
    max_length: int = 32,
    learning_rate: float = 5e-5,
    seed: int = 0,
    corpus: list[str] | None = None,
) -> None:
    """Fine-tunes `base_model_name` on `corpus` (defaults to Experiment
    31's narrow sentiment-template corpus for backward compatibility --
    pass a different corpus, e.g. Experiment 32's
    `atlas_nn.stage_c_real.diverse_corpus.generate_diverse_corpus`, to
    test how corpus properties other than length affect the result),
    saving a full checkpoint (model + tokenizer) at each step count in
    `output_dirs` (e.g. {20: "path/steps_20", 100: "path/steps_100"}).
    Step counts must be given in ascending order -- training continues
    from where the previous checkpoint left off rather than restarting,
    so N checkpoints cost the same total compute as one run to the
    largest step count.
    """
    torch.manual_seed(seed)
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(base_model_name)
    model.train()

    if corpus is None:
        corpus = build_finetune_corpus(seed=seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    step_targets = sorted(output_dirs.keys())
    total_steps = step_targets[-1]
    step = 0
    corpus_pos = 0

    while step < total_steps:
        batch_texts = []
        for _ in range(batch_size):
            batch_texts.append(corpus[corpus_pos % len(corpus)])
            corpus_pos += 1
        encoded = tokenizer(
            batch_texts,
            truncation=True,
            max_length=max_length,
            padding="max_length",
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"]
        labels = input_ids.clone()
        labels[encoded["attention_mask"] == 0] = -100

        optimizer.zero_grad()
        loss = model(input_ids=input_ids, labels=labels).loss
        loss.backward()
        optimizer.step()
        step += 1

        if step in output_dirs:
            print(f"step={step:>5} loss={loss.item():.4f} -> saving {output_dirs[step]}", flush=True)
            save_dir = Path(output_dirs[step])
            save_dir.mkdir(parents=True, exist_ok=True)
            model.eval()
            model.save_pretrained(save_dir)
            tokenizer.save_pretrained(save_dir)
            model.train()
