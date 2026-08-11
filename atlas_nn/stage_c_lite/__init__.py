"""Stage C "lite": a small Transformer classifier trained from scratch on a
real (self-authored, no external download) sentiment task.

Mission Stage C calls for "manageable open pretrained models" -- this
session's network policy blocks huggingface.co (403, confirmed via the
egress proxy status endpoint), so a literal pretrained-checkpoint Stage C
is not reachable here. Per the user's choice, this module is the agreed
substitute: not a pretrained model, but a genuinely different architecture
(attention, not just MLP layers) trained on real English text (not
synthetic feature vectors), used to test whether the capacity-slack
findings from Stage B (docs/RESEARCH_LOG.md Experiments 3-7) transfer
beyond the MLP/XOR setting they were established on.

See docs/RESEARCH_LOG.md Experiment 8 for what was found.
"""
