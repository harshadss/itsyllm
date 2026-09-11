#!/usr/bin/env bash
# Copy commands from this file while in the repository root.
# Do not run this file directly: the first and full runs are intentionally separate.

# One-time W&B setup for the optional tracking commands below.
uv sync --extra wandb
uv run wandb login

# Sanity run: same configuration as the real run, but only about three optimizer
# updates. Check the log and GPU memory before starting the full run.
SANITY_LOG="logs/pretrain_extra_small_gqa_4096_sanity_$(date +%F_%H%M%S).log"
nohup uv run python -u training/pretrain_lm.py \
  --config configs/training/extra_small_gqa_4096_sanity.toml \
  > "$SANITY_LOG" 2>&1 < /dev/null &

# Monitor a running sanity or full training run.
tail -f "$SANITY_LOG"
nvidia-smi -l 2

# Full run: start this only after the sanity run finishes cleanly. The context is
# 4,096, model is extra-small GQA, and the run trains for 3B tokens.
FULL_RUN_LOG="logs/pretrain_extra_small_gqa_4096_v1_$(date +%F_%H%M%S).log"
nohup uv run python -u training/pretrain_lm.py \
  --config configs/training/extra_small_gqa_4096_full.toml \
  > "$FULL_RUN_LOG" 2>&1 < /dev/null &

# Monitor the full run.
tail -f "$FULL_RUN_LOG"

# Inference: run after stopping training on this single-GPU machine, or on a
# different GPU. The rolling checkpoint is first created at 100M training tokens.
uv run python inference/generate.py \
  --checkpoint artifacts/checkpoints/extra_small_gqa_4096_v1/checkpoint.pt \
  --tokenizer artifacts/tokenizers/sangraha_ultrafineweb_l3_en_indic_unigram_v1 \
  --prompt "भारत की राजधानी" \
  --max-new-tokens 128

# Resume the full run after an interruption (replace the launch command above).
RESUME_LOG="logs/pretrain_extra_small_gqa_4096_v1_resume_$(date +%F_%H%M%S).log"
nohup uv run python -u training/pretrain_lm.py \
  --config configs/training/extra_small_gqa_4096_full.toml \
  --output-dir artifacts/checkpoints/extra_small_gqa_4096_v1 \
  --resume artifacts/checkpoints/extra_small_gqa_4096_v1/checkpoint.pt \
  > "$RESUME_LOG" 2>&1 < /dev/null &
