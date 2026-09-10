# itsyllm

Attempts at building a very small LLM from scratch, integrating all the learnings. Planning to make
some interesting choices like small vocabulary, MQA instead of GQA. End goal will be an Indic friendly
LLM that can run on mobile devices.

## First pretraining run

Train a SentencePiece tokenizer first, then package Parquet text once. The packager treats every
non-empty Parquet row as one document and writes:

- `name.bin`: little-endian signed `int16` token IDs, stored directly without a header.
- `name.document_offsets.bin`: little-endian `uint64` start offsets, with one final sentinel offset.
- `name.json`: format, tokenizer, source, and token-count metadata.

Each document is stored as `BOS + encoded text + EOS`. The trainer memory-maps both binary files,
samples random fixed-length windows, resets RoPE positions at document starts, and uses
document-isolated causal attention. The loss for `EOS -> BOS` transitions is ignored.

```bash
uv run python scripts/package_parquet_dataset.py \
  --tokenizer artifacts/tokenizers/sangraha_verified_sample_unigram_v1 \
  --output artifacts/datasets/sangraha-eng.bin \
  /path/to/sangraha-verified-eng.parquet

uv run python training/pretrain_lm.py \
  --data artifacts/datasets/sangraha-eng.bin \
  --tokenizer artifacts/tokenizers/sangraha_verified_sample_unigram_v1 \
  --output-dir artifacts/checkpoints/first-run \
  --model-config small \
  --max-train-tokens 1000000000
```

`small` is the first 8,192-token model: 16 layers, hidden size 1024, 32 query heads, and one KV
head (MQA). Use `small_gqa` for four KV heads, or `smoke` for a short functional test. The trainer
uses BF16 autocast, FP32 parameters and optimizer state, activation checkpointing, `torch.compile`,
pinned-memory data loading, and one rolling checkpoint at `checkpoint.pt`. Checkpoints default to
once per 500 million training tokens to limit disk use.
