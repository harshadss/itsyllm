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

Estimate the available training tokens before packaging (the estimate uses the same per-document
`BOS + text + EOS` convention):

```bash
uv run python scripts/estimate_parquet_tokens.py \
  --tokenizer artifacts/tokenizers/sangraha_verified_sample_unigram_v1 \
  /path/to/sangraha-verified-eng.parquet /path/to/sangraha-verified-indic.parquet
```

To materialize the first 1 million train rows from each requested English Ultra-FineWeb-L3 subset:

```bash
uv run python scripts/download_ultra_fineweb_l3.py
```

The resulting Parquet files contain only the source `content` normalized to a `text` column. The
source `uid` is intentionally omitted: it is not used for training and is a high-cardinality UUID
that adds storage. Pass `--keep-uid` only when provenance or later deduplication needs it.

```bash
uv run python scripts/package_parquet_dataset.py \
  --tokenizer artifacts/tokenizers/sangraha_verified_sample_unigram_v1 \
  --output artifacts/datasets/sangraha-eng.bin \
  /path/to/sangraha-verified-eng.parquet

cp configs/training/example.toml configs/training/first-run.toml
# Edit configs/training/first-run.toml for the dataset, tokenizer, and run settings.
uv run python training/pretrain_lm.py --config configs/training/first-run.toml
```

`extra-small` is the first 8,192-token model: 16 layers, hidden size 768, 12 query heads of 64
dimensions each, and one KV head (MQA). Use `extra-small-gqa` for three KV heads (one KV head per
four query heads), or `smoke` for a short functional test. The TOML separates model, data,
training, optimizer, scheduler, checkpointing, and logging settings; paths are relative to the
TOML file. The CLI accepts `--resume`, `--output-dir`, and `--wandb-mode` as invocation-only
overrides. Set `[model].repeat_blocks = true` to apply each unique decoder block twice
consecutively with shared weights, doubling effective depth without increasing parameter count.
The trainer uses BF16 autocast, FP32 parameters and optimizer state, activation checkpointing,
`torch.compile`, pinned-memory data loading, and one rolling checkpoint at `checkpoint.pt`.
Checkpoints default to once per 500 million training tokens to limit disk use.
