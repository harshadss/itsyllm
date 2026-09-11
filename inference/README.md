# Inference

Generate text from any checkpoint produced by `training/pretrain_lm.py`:

```bash
uv run python inference/generate.py \
  --checkpoint /path/to/checkpoint.pt \
  --tokenizer /path/to/tokenizer-artifact \
  --prompt "भारत की राजधानी" \
  --max-new-tokens 128
```

Generation samples with `temperature=0.8`, `top-k=50`, `top-p=0.95`, and a fixed seed of
`1337` by default. Use `--temperature 0` for greedy decoding, `--top-k 0` or `--top-p 1` to
disable either sampling filter, and `--no-early-stopping` to continue after EOS.

The model configuration is read from the checkpoint. Prompts that exceed its context length are
trimmed from the left, retaining the most recent tokens. This first implementation recomputes the
full retained context for every generated token; it is correct but does not yet use a KV cache.
