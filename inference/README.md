# Inference

The same script supports SFT and pretraining checkpoints. SFT mode is the default:

```bash
uv run python inference/generate.py \
  --checkpoint /path/to/checkpoint.pt \
  --tokenizer /path/to/tokenizer-artifact \
  --user-input "भारत की राजधानी क्या है?" \
  --max-new-tokens 128
```

For raw completion from a pretraining checkpoint, select pretraining mode:

```bash
uv run python inference/generate.py \
  --mode pretrain \
  --checkpoint /path/to/checkpoint.pt \
  --tokenizer /path/to/tokenizer-artifact \
  --prompt "Once upon a time" \
  --max-new-tokens 128
```

Generation samples with `temperature=0.8`, `top-k=64`, `top-p=0.95`, and a fixed seed of
`1337` by default. Use `--temperature 0` for greedy decoding, `--top-k 0` or `--top-p 1` to
disable either sampling filter, and `--no-early-stopping` to continue after EOS.

In SFT mode, the user input is rendered as `BOS <|user|>\n…\n<|assistant|>\n`, so generation
begins at the assistant turn exactly as it did during SFT. `--prompt` remains accepted as an
alias for `--user-input` in this mode. In pretraining mode, `--prompt` is encoded directly as
`BOS prompt`; the tokenizer does not need the SFT protocol tokens.

The model configuration is read from the checkpoint. Prompts that exceed its context length are
trimmed from the left, retaining the most recent tokens. This first implementation recomputes the
full retained context for every generated token; it is correct but does not yet use a KV cache.
