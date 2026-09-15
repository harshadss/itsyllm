# Inference

Generate an assistant response from an SFT checkpoint:

```bash
uv run python inference/generate.py \
  --checkpoint /path/to/checkpoint.pt \
  --tokenizer /path/to/tokenizer-artifact \
  --user-input "भारत की राजधानी क्या है?" \
  --max-new-tokens 128
```

Generation samples with `temperature=0.8`, `top-k=50`, `top-p=0.95`, and a fixed seed of
`1337` by default. Use `--temperature 0` for greedy decoding, `--top-k 0` or `--top-p 1` to
disable either sampling filter, and `--no-early-stopping` to continue after EOS.

The user input is rendered as `BOS <|user|>\n…\n<|assistant|>\n`, so generation begins at the
assistant turn exactly as it did during SFT. The legacy `--prompt` spelling is accepted as an
alias for `--user-input`.

The model configuration is read from the checkpoint. Prompts that exceed its context length are
trimmed from the left, retaining the most recent tokens. This first implementation recomputes the
full retained context for every generated token; it is correct but does not yet use a KV cache.
