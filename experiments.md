# Experiments & Learnings

Since jumping into final training run can be expensive and prone to mistakes, build in layers, document
learning at each step.

## Experiment 1 : Train tokenizer on subset of Sangraha

  1. Used dataste `ai4bharat/sangraha` to go multi-lingual from day 1.
  2. Start with a sample of 100k each Marathi, English and Hindi.
  3. Tokenizer trained with 16384 vocabulary, set with lot of special tokens upfront.

### Learnings

  1. Sentencepiece training needed almost 16 GB memory. That's worrisome with the later goal of pushing on
  large dataset. :-P. Need to study more how the memory can be kept bounded.
  2. Even on this sample, tokenizer is reasonably good. It learns reasonable Devanagari words splits like '▁क्या' or '▁राह'.
  3. Common Indic names like India, Pune, Delhi are learnt as single token.
  3. Interestingly, for https it learnt _http and s separately. 
  4. Importance of better data mixture is immediately clear: on this sample, word google gets split. So the text dataste
  is not very representative of Internet like text.