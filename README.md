# French-to-English Transformer

Notebook: `translation.ipynb`

## Provenance
This is my own university coursework, published here with permission to publish it. It is included as a portfolio sample.

## What it does
- Problem 1: word-vector analogy with spaCy `en_core_web_md`.
- Problem 2: trains a small encoder-decoder Transformer (2 layers per side, 4 heads, 128-dim embeddings, 10,000-token BPE vocabulary) for **French-to-English** translation on a 20,000-pair subset of `Helsinki-NLP/opus_books`, and compares a few sample outputs.
- Problem 3: inspects GPU tensor memory. Problem 4: reloads the saved model and computes BLEU.

## Limitations
This is a small model trained briefly. **Translation quality is limited**; the notebook's own findings describe the outputs as weak. It is a learning exercise, not a usable translator.

## Dependencies
- Dataset: `Helsinki-NLP/opus_books` (en-fr), downloaded through Hugging Face `datasets`.
- spaCy model `en_core_web_md`: installed by the notebook's first code cell (`pip install` and `spacy download`).
- Python packages: see `requirements.txt` (unpinned).
- Problem 4 loads the model and tokenizer that Problem 2 saves to `assign07_artifacts/`, so run cells in order. Those generated files are git-ignored and not published.

## Run
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook translation.ipynb
```
Internet access is needed on first run.

## Status
- **No fresh end-to-end run has been done, and reproducibility has not been verified.**
- Figures in the notebook's markdown come from the original coursework run and have not been re-checked here.
- The repository contains no data, outputs, weights, or secrets.
