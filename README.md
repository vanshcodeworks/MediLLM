# MediLLM

A small language model built from scratch in PyTorch: tokenizer, model,
training loop and inference, with a Jupyter pipeline notebook.

Everything here is implemented directly rather than pulled from a library, so
the point is understanding each piece.

## Layout

```
config.py        hyperparameters
dataset.py       corpus loading and batching
tokenizer.py     vocabulary construction and encoding
model.py         transformer architecture (attention, FFN, blocks)
train.py         training loop
inference.py     sampling and generation
demo.py          runnable end-to-end demo
pipeline.ipynb   walkthrough notebook
```

## Setup

```bash
pip install torch numpy
```

Configure hyperparameters in `config.py`, then:

```bash
python demo.py
```

## Notes

- This is a research/scratch implementation, not a packaged library. There is no
  `requirements.txt` and no test suite.
- `pipeline.ipynb` is the best starting point if you want to follow the data
  from raw text through to generated output.