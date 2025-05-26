# run_optim_gcg.py README

## Overview

`run_optim_gcg.py` is a script for performing adversarial attacks in Natural Language Processing (NLP) tasks. It generates adversarial text to deceive a target model by leveraging a pre-trained model and a classifier head, and evaluates the attack success rate (ASR).

## Features

- **Adversarial Attack**: Generates adversarial text to flip model predictions.
- **Evaluation Metrics**: Computes Attack Success Rate (ASR), text similarity, and perplexity.
## Requirements

- Python 3.11
- PyTorch
- Transformers
- NLTK
- SentenceTransformers

Install dependencies:
```bash
pip install torch transformers nltk sentence-transformers