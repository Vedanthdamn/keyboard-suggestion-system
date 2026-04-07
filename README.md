# Keyboard Suggestion System

## Problem Statement
Autocomplete quality drops quickly on technical queries when the context is short or unseen. This project builds a production-style next-word suggestion system for Stack Overflow-like search queries and returns the top likely continuations for a partial query.

## Approach Overview
The system combines statistical language modeling with retrieval fallback:

- Trigram language model for context-aware next-word prediction.
- Kneser-Ney smoothing for robust probability estimates on sparse n-grams.
- TF-IDF fallback for low-confidence or unseen contexts.
- Optional tag-aware reranking and hybrid GPT-assisted inference in the notebook pipeline.

## Dataset Description
- Stack Overflow title dataset (`data/raw/train.csv`) used as training corpus.
- Unigram frequency dataset (`data/raw/unigram_freq.csv`) used as prior signal for reranking and safe fallback.

## System Design
### 1. Preprocessing
- Lowercase normalization.
- Punctuation/character cleanup.
- Deduplication of titles.
- Length filtering to keep query-like sequences.

### 2. Model Training
- Trigram model (`n=3`) built with Kneser-Ney smoothing.
- Serialized artifacts for deployment (`models/*.pkl`).

### 3. Prediction Logic
- Primary path: context-based trigram continuation scoring.
- Secondary path: TF-IDF nearest-title fallback when trigram evidence is weak.
- Ranking blend with unigram priors and optional tag signals.

### 4. Hybrid Ranking Mechanism
- `predict_hybrid()` merges statistical and GPT-derived suggestions.
- Rank-fusion weighting keeps deterministic top-N outputs in deployment mode.

## Evaluation
Core metrics used in this repository:

- **Top-5 Accuracy**: whether the true next word appears in top 5 suggestions.
- **Top-1 Accuracy**: whether the first suggestion is correct.

Recent validated run (standalone `app.py` on a 500-sample test split):
- Top-5 Accuracy: **64.4%**
- Top-1 Accuracy: **49.0%**

## Example Outputs
Representative examples from `outputs/test_cases.json`:

- `machine learning` -> `['algorithm', 'for', 'algorithms', 'machine', 'string']`
- `python` -> `['how', 'code', 'python', 'script', 'error']`
- `deep` -> `['learning', 'copy', 'depth', 'the', 'of']`
- `data science` -> `['general', 'the', 'to', 'method', 'of']`
- `how to` -> `['get', 'use', 'make', 'the', 'last']`
- `javascript` -> `['to', 'and', 'css', 'code', 'get']`
- `neural network` -> `['in', 'for', 'with', 'without', 'multiple']`
- `sql` -> `['server', 'query', 'to', 'not', 'and']`
- `git` -> `['repository', 'commit', 'lab', 'merge', 'code']`
- `pandas dataframe` -> `['in', 'column', 'columns', 'using', 'csv']`
- `react` -> `['native', 'the', 'and', 'how', 'can']`
- `sorting algorithm` -> `['in', 'for', 'on', 'the', 'strings']`

## Strengths and Limitations
### Strengths
- Strong performance on structured technical multi-word contexts.
- Deterministic ranking and stable deployment behavior.
- Graceful degradation via fallback strategies.

### Limitations
- Accuracy drops for nonsense/very-short unseen inputs.
- Domain drift requires periodic retraining.
- Generative augmentation (if enabled) can introduce noisy candidates.

## Improvements Tried During Development
- Discount and smoothing tuning for trigram modeling.
- Unigram-prior reranking with rarity penalty.
- Tag-aware score blending.
- Data augmentation strategies.
- Prediction path optimizations for runtime stability.

## How to Run
### 1. Setup
```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

### 2. Run Notebook
```bash
HF_HUB_DISABLE_SSL_VERIFICATION=1 .venv/bin/python -m jupyter nbconvert \
	--to notebook --execute --inplace \
	--ExecutePreprocessor.timeout=0 \
	notebooks/keyword_suggestion.ipynb
```

### 3. Run Standalone Inference
```bash
.venv/bin/python app.py "machine learning"
```

## Optional Demo
The repository currently focuses on notebook + CLI inference. If a UI/demo frontend is added later, it can consume `predict_hybrid()` from `app.py` directly.

## Model Weights
Large model artifacts (especially GPT checkpoints) are excluded from Git where appropriate.

- Google Drive (placeholder): `https://drive.google.com/your-model-link-here`

## Repository Structure
```
.
├── app.py
├── requirements.txt
├── notebooks/
│   └── keyword_suggestion.ipynb
├── models/
│   ├── ngram_weights.pkl
│   ├── ngram_weights_v3.pkl
│   ├── tfidf_vectorizer.pkl
│   ├── tfidf_matrix.pkl
│   ├── tag_model.pkl
│   └── cleaned_titles.pkl
└── outputs/
    └── test_cases.json
```
