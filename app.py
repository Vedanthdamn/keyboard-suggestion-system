"""Inference module for keyword suggestions."""

import csv
import os
import pickle
import re
from collections import Counter, defaultdict

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR = os.path.join(BASE_DIR, "data", "raw")

NGRAM_PATH = os.path.join(MODELS_DIR, "ngram_weights_v3.pkl")
TFIDF_VECTORIZER_PATH = os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl")
TFIDF_MATRIX_PATH = os.path.join(MODELS_DIR, "tfidf_matrix.pkl")
TAG_MODEL_PATH = os.path.join(MODELS_DIR, "tag_model.pkl")
GPT2_DIR = os.path.join(MODELS_DIR, "gpt2_finetuned")
CLEANED_TITLES_PATH = os.path.join(MODELS_DIR, "cleaned_titles.pkl")
UNIGRAM_PATH = os.path.join(DATA_DIR, "unigram_freq.csv")


LM = None
TFIDF_VECTORIZER = None
TFIDF_MATRIX = None
TAG_TO_WORDS = None
TOKENIZER = None
GPT2_MODEL = None
DEVICE = None
TORCH = None

GPT2_AVAILABLE = False

CLEANED_TITLES = None
UNIGRAM_FREQ = None
UNIGRAM_WORDS = None


def clean_text(raw_text):
    text = str(raw_text).lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    words = text.split()
    if len(words) > 10:
        words = words[:10]
    return " ".join(words)


def _dedupe_keep_order(items):
    seen = set()
    result = []
    for item in items:
        if item and item not in seen:
            seen.add(item)
            result.append(item)
    return result


def _extract_next_words(title_tokens, query_tokens):
    if not query_tokens:
        return []

    span = len(query_tokens)
    candidates = []
    for i in range(0, len(title_tokens) - span):
        if title_tokens[i : i + span] == query_tokens:
            nxt = i + span
            if nxt < len(title_tokens):
                candidates.append(title_tokens[nxt])
    return candidates


def _load_unigram_freqs():
    unigram_counts = []
    with open(UNIGRAM_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            word = str(row.get("word", "")).strip().lower()
            if not word.isalpha():
                continue
            try:
                count = float(row.get("count", 0))
            except ValueError:
                count = 0.0
            unigram_counts.append((word, count))

    max_count = max((c for _, c in unigram_counts), default=1.0)
    unigram_freq = {w: (c / max_count) for w, c in unigram_counts}
    unigram_words = [w for w, _ in sorted(unigram_counts, key=lambda x: x[1], reverse=True)]
    return unigram_freq, unigram_words


def load_models():
    global LM, TFIDF_VECTORIZER, TFIDF_MATRIX, TAG_TO_WORDS
    global TOKENIZER, GPT2_MODEL, DEVICE, CLEANED_TITLES, UNIGRAM_FREQ, UNIGRAM_WORDS
    global TORCH, GPT2_AVAILABLE

    with open(NGRAM_PATH, "rb") as f:
        LM = pickle.load(f)
    with open(TFIDF_VECTORIZER_PATH, "rb") as f:
        TFIDF_VECTORIZER = pickle.load(f)
    with open(TFIDF_MATRIX_PATH, "rb") as f:
        TFIDF_MATRIX = pickle.load(f)
    with open(TAG_MODEL_PATH, "rb") as f:
        TAG_TO_WORDS = pickle.load(f)
    with open(CLEANED_TITLES_PATH, "rb") as f:
        CLEANED_TITLES = pickle.load(f)

    UNIGRAM_FREQ, UNIGRAM_WORDS = _load_unigram_freqs()

    try:
        import torch as _torch
        from transformers import GPT2LMHeadModel, GPT2Tokenizer

        TORCH = _torch
        TOKENIZER = GPT2Tokenizer.from_pretrained(GPT2_DIR, local_files_only=True)
        if TOKENIZER.pad_token is None:
            TOKENIZER.pad_token = TOKENIZER.eos_token

        GPT2_MODEL = GPT2LMHeadModel.from_pretrained(GPT2_DIR, local_files_only=True)
        DEVICE = "cpu"
        GPT2_MODEL.to(DEVICE)
        GPT2_MODEL.eval()
        GPT2_AVAILABLE = True
    except Exception:
        GPT2_AVAILABLE = False
        TOKENIZER = None
        GPT2_MODEL = None
        DEVICE = "cpu"


def _ensure_loaded():
    if LM is None:
        load_models()


def _normalize_scores(scores):
    if not scores:
        return {}
    max_v = max(scores.values())
    if max_v <= 0:
        return {k: 0.0 for k in scores}
    return {k: (v / max_v) for k, v in scores.items()}


def _trigram_ranked_candidates(query_tokens, tags=None, top_k=200):
    query_set = set(query_tokens)
    context = tuple(query_tokens[-2:])

    try:
        observed_items = list(LM.counts[context].items())
    except Exception:
        observed_items = []

    observed_items.sort(key=lambda x: x[1], reverse=True)
    observed_items = observed_items[: top_k * 4]

    total_count = float(sum(c for _, c in observed_items)) if observed_items else 0.0

    raw_scores = {}
    tag_raw = {}

    for word, count in observed_items:
        word = str(word)
        if (not word.isalpha()) or (word in query_set):
            continue

        trigram_prob = (float(count) / total_count) if total_count else 0.0
        unigram_f = UNIGRAM_FREQ.get(word, 0.0)

        score = (0.7 * trigram_prob) + (0.3 * unigram_f)

        if unigram_f < 0.0001:
            score *= 0.3

        raw_scores[word] = score

        if tags:
            tag_raw[word] = sum(TAG_TO_WORDS.get(tag, {}).get(word, 0.0) for tag in tags)

    if tags:
        tag_norm = _normalize_scores(tag_raw)
        rescored = {}
        for w, s in raw_scores.items():
            rescored[w] = (0.85 * s) + (0.15 * tag_norm.get(w, 0.0))
        raw_scores = rescored

    ranked = sorted(raw_scores.items(), key=lambda x: (-x[1], x[0]))
    return ranked[:top_k]


def _tfidf_fallback_candidates(query_tokens, top_k=30):
    query = " ".join(query_tokens)
    query_vec = TFIDF_VECTORIZER.transform([query])
    sims = cosine_similarity(query_vec, TFIDF_MATRIX).ravel()
    top_indices = np.argsort(sims)[-5:][::-1]

    counter = Counter()
    for idx in top_indices:
        title_tokens = CLEANED_TITLES[idx].split()
        for candidate in _extract_next_words(title_tokens, query_tokens):
            if candidate not in query_tokens:
                counter[candidate] += 1

    if not counter:
        return []

    max_c = max(counter.values())
    return [(w, c / max_c) for w, c in counter.most_common(top_k)]


def predict(query: str, tags=None, top_n: int = 5) -> list:
    _ensure_loaded()
    tags = tags or []

    cleaned_query = clean_text(query)
    query_tokens = cleaned_query.split()
    if not query_tokens:
        return []

    candidates = _trigram_ranked_candidates(query_tokens, tags=tags, top_k=max(80, top_n * 12))

    if len(candidates) < top_n:
        fallback = _tfidf_fallback_candidates(query_tokens, top_k=50)
        score_map = defaultdict(float)
        for w, s in candidates:
            score_map[w] = max(score_map[w], float(s))
        for w, s in fallback:
            if w not in query_tokens:
                score_map[w] = max(score_map[w], 0.55 * float(s))
        candidates = sorted(score_map.items(), key=lambda x: (-x[1], x[0]))

    suggestions = [w for w, _ in candidates if w not in query_tokens]
    suggestions = _dedupe_keep_order(suggestions)

    if len(suggestions) < top_n:
        for word in UNIGRAM_WORDS:
            if word not in query_tokens and word not in suggestions:
                suggestions.append(word)
            if len(suggestions) >= top_n:
                break

    return suggestions[:top_n]


def gpt2_predict(query: str, top_n: int = 5) -> list:
    _ensure_loaded()
    if not GPT2_AVAILABLE:
        return []

    cleaned_query = clean_text(query)
    if not cleaned_query:
        return []

    inputs = TOKENIZER(cleaned_query, return_tensors="pt", truncation=True, max_length=32)
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    with TORCH.no_grad():
        out = GPT2_MODEL(**inputs)

    last_logits = out.logits[0, -1, :]
    k = max(50, top_n * 10)
    top_ids = TORCH.topk(last_logits, k=k).indices.tolist()

    words = []
    seen = set()
    query_words = set(cleaned_query.split())
    for tok_id in top_ids:
        token_txt = TOKENIZER.decode([tok_id], skip_special_tokens=True)
        token_words = clean_text(token_txt).split()
        for w in token_words:
            if w and w not in query_words and w not in seen:
                seen.add(w)
                words.append(w)
        if len(words) >= top_n:
            break

    return words[:top_n]


def predict_hybrid(query, tags=None, top_n=5):
    ngram_suggestions = predict(query, tags=tags, top_n=max(10, top_n * 2))
    gpt2_suggestions = gpt2_predict(query, top_n=max(10, top_n * 2))

    score_map = defaultdict(float)

    for rank, word in enumerate(ngram_suggestions, start=1):
        score_map[word] += 0.6 * (1.0 / rank)
    for rank, word in enumerate(gpt2_suggestions, start=1):
        score_map[word] += 0.4 * (1.0 / rank)

    merged = [w for w, _ in sorted(score_map.items(), key=lambda x: (-x[1], x[0]))]
    merged = _dedupe_keep_order(merged)

    if len(merged) < top_n:
        base = predict(query, tags=tags, top_n=top_n)
        merged = _dedupe_keep_order(merged + base)

    return merged[:top_n]


if __name__ == "__main__":
    import sys

    _ensure_loaded()

    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
        print(f"Query: {query}")
        print(f"Suggestions: {predict_hybrid(query)}")
    else:
        demos = [
            ("machine learning", None),
            ("python", ["pandas"]),
            ("deep", None),
            ("neural network", ["tensorflow"]),
        ]
        for q, tags in demos:
            print(f"  {q!r:25} -> {predict_hybrid(q, tags)}")

    print("\n✓ READY TO DEPLOY")
