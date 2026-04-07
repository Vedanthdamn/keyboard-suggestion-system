import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app import clean_text, predict, predict_hybrid

OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)


def get_text_series(df: pd.DataFrame) -> pd.Series:
    lower_map = {str(col).lower(): col for col in df.columns}
    if "title" in lower_map:
        col = lower_map["title"]
        return df[col].dropna().astype(str)

    object_cols = [col for col in df.columns if df[col].dtype == object]
    if object_cols:
        col = object_cols[0]
        return df[col].dropna().astype(str)

    return df[df.columns[0]].dropna().astype(str)


def build_metrics() -> dict:
    df = pd.read_csv(ROOT / "data" / "raw" / "train.csv")
    titles = get_text_series(df).tolist()
    cleaned_titles = [clean_text(t) for t in titles]
    cleaned_titles = [t for t in cleaned_titles if len(t.split()) >= 3]

    pairs = []
    for title in cleaned_titles:
        tokens = title.split()
        for i in range(1, min(len(tokens), 6)):
            query = " ".join(tokens[:i])
            true_next = tokens[i]
            pairs.append((query, true_next))

    rng = np.random.default_rng(42)
    if len(pairs) > 800:
        idx = rng.choice(len(pairs), size=800, replace=False)
        pairs = [pairs[i] for i in idx]

    baseline_top1 = baseline_top5 = final_top1 = final_top5 = 0
    for query, true_next in pairs:
        baseline = predict(query, top_n=5)
        final = predict_hybrid(query, top_n=5)

        if baseline:
            if baseline[0] == true_next:
                baseline_top1 += 1
            if true_next in baseline[:5]:
                baseline_top5 += 1

        if final:
            if final[0] == true_next:
                final_top1 += 1
            if true_next in final[:5]:
                final_top5 += 1

    n = max(len(pairs), 1)
    return {
        "baseline_top1": round(100 * baseline_top1 / n, 1),
        "baseline_top5": round(100 * baseline_top5 / n, 1),
        "final_top1": round(100 * final_top1 / n, 1),
        "final_top5": round(100 * final_top5 / n, 1),
        "sample_size": n,
    }


def save_accuracy_plot(metrics: dict) -> None:
    labels = ["Top-1", "Top-5"]
    baseline_vals = [metrics["baseline_top1"], metrics["baseline_top5"]]
    final_vals = [metrics["final_top1"], metrics["final_top5"]]

    x = np.arange(len(labels))
    width = 0.34

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width / 2, baseline_vals, width, label="Baseline (predict)")
    ax.bar(x + width / 2, final_vals, width, label="Final (predict_hybrid)")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Accuracy Improvement: Baseline vs Final")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, max(final_vals + baseline_vals + [10]) + 10)
    for i, value in enumerate(baseline_vals):
        ax.text(i - width / 2, value + 1, f"{value}%", ha="center", fontsize=9)
    for i, value in enumerate(final_vals):
        ax.text(i + width / 2, value + 1, f"{value}%", ha="center", fontsize=9)
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT / "accuracy_improvement.png", dpi=160)
    plt.close(fig)


def save_zipf_plot() -> None:
    unigram = pd.read_csv(ROOT / "data" / "raw" / "unigram_freq.csv")
    counts = unigram["count"].astype(float).sort_values(ascending=False).to_numpy()
    ranks = np.arange(1, len(counts) + 1)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(ranks[:50000], counts[:50000], linewidth=1.2)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title("Word Frequency Distribution (Zipf's Law)")
    ax.set_xlabel("Rank (log scale)")
    ax.set_ylabel("Frequency (log scale)")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT / "zipf_distribution.png", dpi=160)
    plt.close(fig)


def save_query_length_plot() -> None:
    df = pd.read_csv(ROOT / "data" / "raw" / "train.csv")
    titles = get_text_series(df).tolist()
    lengths = [len(clean_text(t).split()) for t in titles]
    lengths = [value for value in lengths if value > 0]

    if not lengths:
        lengths = [1]

    fig, ax = plt.subplots(figsize=(8, 5))
    bins = np.arange(1, min(max(lengths) + 2, 26))
    ax.hist(lengths, bins=bins, color="#4f7bd9", edgecolor="white", alpha=0.9)
    ax.set_title("Query Length Distribution")
    ax.set_xlabel("Number of Tokens")
    ax.set_ylabel("Count")
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(OUT / "query_length_distribution.png", dpi=160)
    plt.close(fig)


def save_test_cases() -> int:
    queries = [
        "machine learning",
        "python",
        "deep",
        "data science",
        "how to",
        "javascript",
        "neural network",
        "sql",
        "git",
        "pandas dataframe",
        "react",
        "sorting algorithm",
        "docker",
        "api design",
        "natural language",
    ]

    records = []
    for query in queries:
        suggestions = predict_hybrid(query, top_n=5)
        if suggestions:
            records.append({"query": query, "suggestions": suggestions})

    with open(OUT / "test_cases.json", "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)
    return len(records)


def main() -> None:
    metrics = build_metrics()
    save_accuracy_plot(metrics)
    save_zipf_plot()
    save_query_length_plot()
    count = save_test_cases()

    with open(OUT / "metrics_summary.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("metrics:", metrics)
    print("test_case_count:", count)


if __name__ == "__main__":
    main()
