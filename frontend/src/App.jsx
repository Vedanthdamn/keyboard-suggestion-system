import { useEffect, useMemo, useState } from "react";
import SuggestionBox from "./SuggestionBox";

const API_URL =
  import.meta.env.VITE_PREDICT_API_URL || "http://localhost:8000/predict";
const EXAMPLES = ["machine learning", "python", "neural network", "data science"];

function useDebouncedValue(value, delayMs) {
  const [debounced, setDebounced] = useState(value);

  useEffect(() => {
    const timer = window.setTimeout(() => setDebounced(value), delayMs);
    return () => window.clearTimeout(timer);
  }, [value, delayMs]);

  return debounced;
}

export default function App() {
  const [query, setQuery] = useState("");
  const [suggestions, setSuggestions] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [latencyMs, setLatencyMs] = useState(null);

  const debouncedQuery = useDebouncedValue(query, 300);

  useEffect(() => {
    const normalized = debouncedQuery.trim();
    if (!normalized) {
      setSuggestions([]);
      setError("");
      setLatencyMs(null);
      return;
    }

    let isCancelled = false;
    setLoading(true);
    setError("");

    const startedAt = performance.now();

    fetch(API_URL, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query: normalized }),
    })
      .then(async (res) => {
        if (!res.ok) {
          throw new Error("Failed to fetch suggestions");
        }
        const data = await res.json();
        if (isCancelled) {
          return;
        }
        setSuggestions(Array.isArray(data.suggestions) ? data.suggestions.slice(0, 5) : []);
        setLatencyMs(Math.round(performance.now() - startedAt));
      })
      .catch(() => {
        if (isCancelled) {
          return;
        }
        setSuggestions([]);
        setError("Backend unavailable. Start FastAPI server on port 8000.");
        setLatencyMs(null);
      })
      .finally(() => {
        if (!isCancelled) {
          setLoading(false);
        }
      });

    return () => {
      isCancelled = true;
    };
  }, [debouncedQuery]);

  const helperText = useMemo(() => {
    if (!query.trim()) {
      return "Type a technical keyword to get instant next-word suggestions.";
    }
    if (latencyMs == null) {
      return "Searching...";
    }
    return `Powered by N-gram + TF-IDF model • ${latencyMs} ms`;
  }, [query, latencyMs]);

  return (
    <main className="page">
      <section className="search-card">
        <h1>Keyword Suggestion</h1>
        <p className="subtitle">Autocomplete for technical search queries</p>

        <div className="search-shell">
          <input
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Try: machine learning"
            className="search-input"
            autoComplete="off"
            aria-label="Keyword query"
          />
          <SuggestionBox
            suggestions={suggestions}
            query={query}
            loading={loading}
            error={error}
          />
        </div>

        <p className="helper-text">{helperText}</p>

        <div className="examples">
          <span>Example queries:</span>
          {EXAMPLES.map((example) => (
            <button
              key={example}
              type="button"
              className="example-chip"
              onClick={() => setQuery(example)}
            >
              {example}
            </button>
          ))}
        </div>
      </section>
    </main>
  );
}
