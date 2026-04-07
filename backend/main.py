"""FastAPI inference service for keyword suggestions."""

from functools import lru_cache
from time import perf_counter

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app import _ensure_loaded, clean_text, predict_hybrid


class PredictRequest(BaseModel):
    query: str


class PredictResponse(BaseModel):
    suggestions: list[str]


app = FastAPI(title="Keyword Suggestion API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@lru_cache(maxsize=4096)
def _cached_predict(normalized_query: str) -> tuple[str, ...]:
    return tuple(predict_hybrid(normalized_query, top_n=5))


@app.on_event("startup")
def _startup() -> None:
    _ensure_loaded()


@app.post("/predict", response_model=PredictResponse)
def predict_endpoint(payload: PredictRequest) -> PredictResponse:
    normalized_query = clean_text(payload.query)
    if not normalized_query:
        return PredictResponse(suggestions=[])

    start = perf_counter()
    suggestions = list(_cached_predict(normalized_query))
    elapsed_ms = (perf_counter() - start) * 1000.0

    # Preserve a predictable top-5 output shape for frontend rendering.
    suggestions = suggestions[:5]

    # Latency is intentionally measured to keep inference path lightweight.
    _ = elapsed_ms

    return PredictResponse(suggestions=suggestions)
