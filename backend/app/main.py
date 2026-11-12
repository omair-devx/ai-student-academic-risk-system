"""Phase 6 — FastAPI backend serving the active Phase-5 registry model.

Endpoints (the contract in docs/ARCHITECTURE.md §6; contract tests in Phase 8):

    GET  /health      liveness + loaded model version
    GET  /model/info  full model card (metrics, features, protocol) —
                      model_card.json served WITHOUT unpickling model.joblib
    POST /predict     validated student dict -> risk class + probabilities

WHY the app never touches preprocessing: predictions go through
ml.inference.predict.predict_student, which loads and applies the exact
fitted pipeline from the registry. The API must not reimplement (and can
never re-fit) any part of training — one pipeline object, trained once,
served everywhere.

WHY schema/card/version are resolved at IMPORT time (not lazily per request):
- FastAPI needs the body type when the route function is declared, so the
  dynamic StudentPayload model must exist before @app.post runs.
- A deployment with no registered model fails immediately at startup with
  the registry's actionable error, instead of coming up "healthy" while
  being unable to predict. Reloading the `current` pointer therefore
  requires a server restart (documented in backend/README.md).

Run:
    python -m backend.app.main                 # 127.0.0.1:8000 (API_HOST/API_PORT)
    python -m backend.app.main --port 9000 --reload
    uvicorn backend.app.main:app --host 127.0.0.1 --port 8000
"""

from __future__ import annotations

import argparse
import os

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware

from backend.app.schemas import PredictResponse, build_student_model
from ml.inference.predict import predict_student
from ml.registry.registry import get_current, load_card

# --- resolved once, at import: fail fast, keep a stable OpenAPI schema -----
MODEL_VERSION: str = get_current()
CARD: dict = load_card(MODEL_VERSION)
StudentPayload = build_student_model(CARD)

# Dev-server origins for the React frontend (Phase 7); override via env.
# 5173 = Vite dev server, 4173 = `vite preview` of the production build.
DEFAULT_ORIGINS = "http://localhost:5173,http://127.0.0.1:5173,http://localhost:4173"

app = FastAPI(
    title="Student Academic Risk API",
    description=(
        "Predicts student academic risk (Dropout/Enrolled/Graduate) from "
        "enrollment-time features. Educational support tool only — not for "
        "automated decision-making."
    ),
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        o.strip()
        for o in os.environ.get("CORS_ORIGINS", DEFAULT_ORIGINS).split(",")
        if o.strip()
    ],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.get("/health", summary="Liveness + loaded model version")
def health() -> dict:
    return {
        "status": "ok",
        "model_version": MODEL_VERSION,
        "horizon": CARD["horizon"],
    }


@app.get("/model/info", summary="Model card: metrics, horizons, feature list")
def model_info() -> dict:
    # Straight from model_card.json (registry.load_card) — no joblib involved.
    return CARD


@app.post(
    "/predict",
    response_model=PredictResponse,
    summary="Risk prediction for one student (422 on any invalid field)",
)
def predict(payload: StudentPayload) -> PredictResponse:
    # model_dump() yields plain floats; the inference layer re-validates
    # (defense in depth) and applies THE trained pipeline.
    try:
        result = predict_student(payload.model_dump(), version=MODEL_VERSION)
    except ValueError as exc:
        # Raised by the inference layer for contract violations pydantic
        # couldn't catch (e.g. booleans, NaN) — same 422 semantics.
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        # Model files vanished mid-flight (bad deployment) — not a client
        # error, so 503 rather than 422/500.
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return PredictResponse(**result)


def cli() -> None:
    parser = argparse.ArgumentParser(description="Run the Student Academic Risk API.")
    parser.add_argument("--host", default=os.environ.get("API_HOST", "127.0.0.1"),
                        help="bind address (env API_HOST)")
    parser.add_argument("--port", type=int,
                        default=int(os.environ.get("API_PORT", "8000")),
                        help="bind port (env API_PORT)")
    parser.add_argument("--reload", action="store_true",
                        help="dev-only auto-reload on code changes")
    args = parser.parse_args()

    import uvicorn  # local import: keeps `import backend.app.main` cheap

    # String target (not the app object) so --reload can re-import it.
    uvicorn.run("backend.app.main:app", host=args.host, port=args.port,
                reload=args.reload)


if __name__ == "__main__":
    cli()
