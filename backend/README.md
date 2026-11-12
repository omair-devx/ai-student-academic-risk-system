# backend/ — FastAPI application (Phase 6)

Serves the active Phase-5 registry model over HTTP. Three endpoints, exactly
the contract in `docs/ARCHITECTURE.md` §6:

| Method | Path | Returns |
|---|---|---|
| GET | `/health` | `{"status": "ok", "model_version": ..., "horizon": ...}` |
| GET | `/model/info` | the full `model_card.json` (metrics, features, protocol) — served **without unpickling** `model.joblib` |
| POST | `/predict` | §6 response: `horizon`, `prediction`, `at_risk`, `probabilities`, `model_version` |

Interactive OpenAPI docs: `http://127.0.0.1:8000/docs`.

## Layout

```
backend/
├── app/
│   ├── main.py       # app, CORS, routes, CLI runner
│   └── schemas.py    # StudentPayload built FROM the model card (dynamic)
└── README.md
```

- **`app/routes/` was deliberately skipped** — three tiny endpoints live in
  `main.py`; a router package makes sense when the API grows (Phase 7+).
- **Contract tests** belong to Phase 8 (`tests/`), per the placeholder plan.

## How validation works

`StudentPayload` is generated at import time from
`ml/registry/<current>/model_card.json` — one source of truth, no hand-written
feature list that can drift. `extra="forbid"` + required **strict** `float`:

| input | status | why |
|---|---|---|
| valid JSON numbers for all 24 fields | 200 | contract met |
| missing field | 422 | no silent defaults (unscoreable student) |
| unknown/typo field | 422 | `extra=forbid` — silent drops hide client bugs |
| `"high"` or `"19"` (string) | 422 | strict float — no coercion of garbage |
| `true` (boolean) | 422 | strict float (lax would coerce `true → 1.0`!) |
| NaN/Infinity | 422 | rejected by the inference layer's finite check |

Predictions always go through `ml/inference.predict.predict_student` — the API
**never** preprocesses anything itself.

## Running

```powershell
python -m backend.app.main                 # 127.0.0.1:8000
python -m backend.app.main --port 9000 --reload   # dev auto-reload
uvicorn backend.app.main:app               # equivalent ASGI form
```

Bind host/port come from `API_HOST`/`API_PORT` (see `.env.example`); CORS
origins from `CORS_ORIGINS` (default: Vite dev servers `localhost:5173`).

**Fail-fast startup:** the app resolves `current` + card at import. No
registered model → uvicorn refuses to start with the registry's actionable
error (rather than a "healthy" service that can't predict). After
re-exporting models, restart the server to pick up a new `current` pointer.

Example:

```powershell
$body = Get-Content demo.json -Raw   # 24-feature student object
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/predict `
    -ContentType "application/json" -Body $body
```
