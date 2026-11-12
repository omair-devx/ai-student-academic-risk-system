"""Phase 6 — request/response schemas.

WHY the request model is generated from the model card instead of written by
hand: the feature list is *data* owned by ml/features + ml/registry/export.py.
A hand-written pydantic class would be a second copy that silently rots the
first time a horizon changes — and a validator that accepts fields the model
wasn't trained on (or misses fields it needs) is exactly the failure the API
contract forbids (docs/ARCHITECTURE.md §6).

Contract details enforced here:
- `extra="forbid"`  → unknown field (e.g. a typo like "Age at enrollmnt") is
  a 422, NOT a silently ignored key.
- required, STRICT `float` fields → missing field is 422 (no silent defaults:
  scoring a student against features they never sent would produce a
  plausible-looking but meaningless risk). Strict is essential: lax float
  coercion accepts `true` → 1.0 and `"19"` → 19.0, i.e. garbage in, silent
  garbage out — JSON numbers (int/float) are the only accepted inputs, which
  matches exactly what ml/inference/predict.py would accept.
- NaN/Infinity (possible via Python-originated JSON) are rejected by the
  inference layer's finite check and also surface as 422.

`PredictResponse` is static — the §6 response shape is fixed contract, so it
gets an explicit model (shows up in OpenAPI /docs, validates what we return).
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, create_model


class PredictResponse(BaseModel):
    """Exact response shape from docs/ARCHITECTURE.md §6."""
    horizon: str
    prediction: str
    at_risk: bool
    probabilities: dict[str, float]
    model_version: str


def build_student_model(card: dict):
    """Dynamic StudentPayload class for one registered model card.

    Built at import time in main.py (see WHY there): FastAPI needs the body
    type when the route is declared, and failing import when no model exists
    is the fail-fast behavior we want from a deployment.
    """
    horizon = card["horizon"]
    features = card["data"]["features"]
    field_defs = {
        name: (
            float,
            Field(..., strict=True,
                  description=f"required training feature ({horizon}); "
                              "JSON number only"),
        )
        for name in features
    }
    return create_model(
        f"StudentPayload_{horizon}",
        __config__=ConfigDict(extra="forbid"),
        **field_defs,
    )
