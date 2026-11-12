// Typed browser client for the Student Academic Risk API.
// Single source of truth for the field list is always the served model card
// (GET /model/info) — the form can never drift from backend validation.

const rawBase: unknown = import.meta.env.VITE_API_URL;
export const API_BASE = (
  typeof rawBase === "string" && rawBase.trim() !== ""
    ? rawBase.trim()
    : "http://127.0.0.1:8000"
).replace(/\/+$/, "");

export interface Health {
  status: string;
  model_version: string;
  horizon: string;
}

export interface Ablation {
  features_excluded: string[];
  reason: string;
  cv_macro_f1_with_A: number;
  test_macro_f1_with_A: number;
  cv_macro_f1_without_B: number;
  test_macro_f1_without_B: number;
  delta_cv_B_minus_A: number;
  delta_test_B_minus_A: number;
  verdict: string;
}

// Mirrors ml/registry/model_card.json exactly (served by GET /model/info).
export interface ModelCard {
  version: string;
  created_on: string;
  horizon: string;
  experiment: string;
  model: {
    config_id: string;
    label: string;
    params: string;
    scale_numeric: boolean;
  };
  selection: { rule: string; forced: boolean };
  data: {
    name: string;
    doi: string;
    url: string;
    license: string;
    citation: string;
    rows_total: number;
    train_rows: number;
    test_rows: number;
    split: string;
    target_column: string;
    target_classes: string[];
    features: string[];
    numerical: string[];
    categorical: string[];
  };
  protocol: { cv: string; selection: string; test: string; metrics: string[] };
  metrics: { cv: Record<string, number> | null; test: Record<string, number> };
  environment: Record<string, string>;
  notes: string[];
  uncertain_feature_ablation?: Ablation;
}

export interface PredictResponse {
  horizon: string;
  prediction: string;
  at_risk: boolean;
  probabilities: Record<string, number>;
  model_version: string;
}

interface FieldIssue {
  type: string;
  loc: (string | number)[];
  msg: string;
}

export class ApiRequestError extends Error {
  readonly status: number;
  /** Per-field messages keyed by feature name (only for FastAPI 422 arrays). */
  readonly fieldErrors: Record<string, string>;

  constructor(
    status: number,
    message: string,
    fieldErrors: Record<string, string> = {},
  ) {
    super(message);
    this.name = "ApiRequestError";
    this.status = status;
    this.fieldErrors = fieldErrors;
  }
}

function parseError(status: number, body: unknown): ApiRequestError {
  const detail: unknown = (body as { detail?: unknown } | null)?.detail;

  // FastAPI request-validation errors: detail = list of issue objects with
  // loc like ["body", "<field>", ...]. Our own HTTPException 422/503 uses a
  // plain string — both shapes end up as useful messages.
  if (Array.isArray(detail)) {
    const fieldErrors: Record<string, string> = {};
    let general: string | null = null;
    for (const raw of detail as FieldIssue[]) {
      const field = typeof raw.loc?.[1] === "string" ? raw.loc[1] : null;
      if (field !== null) {
        if (!(field in fieldErrors)) fieldErrors[field] = raw.msg;
      } else {
        general = raw.msg;
      }
    }
    const message =
      general ??
      (Object.keys(fieldErrors).length > 0
        ? `Invalid input: ${Object.keys(fieldErrors).length} field(s) need attention.`
        : "The API rejected the request.");
    return new ApiRequestError(status, message, fieldErrors);
  }

  if (typeof detail === "string" && detail.trim() !== "") {
    return new ApiRequestError(status, detail);
  }
  return new ApiRequestError(status, `Request failed with status ${status}.`);
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let res: Response;
  try {
    res = await fetch(`${API_BASE}${path}`, {
      ...init,
      headers: { "Content-Type": "application/json", ...init?.headers },
    });
  } catch {
    throw new ApiRequestError(
      0,
      `Cannot reach the API at ${API_BASE} — is the backend running?`,
    );
  }
  const body: unknown = await res.json().catch(() => null);
  if (!res.ok) throw parseError(res.status, body);
  return body as T;
}

export function fetchHealth(): Promise<Health> {
  return request<Health>("/health");
}

export function fetchModelInfo(): Promise<ModelCard> {
  return request<ModelCard>("/model/info");
}

export function predict(payload: Record<string, number>): Promise<PredictResponse> {
  return request<PredictResponse>("/predict", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}
