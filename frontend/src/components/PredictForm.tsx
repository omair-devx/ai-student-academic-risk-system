import { type FormEvent, type ReactNode, useMemo, useState } from "react";
import {
  ApiRequestError,
  predict,
  type ModelCard,
  type PredictResponse,
} from "../api";
import {
  EXAMPLE_STUDENT,
  FIELD_META,
  GROUPS,
  type FieldMeta,
  type GroupId,
} from "../fieldMeta";
import { ResultCard } from "./ResultCard";

interface Props {
  card: ModelCard;
}

interface ResultState {
  data: PredictResponse;
  ms: number;
}

function slug(name: string): string {
  return name.replace(/[^a-zA-Z0-9]+/g, "-").replace(/^-|-$/g, "").toLowerCase();
}

function FieldControl({
  name,
  meta,
  value,
  error,
  onChange,
}: {
  name: string;
  meta: FieldMeta;
  value: string;
  error?: string;
  onChange: (next: string) => void;
}) {
  const id = `f-${slug(name)}`;
  const hintId = `${id}-hint`;
  const errId = `${id}-err`;
  const describedBy =
    [meta.hint ? hintId : null, error ? errId : null]
      .filter(Boolean)
      .join(" ") || undefined;

  let control: ReactNode;
  if (meta.kind === "segmented" && meta.options) {
    control = (
      <div
        className="segmented"
        role="radiogroup"
        aria-labelledby={`${id}-label`}
        aria-describedby={describedBy}
        aria-invalid={Boolean(error)}
      >
        {meta.options.map((opt) => (
          <label key={opt.value} className="seg-opt">
            <input
              type="radio"
              name={id}
              value={String(opt.value)}
              checked={value !== "" && Number(value) === opt.value}
              onChange={() => onChange(String(opt.value))}
            />
            <span>{opt.label}</span>
          </label>
        ))}
      </div>
    );
  } else if (meta.kind === "select" && meta.options) {
    control = (
      <select
        id={id}
        className="control"
        value={value}
        onChange={(e) => onChange(e.target.value)}
        aria-describedby={describedBy}
        aria-invalid={Boolean(error)}
      >
        <option value="">Select…</option>
        {meta.options.map((opt) => (
          <option key={opt.value} value={String(opt.value)}>
            {opt.label}
          </option>
        ))}
      </select>
    );
  } else {
    control = (
      <input
        id={id}
        type="number"
        className="control"
        inputMode="decimal"
        step={meta.step ?? "any"}
        value={value}
        placeholder="—"
        onChange={(e) => onChange(e.target.value)}
        aria-describedby={describedBy}
        aria-invalid={Boolean(error)}
      />
    );
  }

  return (
    <div className={`field${error ? " field-invalid" : ""}`}>
      {meta.kind === "segmented" ? (
        <span className="field-label" id={`${id}-label`}>
          {meta.label}
        </span>
      ) : (
        <label className="field-label" htmlFor={id}>
          {meta.label}
        </label>
      )}
      {control}
      {meta.hint && (
        <span className="field-hint" id={hintId}>
          {meta.hint}
        </span>
      )}
      {error && (
        <span className="field-error" id={errId} role="alert">
          {error}
        </span>
      )}
    </div>
  );
}

export function PredictForm({ card }: Props) {
  const features = card.data.features;

  const [values, setValues] = useState<Record<string, string>>(() =>
    Object.fromEntries(features.map((f) => [f, ""])),
  );
  const [result, setResult] = useState<ResultState | null>(null);
  const [fieldErrors, setFieldErrors] = useState<Record<string, string>>({});
  const [requestError, setRequestError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  const { grouped, extra } = useMemo(() => {
    const grouped = new Map<GroupId, string[]>();
    const extra: string[] = [];
    for (const f of features) {
      const meta = FIELD_META[f];
      if (!meta) {
        extra.push(f);
        continue;
      }
      const list = grouped.get(meta.group) ?? [];
      list.push(f);
      grouped.set(meta.group, list);
    }
    return { grouped, extra };
  }, [features]);

  function setValue(name: string, next: string) {
    setValues((prev) => ({ ...prev, [name]: next }));
    setFieldErrors((prev) => {
      if (!(name in prev)) return prev;
      const { [name]: _removed, ...rest } = prev;
      return rest;
    });
  }

  function clearAll() {
    setValues(Object.fromEntries(features.map((f) => [f, ""])));
    setFieldErrors({});
    setRequestError(null);
    setResult(null);
  }

  function loadExample() {
    setValues(
      Object.fromEntries(features.map((f) => [f, String(EXAMPLE_STUDENT[f] ?? "")])),
    );
    setFieldErrors({});
    setRequestError(null);
    setResult(null);
  }

  async function onSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setRequestError(null);
    setFieldErrors({});

    // Completeness/type check mirrors backend semantics (strict floats, no
    // coercion) but runs locally first so the common case is a friendly
    // inline error instead of a round trip.
    const payload: Record<string, number> = {};
    const errs: Record<string, string> = {};
    for (const f of features) {
      const raw = (values[f] ?? "").trim();
      const n = Number(raw);
      if (raw === "" || !Number.isFinite(n)) {
        errs[f] = "Required — enter a valid number.";
      } else {
        payload[f] = n;
      }
    }
    if (Object.keys(errs).length > 0) {
      setFieldErrors(errs);
      return;
    }

    setBusy(true);
    const t0 = performance.now();
    try {
      const data = await predict(payload);
      setResult({ data, ms: Math.round(performance.now() - t0) });
    } catch (err) {
      setResult(null);
      if (err instanceof ApiRequestError) {
        if (Object.keys(err.fieldErrors).length > 0) {
          setFieldErrors(err.fieldErrors);
        } else {
          setRequestError(err.message);
        }
      } else {
        setRequestError("Unexpected error while contacting the API.");
      }
    } finally {
      setBusy(false);
    }
  }

  const ablation = card.uncertain_feature_ablation;

  return (
    <div className="predict-layout">
      <form className="predict-form" onSubmit={onSubmit} noValidate>
        <div className="page-heading">
          <div>
            <h2>Risk prediction</h2>
            <p className="muted">
              24 enrollment-time features → {card.model.label} → Dropout /
              Enrolled / Graduate
            </p>
          </div>
          <div className="heading-actions">
            <button type="button" className="btn btn-ghost" onClick={loadExample}>
              Load example
            </button>
            <button type="button" className="btn btn-ghost" onClick={clearAll}>
              Clear
            </button>
          </div>
        </div>

        {requestError && (
          <div className="banner banner-error" role="alert">
            {requestError}
          </div>
        )}

        {GROUPS.map((group) => {
          const members = grouped.get(group.id) ?? [];
          if (members.length === 0) return null;
          return (
            <section key={group.id} className="panel group-panel">
              <header className="group-header">
                <h3>{group.title}</h3>
                <p className="muted">{group.blurb}</p>
                {group.id === "circumstances" && ablation && (
                  <p className="note note-amber">
                    Provenance of Displaced / Debtor / Tuition fees up to date /
                    Scholarship holder is not documented by UCI — the ablation on
                    the Model &amp; metrics tab shows these four features matter.
                  </p>
                )}
              </header>
              <div className="field-grid">
                {members.map((f) => {
                  const meta = FIELD_META[f];
                  return (
                    <FieldControl
                      key={f}
                      name={f}
                      meta={meta}
                      value={values[f] ?? ""}
                      error={fieldErrors[f]}
                      onChange={(next) => setValue(f, next)}
                    />
                  );
                })}
              </div>
            </section>
          );
        })}

        {extra.length > 0 && (
          <section className="panel group-panel">
            <header className="group-header">
              <h3>Other features</h3>
              <p className="muted">Fields served by the model without UI metadata.</p>
            </header>
            <div className="field-grid">
              {extra.map((f) => (
                <FieldControl
                  key={f}
                  name={f}
                  meta={{ label: f, group: "macro", kind: "number" }}
                  value={values[f] ?? ""}
                  error={fieldErrors[f]}
                  onChange={(next) => setValue(f, next)}
                />
              ))}
            </div>
          </section>
        )}

        <div className="form-actions">
          <button type="submit" className="btn btn-primary" disabled={busy}>
            {busy ? "Running…" : "Run prediction"}
          </button>
          <p className="muted small">
            Synthetic input only — no real student data leaves this browser
            (it is sent to the local API).
          </p>
        </div>
      </form>

      <aside className="result-col" aria-label="Prediction result">
        <ResultCard
          result={result}
          targetClasses={card.data.target_classes}
        />
      </aside>
    </div>
  );
}
