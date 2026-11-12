import type { PredictResponse } from "../api";

interface Props {
  result: { data: PredictResponse; ms: number } | null;
  targetClasses: string[];
}

// Risk tones shared by badges, bars and the chart — one colour per class.
const CLASS_TONE: Record<string, string> = {
  Dropout: "red",
  Enrolled: "amber",
  Graduate: "green",
};

const INTERPRETATION: Record<string, string> = {
  Dropout:
    "High risk of dropout — consider early outreach (advising, financial aid check).",
  Enrolled: "Still enrolled at this horizon — monitor progress.",
  Graduate: "On track to graduate.",
};

function toneFor(name: string): string {
  return CLASS_TONE[name] ?? "indigo";
}

export function ResultCard({ result, targetClasses }: Props) {
  if (!result) {
    return (
      <div className="panel result-card">
        <div className="empty-state">
          <svg viewBox="0 0 48 48" aria-hidden="true" className="empty-icon">
            <path
              d="M24 5 41 12v12c0 9.6-6.7 17.4-17 20-10.3-2.6-17-10.4-17-20V12L24 5Z"
              fill="none"
              stroke="currentColor"
              strokeWidth="2.5"
              strokeLinejoin="round"
            />
            <path
              d="M16 24.5l5.5 5.5L32 19.5"
              fill="none"
              stroke="currentColor"
              strokeWidth="2.5"
              strokeLinecap="round"
              strokeLinejoin="round"
            />
          </svg>
          <h3>No prediction yet</h3>
          <p className="muted">
            Fill the enrollment-time fields and run a prediction to see the risk
            class and per-class probabilities here.
          </p>
        </div>
      </div>
    );
  }

  const { data, ms } = result;
  const tone = toneFor(data.prediction);

  return (
    <div className="panel result-card" aria-live="polite">
      <div className={`result-hero tone-${tone}`}>
        <span className="result-kicker">Prediction</span>
        <span className={`result-badge tone-${tone}`}>{data.prediction}</span>
        <p className="result-interpretation">
          {INTERPRETATION[data.prediction] ??
            "See per-class probabilities below."}
        </p>
        <span
          className={`risk-chip${data.at_risk ? " risk-on" : ""}`}
          title="at_risk = (prediction == 'Dropout'), set by the API"
        >
          {data.at_risk ? "At-risk flag: YES" : "At-risk flag: NO"}
        </span>
      </div>

      <div className="prob-block">
        <h4>Class probabilities</h4>
        {targetClasses.map((cls) => {
          const p = data.probabilities[cls] ?? 0;
          const pctValue = p * 100;
          return (
            <div key={cls} className="prob-row">
              <span className="prob-label">{cls}</span>
              <div className="prob-track">
                <div
                  className={`prob-fill tone-${toneFor(cls)}`}
                  style={{ width: `${pctValue.toFixed(1)}%` }}
                />
              </div>
              <span className="prob-value">{pctValue.toFixed(1)}%</span>
            </div>
          );
        })}
      </div>

      <dl className="result-meta">
        <div>
          <dt>Response time</dt>
          <dd>{ms} ms</dd>
        </div>
        <div>
          <dt>Model</dt>
          <dd className="mono">{data.model_version}</dd>
        </div>
        <div>
          <dt>Horizon</dt>
          <dd>{data.horizon}</dd>
        </div>
      </dl>
    </div>
  );
}
