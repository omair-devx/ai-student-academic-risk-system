import { useState } from "react";
import type { ModelCard } from "../api";
import { ComparisonChart } from "./ComparisonChart";

interface Props {
  card: ModelCard;
}

function pct(x: number | undefined, digits = 1): string {
  return x === undefined ? "—" : `${(x * 100).toFixed(digits)}%`;
}

function f3(x: number | undefined): string {
  return x === undefined ? "—" : x.toFixed(3);
}

function MetricTile({
  label,
  value,
  sub,
  primary,
}: {
  label: string;
  value: string;
  sub: string;
  primary?: boolean;
}) {
  return (
    <div className={`metric-tile${primary ? " metric-primary" : ""}`}>
      <span className="metric-label">{label}</span>
      <span className="metric-value">{value}</span>
      <span className="metric-sub">{sub}</span>
    </div>
  );
}

export function ModelInfo({ card }: Props) {
  const [copied, setCopied] = useState(false);
  const cv = card.metrics.cv;
  const test = card.metrics.test;
  const ablation = card.uncertain_feature_ablation;

  async function copyCitation() {
    try {
      await navigator.clipboard.writeText(card.data.citation);
      setCopied(true);
      window.setTimeout(() => setCopied(false), 2000);
    } catch {
      // Clipboard can be blocked (permissions/insecure context) — the
      // citation stays selectable as plain text, so do nothing.
    }
  }

  const overallRows: { label: string; cv?: number; test?: number }[] = [
    { label: "Macro-F1 (primary)", cv: cv?.macro_f1, test: test?.macro_f1 },
    {
      label: "Balanced accuracy",
      cv: cv?.balanced_accuracy,
      test: test?.balanced_accuracy,
    },
    { label: "Accuracy (reference only)", cv: cv?.accuracy, test: test?.accuracy },
    { label: "ROC-AUC (Dropout vs rest)", cv: cv?.auc_dropout, test: test?.auc_dropout },
  ];

  return (
    <div className="model-info">
      <div className="page-heading">
        <div>
          <h2>Model &amp; metrics</h2>
          <p className="muted">
            Everything below comes from the served model card
            (<span className="mono">{card.version}</span>, created{" "}
            {card.created_on}) — the same JSON the API returns.
          </p>
        </div>
        <div className="heading-chips">
          <span className="chip chip-indigo">{card.horizon}</span>
          <span className="chip">Experiment {card.experiment}</span>
          <span className="chip chip-green">selected by CV rule</span>
        </div>
      </div>

      <section className="metric-grid" aria-label="Headline metrics">
        <MetricTile
          label="CV macro-F1 (train, 5-fold)"
          value={pct(cv?.macro_f1)}
          sub={`± ${pct(cv?.macro_f1_std, 1)} across folds`}
        />
        <MetricTile
          label="Test macro-F1 (terminal)"
          value={pct(test?.macro_f1)}
          sub="untouched 20% holdout"
          primary
        />
        <MetricTile
          label="Test balanced accuracy"
          value={pct(test?.balanced_accuracy)}
          sub="class-imbalance aware"
        />
        <MetricTile
          label="Test ROC-AUC (Dropout)"
          value={f3(test?.auc_dropout)}
          sub="ranking quality"
        />
      </section>

      <div className="info-two-col">
        <section className="panel">
          <header className="panel-header">
            <h2>Overall metrics — CV vs terminal test</h2>
          </header>
          <table className="data-table">
            <thead>
              <tr>
                <th scope="col">Metric</th>
                <th scope="col">CV (train)</th>
                <th scope="col">Test</th>
              </tr>
            </thead>
            <tbody>
              {overallRows.map((row) => (
                <tr key={row.label}>
                  <th scope="row">{row.label}</th>
                  <td>{pct(row.cv)}</td>
                  <td className="cell-strong">{pct(row.test)}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <p className="muted small table-note">
            Accuracy is shown for reference only: the majority-class baseline is
            already 49.9% accuracy at 0.222 macro-F1 — macro-F1 is the headline
            metric for this imbalanced 3-class problem.
          </p>
        </section>

        <section className="panel">
          <header className="panel-header">
            <h2>Per-class metrics — test holdout</h2>
          </header>
          <table className="data-table">
            <thead>
              <tr>
                <th scope="col">Class</th>
                <th scope="col">Precision</th>
                <th scope="col">Recall</th>
                <th scope="col">F1</th>
              </tr>
            </thead>
            <tbody>
              {card.data.target_classes.map((cls) => (
                <tr key={cls}>
                  <th scope="row">{cls}</th>
                  <td>{pct(test[`precision_${cls}`])}</td>
                  <td>{pct(test[`recall_${cls}`])}</td>
                  <td className="cell-strong">{pct(test[`f1_${cls}`])}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <p className="muted small table-note">
            Macro-F1 above = unweighted mean of the three per-class F1 scores.
          </p>
        </section>
      </div>

      <section className="panel">
        <header className="panel-header">
          <h2>Selected configuration</h2>
        </header>
        <div className="config-block">
          <div>
            <h3 className="config-label">{card.model.label}</h3>
            <p className="mono muted small">{card.model.config_id}</p>
          </div>
          <code className="config-params">{card.model.params}</code>
        </div>
        <dl className="kv-list">
          <div>
            <dt>Selection rule</dt>
            <dd>{card.selection.rule}</dd>
          </div>
          <div>
            <dt>Forced pick?</dt>
            <dd>
              {card.selection.forced
                ? "Yes — forced override (out of protocol)"
                : "No — won by the pre-declared CV rule; the test set was never consulted"}
            </dd>
          </div>
        </dl>
      </section>

      <ComparisonChart
        activeHorizon={card.horizon}
        activeConfigId={card.model.config_id}
      />

      <section className="panel">
        <header className="panel-header">
          <h2>Input features ({card.data.features.length})</h2>
          <p className="muted">
            {card.data.numerical.length} numerical +{" "}
            {card.data.categorical.length} categorical — treatment per
            docs/PREPROCESSING.md; the API validates against exactly this list.
          </p>
        </header>
        <div className="chip-cloud">
          {card.data.numerical.map((f) => (
            <span key={f} className="chip chip-blue">
              {f}
            </span>
          ))}
          {card.data.categorical.map((f) => (
            <span key={f} className="chip">
              {f}
            </span>
          ))}
        </div>
      </section>

      {ablation && (
        <section className="panel panel-amber">
          <header className="panel-header">
            <h2>Uncertain-feature ablation (Experiment B)</h2>
            <p className="muted">{ablation.reason}</p>
          </header>
          <div className="chip-cloud">
            {ablation.features_excluded.map((f) => (
              <span key={f} className="chip chip-amber">
                {f}
              </span>
            ))}
          </div>
          <table className="data-table">
            <thead>
              <tr>
                <th scope="col">Feature set</th>
                <th scope="col">CV macro-F1</th>
                <th scope="col">Test macro-F1</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <th scope="row">A — all {card.data.features.length} (deployed)</th>
                <td>{f3(ablation.cv_macro_f1_with_A)}</td>
                <td className="cell-strong">{f3(ablation.test_macro_f1_with_A)}</td>
              </tr>
              <tr>
                <th scope="row">B — without the four flags</th>
                <td>{f3(ablation.cv_macro_f1_without_B)}</td>
                <td className="cell-strong">{f3(ablation.test_macro_f1_without_B)}</td>
              </tr>
              <tr>
                <th scope="row">Δ (B − A)</th>
                <td className="cell-negative">{f3(ablation.delta_cv_B_minus_A)}</td>
                <td className="cell-negative">{f3(ablation.delta_test_B_minus_A)}</td>
              </tr>
            </tbody>
          </table>
          <p className="note note-amber">{ablation.verdict}</p>
        </section>
      )}

      <div className="info-two-col">
        <section className="panel">
          <header className="panel-header">
            <h2>Evaluation protocol</h2>
          </header>
          <dl className="kv-list">
            <div>
              <dt>Cross-validation</dt>
              <dd>{card.protocol.cv}</dd>
            </div>
            <div>
              <dt>Selection</dt>
              <dd>{card.protocol.selection}</dd>
            </div>
            <div>
              <dt>Terminal test</dt>
              <dd>{card.protocol.test}</dd>
            </div>
            <div>
              <dt>Reported metrics</dt>
              <dd>{card.protocol.metrics.join(", ")}</dd>
            </div>
          </dl>
        </section>

        <section className="panel">
          <header className="panel-header">
            <h2>Dataset</h2>
          </header>
          <dl className="kv-list">
            <div>
              <dt>Name</dt>
              <dd>{card.data.name}</dd>
            </div>
            <div>
              <dt>Rows</dt>
              <dd>
                {card.data.rows_total} total → {card.data.train_rows} train /{" "}
                {card.data.test_rows} test ({card.data.split})
              </dd>
            </div>
            <div>
              <dt>Target</dt>
              <dd>
                {card.data.target_column} = {card.data.target_classes.join(" / ")}
              </dd>
            </div>
            <div>
              <dt>License</dt>
              <dd>{card.data.license}</dd>
            </div>
            <div>
              <dt>Links</dt>
              <dd>
                <a href={card.data.url} target="_blank" rel="noreferrer">
                  UCI dataset
                </a>
                {" · "}
                <a href={card.data.doi} target="_blank" rel="noreferrer">
                  DOI
                </a>
              </dd>
            </div>
          </dl>
          <div className="citation-box">
            <span className="metric-label">Citation</span>
            <p>{card.data.citation}</p>
            <button type="button" className="btn btn-ghost" onClick={() => void copyCitation()}>
              {copied ? "Copied ✓" : "Copy citation"}
            </button>
          </div>
        </section>
      </div>

      <section className="panel">
        <header className="panel-header">
          <h2>Environment &amp; notes</h2>
        </header>
        <div className="chip-cloud">
          {Object.entries(card.environment).map(([k, v]) => (
            <span key={k} className="chip chip-blue">
              {k} {v}
            </span>
          ))}
        </div>
        <ul className="notes-list">
          {card.notes.map((n) => (
            <li key={n}>{n}</li>
          ))}
        </ul>
      </section>
    </div>
  );
}
