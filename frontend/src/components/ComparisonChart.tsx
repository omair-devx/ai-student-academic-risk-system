import raw from "../data/model_comparison.json";

interface Row {
  horizon: string;
  config_id: string;
  label: string;
  cv_macro_f1: number;
  test_macro_f1: number | null;
  experiment: string;
}

// Snapshot generated from reports/model_comparison.csv (regeneration command
// in frontend/README.md) so the UI ships without any Python dependency.
const ROWS: Row[] = (raw as Row[]).filter((r) => r.experiment === "A");

const HORIZONS: { id: string; title: string; note: string }[] = [
  {
    id: "H1",
    title: "H1 — known at enrollment",
    note: "24 features · earliest warning",
  },
  {
    id: "H2",
    title: "H2 — after semester 1",
    note: "30 features · adds semester-1 results",
  },
];

interface Props {
  activeHorizon: string;
  activeConfigId: string;
}

export function ComparisonChart({ activeHorizon, activeConfigId }: Props) {
  // One shared scale across both panels so heights are comparable.
  const maxCv = Math.max(...ROWS.map((r) => r.cv_macro_f1));

  return (
    <section className="panel">
      <header className="panel-header">
        <h2>Model comparison</h2>
        <p className="muted">
          All pre-declared configs, sorted by 5-fold CV macro-F1 (the selection
          metric). The highlight is the deployed model; hover a bar for its
          terminal test score.
        </p>
      </header>

      <div className="chart-grid">
        {HORIZONS.map(({ id, title, note }) => {
          const rows = ROWS.filter((r) => r.horizon === id).sort(
            (a, b) => b.cv_macro_f1 - a.cv_macro_f1,
          );
          const active = rows.find((r) => r.config_id === activeConfigId);
          return (
            <div key={id} className="chart-panel">
              <div className="chart-panel-head">
                <div>
                  <h3>{title}</h3>
                  <p className="muted small">{note}</p>
                </div>
                {active && (
                  <span className="chart-active-chip">
                    active · test {active.test_macro_f1?.toFixed(3) ?? "—"}
                  </span>
                )}
              </div>
              <ul className="bar-list">
                {rows.map((r) => {
                  const isActive =
                    r.horizon === activeHorizon &&
                    r.config_id === activeConfigId;
                  return (
                    <li
                      key={r.config_id}
                      className={`bar-row${isActive ? " bar-active" : ""}`}
                      title={`${r.label} — CV macro-F1 ${r.cv_macro_f1.toFixed(3)} · test macro-F1 ${
                        r.test_macro_f1?.toFixed(3) ?? "n/a"
                      }`}
                    >
                      <span className="bar-label">{r.label}</span>
                      <span className="bar-track">
                        <span
                          className="bar-fill"
                          style={{ width: `${(r.cv_macro_f1 / maxCv) * 100}%` }}
                        />
                      </span>
                      <span className="bar-value">{r.cv_macro_f1.toFixed(3)}</span>
                    </li>
                  );
                })}
              </ul>
            </div>
          );
        })}
      </div>

      <p className="muted small chart-footnote">
        Source: reports/model_comparison.csv (15 configs × 2 horizons,
        Experiment A). Ablation Experiment B is excluded here and described
        below.
      </p>
    </section>
  );
}
