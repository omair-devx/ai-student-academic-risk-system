import { useCallback, useEffect, useState } from "react";
import {
  ApiRequestError,
  fetchHealth,
  fetchModelInfo,
  type Health,
  type ModelCard,
} from "./api";
import { ModelInfo } from "./components/ModelInfo";
import { PredictForm } from "./components/PredictForm";

type Tab = "predict" | "model";
type LoadState = "loading" | "ready" | "error";

export default function App() {
  const [tab, setTab] = useState<Tab>("predict");
  const [loadState, setLoadState] = useState<LoadState>("loading");
  const [health, setHealth] = useState<Health | null>(null);
  const [card, setCard] = useState<ModelCard | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);

  const load = useCallback(async () => {
    setLoadState("loading");
    try {
      const [h, c] = await Promise.all([fetchHealth(), fetchModelInfo()]);
      setHealth(h);
      setCard(c);
      setLoadState("ready");
    } catch (err) {
      setLoadError(
        err instanceof ApiRequestError
          ? err.message
          : "Unexpected error while loading the dashboard.",
      );
      setLoadState("error");
    }
  }, []);

  useEffect(() => {
    void load();
    // Health polling keeps the status pill truthful across API restarts
    // (switching models requires a backend restart — backend/README.md).
    const id = window.setInterval(() => {
      void fetchHealth()
        .then(setHealth)
        .catch(() => setHealth(null));
    }, 30_000);
    return () => window.clearInterval(id);
  }, [load]);

  return (
    <div className="app">
      <header className="topbar">
        <div className="topbar-inner">
          <div className="brand">
            <svg
              className="brand-logo"
              viewBox="0 0 32 32"
              aria-hidden="true"
            >
              <rect width="32" height="32" rx="7" fill="#4f46e5" />
              <path
                d="M8 21.5 16 9l8 12.5h-4.4L16 15.8l-3.6 5.7Z"
                fill="white"
              />
              <rect x="8" y="23.4" width="16" height="2.4" rx="1.2" fill="white" />
            </svg>
            <div className="brand-text">
              <h1>Student Academic Risk</h1>
              <p>Early-warning dashboard · UCI #697</p>
            </div>
          </div>

          <nav className="tabs" aria-label="Dashboard sections">
            <button
              type="button"
              className={`tab${tab === "predict" ? " tab-active" : ""}`}
              aria-current={tab === "predict" ? "page" : undefined}
              onClick={() => setTab("predict")}
            >
              Predict
            </button>
            <button
              type="button"
              className={`tab${tab === "model" ? " tab-active" : ""}`}
              aria-current={tab === "model" ? "page" : undefined}
              onClick={() => setTab("model")}
            >
              Model &amp; metrics
            </button>
          </nav>

          <div
            className={`status-pill${health ? "" : " status-down"}`}
            title={
              health
                ? `API online — serving ${health.model_version}`
                : "API unreachable"
            }
          >
            <span className="status-dot" aria-hidden="true" />
            {health ? `API · ${health.horizon}` : "API offline"}
          </div>
        </div>
      </header>

      <main className="page">
        {loadState === "loading" && (
          <div className="panel center-panel" aria-live="polite">
            <span className="spinner" aria-hidden="true" />
            <p>Loading model card…</p>
          </div>
        )}

        {loadState === "error" && (
          <div className="panel center-panel error-panel" role="alert">
            <h2>Backend unreachable</h2>
            <p>{loadError}</p>
            <p className="muted">
              Start it from the project root with{" "}
              <code>python -m backend.app.main</code>.
            </p>
            <button
              type="button"
              className="btn btn-primary"
              onClick={() => void load()}
            >
              Retry
            </button>
          </div>
        )}

        {loadState === "ready" && card && tab === "predict" && (
          <PredictForm card={card} />
        )}
        {loadState === "ready" && card && tab === "model" && (
          <ModelInfo card={card} />
        )}
      </main>

      <footer className="footer">
        <p>
          Educational support tool only — not for automated decision-making
          (docs/ARCHITECTURE.md §9). All API inputs are synthetic; metrics come
          from the public UCI #697 dataset.
        </p>
      </footer>
    </div>
  );
}
