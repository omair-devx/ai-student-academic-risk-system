# frontend/ — React dashboard (Phase 7)

Vite 7 + React 19 + TypeScript, hand-rolled CSS design system (no UI
framework, no chart library — everything below is dependency-light on
purpose). Talks to the Phase 6 API; nothing else runs in the browser.

## Quick start (two terminals)

```powershell
# terminal 1 — API (project root)
python -m backend.app.main            # http://127.0.0.1:8000

# terminal 2 — UI
cd frontend
npm install                           # first time only
npm run dev                           # http://localhost:5173
```

Open <http://localhost:5173>. The green pill in the header is the API status
(polled every 30 s); it turns red if the backend goes down.

## Pages

- **Predict** — all 24 H1 enrollment-time fields, grouped into six fieldsets
  (demographics / application path / prior education / family background /
  financial circumstances / macroeconomic context). Coded fields use official
  UCI #697 codebook labels (marital status, courses, qualifications, …),
  cross-checked against the observed data levels so every training code is
  selectable. Binary/flag fields are segmented controls. `Load example` fills
  a synthetic profile (labelled as such — no real student data). The result
  card shows the risk badge, at-risk flag, per-class probability bars,
  response time and the serving model version.
- **Model & metrics** — everything from `GET /model/info`: headline metric
  tiles (CV ± / terminal test / balanced accuracy / ROC-AUC), CV-vs-test and
  per-class tables, the selected configuration + selection rule, the model
  comparison chart (CV macro-F1 for all 30 configs across both horizons),
  feature chips, the uncertain-feature ablation, the evaluation protocol,
  dataset card + citation (copy button), environment and notes.

## How it talks to the API

- `src/api.ts` — typed client for `/health`, `/model/info`, `/predict`.
  Base URL from `VITE_API_URL` (default `http://127.0.0.1:8000`; CORS on the
  backend already allows the Vite dev origin).
- The field list always comes from the served model card — the form cannot
  drift from backend validation. `src/fieldMeta.ts` only decides *rendering*
  (labels, codebook options, hints) plus the synthetic example.
- FastAPI 422 issue arrays are mapped to per-field inline errors; other
  failures surface as a banner.

## Commands

```powershell
npm run dev       # dev server on :5173
npm run build     # tsc --noEmit (type-check) + vite build -> dist/
npm run preview   # serve the production build on :4173
```

Both :5173 and :4173 are in the backend's default `CORS_ORIGINS`; any other
origin needs `CORS_ORIGINS` on the API side or the browser will block calls.

## Regenerating the comparison snapshot

`src/data/model_comparison.json` is a static snapshot of
`reports/model_comparison.csv` (so the UI needs no Python at runtime).
After re-running training, refresh it with:

```powershell
python -c "import csv,json,pathlib;rows=[{'horizon':r['horizon'],'config_id':r['config_id'],'label':r['label'],'cv_macro_f1':round(float(r['cv_macro_f1']),4),'test_macro_f1':round(float(r['test_macro_f1']),4),'experiment':r.get('experiment','A')} for r in csv.DictReader(open(r'reports\model_comparison.csv',encoding='utf-8-sig'))];pathlib.Path(r'frontend\src\data\model_comparison.json').write_text(json.dumps(rows,indent=2)+'\n',encoding='utf-8')"
```

## Verification (Phase 7)

- `npm run build` passes (strict TS + Vite production build).
- Headless-browser DOM dump shows the full UI mounted with the live model
  card (form groups, buttons, empty result state; no error panels).
- 8-point live contract check against the running stack: `/health`,
  `/model/info`, example `/predict` (probs sum to 1, version match),
  422 for missing/extra fields (issue-array shape the client parses),
  CORS for the dev origin, dev-server shell — all green.

## Notes

- Educational-support tool only — the footer disclaimer mirrors the API
  description and `docs/ARCHITECTURE.md` §9.
- If the active model is switched (`ml.registry.set_current`), restart the
  backend; the dashboard picks up the new card/version on its next load.
