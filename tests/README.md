# tests/ — automated tests (Phase 8)

**Not written yet.** Placeholder.

Planned test layers:
1. **Preprocessing tests** — pipeline output shapes, no NaNs, categories
   handled, fitted only on train folds
2. **Feature-set tests** — H1 features must never contain semester columns
   (guards the leakage audit from Phase 2)
3. **API contract tests** — valid/invalid payloads against `/predict`
4. **Smoke test** — registry model loads and returns probabilities summing to 1

Runner (Phase 8): `python -m pytest`.
