# data/ — dataset storage

Rules for this directory:

1. **`raw/` is immutable.** Files under `raw/` are the exact bytes downloaded
   from the official source (zips kept, extracted copies untouched). Nothing in
   this project may edit them — cleaning/encoding results go to `processed/`.
2. **Never hand-edit a CSV.** If data is wrong, fix it in code so the fix is
   reproducible.
3. **`raw/` is gitignored** (re-downloadable from official URLs below);
   this README documents provenance so anyone can restore it.

## Inventory & provenance

### UCI #697 — Predict Students' Dropout and Academic Success (PRIMARY)
- Official page: https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success
- Download URL: https://archive.ics.uci.edu/static/public/697/predict+students+dropout+and+academic+success.zip
- DOI: 10.24432/C5MC89 · License: **CC BY 4.0** · Downloaded: 2026-09-29
- Files:
  | File | Size | SHA256 |
  |---|---|---|
  | `raw/uci697/uci697.zip` | 533,344 B | `E90E55FD65EC462AE283EBEB2CCA409319E3460ED898D8754F62FB35CC83A65D` |
  | `raw/uci697/data.csv` | 533,230 B | `3EF126DE5CEFFF26EB11FBB4237F1A1401CB64B488E2F1D598C23CEDEB4C45AE` |
- Verified shape: **4,424 rows × 37 columns** (matches UCI metadata)
- Format quirk: **semicolon-separated** with `utf-8-sig` BOM; one header cell
  contains a stray tab (`Daytime/evening attendance\t`) — cleaned on load only

### UCI #320 — Student Performance (COMPARISON)
- Official page: https://archive.ics.uci.edu/dataset/320/student+performance
- Download URL: https://archive.ics.uci.edu/static/public/320/student+performance.zip
- DOI: 10.24432/C5TG7T · License: **CC BY 4.0** · Downloaded: 2026-09-29
- Files:
  | File | Size | SHA256 |
  |---|---|---|
  | `raw/uci320/uci320.zip` | 40,735 B | `82AE9D66437B9808DF42E8C89D2BB179C46E9CFBCF06F38ABC1D20B3B747E177` |
  | `raw/uci320/student/student-por.csv` | 93,220 B | `A7594A11D7771C0EFE1A740824E0E833DA9C4CAD07C39A9766A874575563FB3F` |
  | `raw/uci320/student/student-mat.csv` | 56,993 B | `E47F9EE225E1EE6E69B7564E6DAC7123E80B8486677FE111F351964CEF5DEC80` |
- Verified shapes: student-por **649 × 33**, student-mat **395 × 33**
- Format: semicolon-separated, no BOM

## License / attribution
Both datasets are **CC BY 4.0**: free to use and adapt for any purpose with
attribution. Required citations are mirrored in the root `README.md`.

## Checksums
Recompute with:
```powershell
Get-FileHash -Algorithm SHA256 data\raw\uci697\data.csv
```
If a hash no longer matches this file, the raw data changed — treat as an error,
re-extract from the original zip (do not patch the CSV).
