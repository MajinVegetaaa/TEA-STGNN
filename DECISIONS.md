# Decisions log

Format: date — decision — who decided.

- 2026-10-15 — Temporal split: train 2016–2022, validate 2023, test 2024; seeds 0–4 (configs/split.yaml) — team plan
- 2026-10-15 — Prediction CSV format (hand-off 3): `model,seed,campus_id,year,target,y_true,y_pred`; campus_id is 9-digit text — team plan
- 2026-10-15 — Data covers 2019–2023 only (Dataset.zip); split.yaml years must be revised (e.g. train ≤2021, val 2022, test 2023) — pending team agreement
- 2026-10-15 — Cohort = Directory REGULAR INSTRUCTIONAL, non-AEA, grade range includes 12, graduates reported in ≥1 year → 1,653 campuses (see DATA_PIPELINE.md) — data pipeline; confirm with professor
- 2026-10-15 — ACT composite is code CAA (CAC = ACT Science); earlier audit dictionary was wrong — verified from 2023 description row and old file
- 2026-10-15 — Temporal features use only years < t on row t — data pipeline
- 2026-10-15 — Graph defaults (pending professor Q10): geographic k=5 within 100 km, weight exp(-km/20), virtual campuses excluded; academic 10-NN on z-scored t-1 profile (7 indicators, ≥4 shared), weight exp(-d²/2σ²) with σ = median edge distance — data pipeline
- 2026-10-15 — Demographic and transition edges not built: no composition or feeder data in the folder — data pipeline
- 2026-10-15 — Split: train 2019–2021, validate 2022, test 2023 (configs/split.yaml) — user
- 2026-10-15 — Graph parameters kept as defaults in configs/graph.yaml, flagged pending professor review — user
- 2026-10-15 — Academic edges built per target family (predicted target excluded from similarity profile; 6 edge sets) — user, on recommendation
- 2026-10-15 — PPT check: retention (slide 2 "Grade-Level Retention") and demographics (slide 2 "Student Data", slides 3/5/7) ARE listed in the PPT but not in Dataset.zip → still to download — verified from PPT
- 2026-10-20 — Added TAPR campus Profile (demographics, file Y → year Y) and Grade-Level Retention (file (Y−1)(Y) → year Y; AG rows) — data pipeline, verified by campus-list match
- 2026-10-20 — Demographic edges built from year t−1 composition (10-NN, ≥6 shared, log enrollment); parameters pending professor review — data pipeline
- 2026-10-20 — Academic Risk Index not computed: Eq. 1 weights are qualitative only; ask professor for numeric weights — data pipeline
- 2026-10-20 — Academic Risk Index built as a corrected version of last year's phase2 code: correct risk direction for dropout/retention, missing kept missing (≥50% weight coverage), fixed default weights (no circular tuning), decline vs prior years only, location dropped; weights pending professor review — user approved corrected-only
