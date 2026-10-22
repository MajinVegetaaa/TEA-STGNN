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
- 2026-10-20 — Repo pushed to private GitHub MajinVegetaaa/TEA-STGNN — team
- 2026-10-20 — Lightning AI Studio chosen for GPU work (CPU for setup, GPU only for training) — team
- 2026-10-20 — Master plan and project log kept in PROJECT_PLAN.md — team
- 2026-10-22 — COVID fix tested in three target modes (level / relative-to-year-mean / z-score-by-year); results reported as total MAE plus statewide-shift and school-level MAE — Phase 3
- 2026-10-22 — Raw data reorganised into numbered source folders; src/data/audit_raw.py proves each file's year from its contents (63/63 pass) — data lead
- 2026-10-22 — Data extended to 2018–2024 (7 years): 2018 and 2024 SAT/ACT, attendance/dropout, demographics and retention added; 2017 downloaded but not used until its 2016-17 retention is fetched from TEA's per-campus reports — user
- 2026-10-22 — All raw file names and years moved to configs/data_sources.yaml; pipeline, validators, audit and setup read it — data lead
- 2026-10-22 — Older retention masks ("-", "<200", "<0.1") treated as suppressed; SAT total vs ERW+Math allowed ±5 (3 rows in 2018 published 2–4 apart) — data lead
- 2026-10-22 — Split: train 2018–2022, validate 2023, test 2024 (training targets in practice 2019–2022) — user
- 2026-10-22 — Baseline target version chosen per target on the 2023 validation year: level for ACT, SAT, college-ready; relative for dropout; attendance: no model beat persistence on 2023, so persistence is its baseline. 2024 test: ACT −8.6%, college-ready −10.8%, dropout −6.5% vs persistence (significant); SAT −2.1% (not significant); attendance tie — Phase 3
- 2026-10-22 — Phase 3 re-run on 7 targets (added grade retention and Academic Risk Index), with previous-year demographics as context features, the demographic edge layer, and XGBoost (src/baselines/run_experiment.py; risk index has no academic layer, since every academic indicator is one of its components). Supersedes the 5-target numbers above — Phase 3
- 2026-10-22 — Protocol frozen: train 2018–2022, tune/select on 2023 only, refit on 2018–2023, test once on 2024; 5 seeds; MAE/RMSE/R², paired bootstrap (2000 resamples) on identical schools; each target's (target version, model) is the lowest 2023 MAE, chosen by src/eval/summarize.py → results/selection.csv; breakdowns (urbanicity, ESC region, size tertile) → results/breakdowns.csv — Phase 3
- 2026-10-22 — Numbers to beat (2024 test MAE, pick vs persistence, 95% CI excludes 0): ACT 1.148 vs 1.276 (−10.2%, level, RF +context +graph); college-ready 5.91 vs 6.75 (−12.7%, level, RF +context); dropout 0.419 vs 0.449 (−6.5%, relative, LR own lags); risk index 6.66 vs 7.67 (−13.3%, relative, RF +context). Not significant: SAT total 30.2 vs 30.8 (−2.0%), retention 1.038 vs 1.071 (−3.1%) — Phase 3
- 2026-10-22 — Gate: attendance does not beat persistence. On 2023 persistence's school-level MAE is 0.95 vs 1.16–1.44 for every model; the 2023 statewide drop (~1.2 points) hits all methods equally. Likely cause: the training years are dominated by pandemic swings (statewide mean +3.3 in 2020, then −3.6 and −2.7), which do not carry over to 2023–24. Persistence stays the attendance baseline; graph models must beat it on attendance or be reported as a tie. Retention and SAT: models ahead of persistence but not significantly (both are already near-saturated, R² 0.87 and 0.82) — Phase 3
- 2026-10-22 — Risk index is already standardised to mean ~50 each year, so level/relative/zscore are expected to agree; its target-version choice is not a finding — Phase 3
- 2026-10-22 — 2016-17 grade-level retention obtained: https://tea.texas.gov/data-reports/school-performance/accountability-research/grade-level-retention/campus-download-1617.xlsx (TEA's 2016-17 page links a retired path; the 2017-18 naming works). Saved unedited as data/raw/04_retention/campus-download-1617.xlsx, listed under not_yet_used; verified title, campus rows and not-copied check. Pipeline still starts at 2018 — adding year 2017 would change the frozen split and results, so it waits for a decision — data lead
