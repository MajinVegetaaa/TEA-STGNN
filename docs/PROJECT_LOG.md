# TEA STGNN — Master Plan and Project Log (history)

> **Superseded as the plan on 2026-11-21 by [`ROADMAP.md`](../ROADMAP.md).** Kept unchanged as the record of
> Phases 1–5 (was `PROJECT_PLAN.md` at the repo root).

Last updated: 2026-10-20 · Repo: https://github.com/MajinVegetaaa/TEA-STGNN (public)

## 1. What this project is

**We forecast next year's outcomes for every Texas public high school (SAT, ACT, college readiness, dropout,
retention, attendance and a risk index), using a graph that connects related schools.** The research claim: a
school's future is predicted better when the model also learns from related schools than when each school is
predicted on its own.

- **The problem.** Texas has about 1,650 regular public high schools. Each year the Texas Education Agency (TEA)
  publishes their test scores, college-readiness rates, dropout, retention, attendance and student make-up.
  Forecasting next year's numbers lets struggling schools be flagged early.
- **What last year's work did.** The professor's paper (Lanka et al., IEEE ToE 2026) predicted each school from
  its own past three years with linear regression, random forest, XGBoost and KNN, plus a risk index and LLM
  recommendations. Each school was treated as if it existed alone. Its "future work" section names
  spatio-temporal graph neural networks and graph attention. This project is that step.
- **Our idea.** Schools are not isolated: districts share budgets and policies, nearby schools share a local
  economy, and schools with similar results or students tend to move together. We connect schools with **seven
  link types** and rebuild the graph every year:
  1. district
  2. geographic
  3. academic similarity
  4. demographic similarity
  5. academic transition
  6. ESC region
  7. the same school across years
- **The model** is a Spatio-Temporal Graph Neural Network (STGNN):
  - a graph attention network learns how much to listen to each linked school (spatial)
  - a GRU tracks how each school changes year to year (temporal)
  - multi-task heads predict all outcomes together
- **Why it is new.** Earlier graph models in education (APP-TGN, APP-DGNN) predict individual students in online
  courses. We predict institutions statewide, with a campus–district–region hierarchy, typed relationships and
  yearly graphs.
- **Success** = the proposed model beats every baseline on the 2024 test year (2023 is the validation year; corrected 2026-11-21), consistently across seeds, and
  ablations show which relationships matter.

## 2. Where we are

| Area | Status | Key numbers |
|---|---|---|
| Dataset (2018–2024) | Done | 1,670 high schools, 11,546 school-years; 89/89 raw-audit, 65/65 data and 86/86 graph checks |
| Targets | Done | SAT, ACT, college-ready, participation, attendance, chronic absenteeism, dropout, retention (2020–23), risk index |
| Student demographics | Done | 13 TAPR fields, matched for 99–100% of school-years |
| Academic Risk Index | Done, weights pending review | corrected version of last year's Eq. 1 |
| Yearly graphs | 6 of 7 relations | district, geographic, academic, demographic, region, temporal; 84/84 checks |
| Transition edges | Waiting on professor | needs feeder-pattern data TEA does not publish |
| First baselines | Done | models beat "same as last year" by 6–14% on ACT, SAT, college-ready |
| Graph neural networks | Not started | needs PyTorch Geometric and a GPU (Lightning AI) |
| Paper | Not started | IEEE format |

## 3. Master plan

There are eight phases, from data to paper. Phases 1–3 are done (all on a laptop); Phases 4–7 run on
Lightning AI GPUs. Each phase ends with a **gate**: the condition to meet before relying on its output.

### Phase 1 — Data (done)
Output: `data/processed/` (`master_school_year.csv`, `school_metadata.csv`, `temporal_features.csv`).
- [x] Clean 2018–2024 campus files (2017 on disk for later); standardise IDs; suppression codes → missing values with a flag
- [x] Select 1,670 regular high schools (regular instruction, not alternative accountability, serves grade 12, reported graduates)
- [x] Add geography (NCES coordinates), TAPR demographics, grade-level retention
- [x] History features from earlier years only
- [x] Corrected Academic Risk Index
- [x] Add 2018-19 retention to fill 2019
- [x] Fetch 2016-17 retention (on disk, not yet in the pipeline: adding 2017 as a data year needs a team decision)

Gate: passed (64/64 independent checks).

### Phase 2 — Graph construction (done, 6 of 7)
Output: `data/graph/` (yearly nodes and edge files).
- [x] District, ESC region, geographic, temporal edges
- [x] Academic similarity from the previous year, one set per target with the target left out
- [x] Demographic similarity from the previous year's student make-up
- [ ] Transition edges, only if the professor keeps them in scope
- [ ] Final graph parameters after professor review (`configs/graph.yaml`)

Gate: passed (84/84 checks, including leakage tests).

### Phase 3 — Evaluation protocol and non-graph baselines (laptop, done)
Output: one scoring script and the "numbers to beat".
- [x] Fix the COVID distortion: three target versions (raw / vs yearly mean / vs yearly mean and spread), chosen per target on 2023; plus the 2018–2024 data, which makes COVID 2 of 4 training years
- [x] Re-run baselines with all targets (incl. retention, risk index) and demographic features: persistence, LR, KNN, RF, XGBoost
- [x] Breakdowns: urban/rural, ESC region, school size (`results/breakdowns.csv`)
- [x] Freeze the protocol: train 2018–2022, validate 2023, test 2024, 5 seeds, MAE/RMSE/R², paired bootstrap

Gate: baselines beat persistence on every target, or the reason is understood.
Result: beat on ACT, college-ready, dropout, risk index (significant). SAT and retention: ahead but not significant. Attendance: no model beats persistence — see `DECISIONS.md` (2026-10-22). Numbers to beat: `results/selection.csv`.
Added 2026-10-25: TSI (`tsi_both`) from TAPR CCMR files; persistence wins on validation, the selected model beats it on test by 2.6% (significant). Phase 3 numbers for the other targets unchanged (`DECISIONS.md`).

### Phase 4 — GPU environment (Lightning AI)
Output: a cloud Studio that reproduces our data and runs a graph model.
- [x] Pin versions: `requirements.txt` (pipeline + baselines) and `requirements-gpu.txt` (torch 2.12.1, PyG, PyG Temporal);
  Python 3.13, the laptop's version (3.12 also works; not 3.11: xgboost 3.4.1, used for the Phase 3 numbers, needs ≥3.12).
  Fresh 3.13 environment: 89 / 65 / 86 / 60 checks pass and Phase 3 ACT predictions match the laptop's (max difference 3e-14).
  torch 2.12.1, not the newest: PyG Temporal needs `torch_sparse`, which has builds for 2.12.1 but none for 2.13+.
- [x] Studio: Python 3.13, torch from the CUDA wheel index, then both requirements files (steps below)
- [x] Clone the repo, upload the raw data, rebuild and validate (Section 4): audit 89, data 65, graph 86 checks pass on the Studio
- [x] Convert yearly CSVs to PyTorch Geometric snapshots: `src/graphs/to_pyg.py` → `data/graph/pyg/snapshots.pt`
  (one `HeteroData` per year, fixed 1,670-campus order, 273 inputs, 60/60 self-checks incl. evaluation rows = Phase 3 rows;
  since 2026-10-25 also the TSI target and its history: 8 targets, 288 inputs, 62/62)
- [x] Smoke test on laptop CPU: `src/models/smoke_test.py` (SAGEConv per relation + GRU across years), ACT, 0.12 s/epoch
- [x] Smoke test on the Studio GPU: ACT, 150 epochs in 5.1 s (0.03 s/epoch; Studio CPU 0.61, laptop 0.17);
  validation MAE 1.2052 at epoch 25 (laptop 1.2051), persistence 1.2936. Converter 46/46 there: the 14 Phase 3 row
  checks need the gitignored `results/level/predictions.csv` (upload it for 60/60).

Studio steps (needs the account owner):
1. New Studio (CPU is enough for steps 2–4); Python 3.13 (check `python --version`).
2. `git clone https://github.com/MajinVegetaaa/TEA-STGNN.git && cd TEA-STGNN`
3. Upload `data/raw/` (not in git; incl. `07_tsi/` since 2026-10-25) to the same path, or upload `Dataset.zip` plus the TAPR,
   retention and TSI folders and run `src/data/setup_raw.py`.
4. `pip install -r requirements.txt`, then run the four rebuild-and-validate commands in the README (expect 113 / 69 / 86 checks; 89 / 65 / 86 before TSI was added on 2026-10-25).
   The rebuilt `edges_academic.csv` breaks one distance tie (2024, excl_criterion) differently from the laptop build
   (depends on the numpy build, see `DECISIONS.md`); for byte-identical Phase 3 inputs, upload the laptop's `data/graph/` too.
5. Switch to a GPU; `pip install torch==2.12.1 --index-url https://download.pytorch.org/whl/cu130` (cu126 if
   `nvidia-smi` shows an older driver), then
   `pip install -r requirements-gpu.txt -f https://data.pyg.org/whl/torch-2.12.1+cu130.html`; `python src/graphs/to_pyg.py` (62/62, or 46/46 plus a SKIP line without `results/level/predictions.csv`);
   `python src/models/smoke_test.py --target act_composite --out /tmp/smoke`.

- [x] Google Colab (used while the Lightning account is pending), T4 runtime, Python 3.13.15, CUDA 13.0: audit 113, data 69,
  graph 84 + the 2 known tie failures (laptop graph uploaded, see `DECISIONS.md`), converter 46/46 with 288 inputs;
  smoke tests 0.03 s/epoch on cuda (ACT, TSI)

Colab steps (every new session: the machine is wiped, only Drive persists):
1. Runtime → Change runtime type → GPU; check `!python --version` and `!nvidia-smi` (cu126 instead of cu130 below if
   its CUDA version is below 13.0).
2. Upload `data_raw.zip` (zip of `data/raw/`) and `graph.zip` (zip of `data/graph/` without `pyg/`) to My Drive. Check each
   with `!unzip -tq` before using it: an unfinished Drive upload gives "End-of-central-directory signature not found".
3. Setup cell (repo is private: clone with a read-only token typed through `getpass`):
   ```
   from google.colab import drive; drive.mount('/content/drive')
   from getpass import getpass; tok = getpass('GitHub token: ')
   !git clone -q https://{tok}@github.com/MajinVegetaaa/TEA-STGNN.git /content/TEA-STGNN
   %cd /content/TEA-STGNN
   !unzip -q /content/drive/MyDrive/data_raw.zip -d data && unzip -q /content/drive/MyDrive/graph.zip -d data
   !git checkout -- data/raw/README.md          # the zip carries a copy of this tracked file
   !pip install -q torch==2.12.1 --index-url https://download.pytorch.org/whl/cu130
   !pip install -q -r requirements.txt -r requirements-gpu.txt -f https://data.pyg.org/whl/torch-2.12.1+cu130.html
   !python src/data/build_master.py > /dev/null && python src/graphs/to_pyg.py | tail -2
   ```
   Don't run `build_graph_data.py` (it would replace the laptop graph). Both installs are needed every session: the
   first picks the CUDA build of torch, the second needs PyG's wheel page for torch_scatter / torch_sparse.
4. Save anything worth keeping (predictions, checkpoints) to Drive; `/content` is lost when the session ends.

Gate: validators pass in the cloud and one model trains. **Passed 2026-10-23** (Studio), **2026-10-30** (Colab).

### Phase 5 — Standard STGNN baselines (PPT slide 16)
Output: prediction files for each model in the shared format.
- [x] GCN + LSTM — local benchmark and Colab GPU summary checked
- [x] T-GCN and GConvGRU — local benchmarks audited
- [x] A3TGCN — local benchmark audited
- [x] GraphWaveNet with fixed campus graph supports — local benchmark audited
- [ ] MTGNN — awaiting the professor’s architecture choice

These take one graph, so run them on the combined edges and on the strongest single relation.

Gate: each tuned on 2023 (validation) as carefully as our own model.

### Phase 6 — The proposed model (PPT slide 14)
Output: the model, its predictions and attention weights.
- [ ] Heterogeneous graph: campus, district, region nodes; 6 relation types
- [ ] Spatial encoder: relation-aware graph attention (one attention per edge type) or a Heterogeneous Graph Transformer
- [ ] Temporal encoder: GRU across the yearly snapshots
- [ ] Multi-task heads, one per target
- [ ] Training: masked loss for missing targets, per-target scaling, early stopping on 2023
- [ ] Tune on 2023 only; final run on 2024 with 5 seeds

Gate: beats the best baseline on the validation year.

### Phase 7 — Experiments and analysis
Output: every table and figure in Section 5.
- [ ] Main comparison table
- [ ] Ablations: each relation removed; static vs yearly graphs; single- vs multi-task; spatial vs temporal
- [ ] Breakdowns by urban/rural, region, size
- [ ] Attention analysis and the at-risk school ranking

Gate: results hold across seeds (paired bootstrap).

### Phase 8 — Paper
- [ ] IEEE-format draft: introduction, related work, data, method, experiments, discussion
- [ ] Each person writes their part; everyone reviews; the professor reviews the draft

## 4. Running the models on Lightning AI

Use one Lightning AI Studio as the shared cloud workspace. Do setup and data work on a cheap CPU machine, and
switch the same Studio to a GPU only for training. Our laptop (Apple M1, 8 GB, no CUDA) stays for writing and
small experiments.

**One-time setup**
1. Create a Studio; start it on a CPU machine.
2. Python 3.13 environment: install torch with CUDA, then `requirements.txt` and `requirements-gpu.txt` (Phase 4 steps).
3. `gh auth login`, then clone `MajinVegetaaa/TEA-STGNN`.
4. Upload `Dataset.zip` and the TAPR and retention download folders into the Studio's persistent storage (never into git); `setup_raw.py` places them in `data/raw/`.
5. Rebuild and check:
   ```bash
   python3 src/data/setup_raw.py --zip Dataset.zip --tapr <TAPR folder> --retention <retention folder>
   python src/data/build_master.py && python src/data/validate_processed.py
   python src/graphs/build_graph_data.py && python src/graphs/validate_graph.py
   ```
   Both validators must pass (64 and 84 checks) before any model is trusted.

**Each training session**
1. Write and debug on CPU with a small subset (one year, one target).
2. Switch to a GPU machine. A 16–24 GB GPU is enough for ~1,650 schools per year.
3. Run with a fixed seed. Every run appends model, seed, settings and scores to `results/runs.csv`.
4. Switch back to CPU, or stop the Studio, when training ends.
5. Commit code and small result files; keep large prediction files in Studio storage.

**Keeping GPU cost down**
- Time one pilot run, then estimate GPU-hours before launching the ~500-run grid.
- Never leave a GPU idle; let the Studio auto-sleep.
- Tune on a small grid first; widen only where results change.

Check current machine types and credit limits in your Lightning AI account; this plan doesn't depend on a
specific GPU.

## 5. Expected experiments and results

All results are on the 2024 test year (corrected 2026-11-21; 2023 is validation), averaged over 5 seeds, with every model scored on the same schools.

| # | Result | What it shows | Phase |
|---|---|---|---|
| 1 | **Main table**: MAE, RMSE, R² per target for every model (persistence, LR, KNN, RF, XGBoost, 6 standard STGNNs, ours) | Whether our model beats every baseline | 7 |
| 2 | **Significance**: paired bootstrap, ours vs best baseline | Whether the gain is real or seed noise | 7 |
| 3 | **Relation ablation**: remove one relation type at a time | Which relationships matter | 7 |
| 4 | **Static vs yearly graph** | Whether rebuilding the graph each year helps | 7 |
| 5 | **Single- vs multi-task** | Whether predicting all outcomes together helps | 7 |
| 6 | **Spatial vs temporal**: GAT only, GRU only, both | Neighbours vs history | 7 |
| 7 | **Breakdowns**: urban/rural, ESC region, size | Where the model helps most | 7 |
| 8 | **Attention analysis**: one school's neighbours; average attention per relation | Interpretability | 7 |
| 9 | **At-risk ranking**: how many forecast highest-risk schools really were | Practical value | 7 |
| 10 | **Data section**: school selection, graph statistics, data-quality findings | Trust in the data | done |

**What to expect, honestly**
- **Baselines are strong.** "Same as last year" is hard to beat with five yearly snapshots. Learned models
  already beat it by only 6–14%.
- **Neighbour averages added about 0%** to a random forest. Whether *learned* attention over typed relations does
  better is the central experiment.
- **Either answer is publishable** if shown rigorously. A clear gain supports the claim. A small or null gain,
  with ablations showing which relations carry signal, is still a solid contribution.
- **COVID years (2020–21)** must be handled in Phase 3, or attendance and dropout results mislead.

## 6. Decision log

Newest first. Add one line per decision: date, decision, who, why. Detailed technical entries are also in
`DECISIONS.md`.

**2026-10-23**
- Phase 4 closed: validators pass on the Studio and the smoke test runs on its GPU — team.
- Smoke test runs on the Studio GPU: 0.03 s/epoch, same validation curve as the laptop up to epoch 50 — team.
- Phase 4 laptop side: Python 3.13 (3.12+; xgboost needs it), torch 2.12.1 so PyG Temporal works, snapshot converter and CPU smoke test done — team.
- 2017 tried as a data year and dropped: unusable columns and no accuracy gain; data stays 2018–2024 and the 2018–2024 Phase 3 results stand — user.

**2026-10-22**
- Data extended to 2018–2024; split changed to train 2018–2022, validate 2023, test 2024 — team.
- All raw file names moved to `configs/data_sources.yaml` — data lead.
- Phase 3 closed: protocol frozen (train 2018–2022, select on 2023, test 2024); numbers to beat in `results/selection.csv`; attendance stays at persistence — team.
- COVID fix: models predict each school's position relative to the yearly statewide average (and spread); results report statewide-shift and school-level error separately — team.

**2026-10-20**
- Repo pushed to private GitHub `MajinVegetaaa/TEA-STGNN` — team.
- Lightning AI Studio chosen for GPU work — team.
- Academic Risk Index built as a corrected version of last year's code. Fixes: risk direction for dropout and
  retention, missing kept missing, fixed default weights instead of circular tuning, decline vs prior years only,
  location dropped. Weights pending professor review — team, after reviewing last year's repo.
- Corrected index only; no faithful copy of the old index unless the professor asks — team.
- Proceed with 6 of 7 relation types; transition edges wait for the professor — team.
- Demographic edges from the previous year's student make-up (10 most similar, 12 features) — data lead.
- TAPR profile and grade-level retention added; years verified by matching campus lists — data lead.

**2026-10-15**
- Split: train 2019–2021, validate 2022, test 2023, 5 seeds — team (superseded 2026-10-22).
- Graph settings kept as defaults in `configs/graph.yaml`, flagged for professor review — team.
- Academic similarity edges per target, with the predicted variable left out of its own profile — team.
- Every feature and edge for year *t* uses only earlier years (no leakage) — data lead.
- Cohort: regular, non-alternative-accountability campuses serving grade 12 that reported graduates, giving
  1,653 schools. Grade 12 rather than full 9–12, to keep senior-high campuses — data lead.
- ACT composite is TEA code `CAA`; `CAC` is ACT Science — verified from TEA's column descriptions.
- Raw TEA files used instead of the old merged Excel, whose SAT values were copied between years — data lead.
- Data covers 2019–2023 only; no years invented — team.

## 7. Open questions for the professor

- [ ] **Transition edges:** keep in scope? Needs middle schools and feeder data TEA doesn't publish (options:
  NCES attendance boundaries, district websites, a labelled proxy).
- [ ] **Graph settings:** preferred neighbour counts, distance limit, similarity measure, or tune on 2023?
- [ ] **Risk Index weights:** accept last year's defaults, or supply the tuned weights from the original output workbook?
- [ ] **Last year's code:** dropout and retention scored in the wrong direction in the risk index; weight tuning
  is circular; Table 6 scores missing values as 0. How should the paper refer to this?
- [ ] **MTGNN on slide 16:** Wu et al. (KDD 2020) or the student-level Multi-Topology GNN?
- [ ] **More years:** add 2016–2018 to lengthen the history?
- [ ] **Venue and timeline:** target venue, page limit, internal deadlines, authorship.

## Source additions verified and staged — 2026-11-16

User downloaded AP, staff and PEIMS actual financial files; organized into numbered raw folders
and registered under `not_yet_used` in `configs/data_sources.yaml`.

- [x] AP files for 2018–2024 downloaded; campus-by-student-group grain. AP only, not IB.
- [x] PEIMS actual files for file years 2018–2024 downloaded; ZIP integrity checked.
- [x] Staff 2024 downloaded; additional CSTAF file presumed 2023, with year confirmation pending.
- [ ] Obtain staff 2018–2022 and confirm CSTAF report year.
- [ ] Obtain IB campus files if the proposed AP/IB feature is intended to include IB.
- [x] Verify finance fiscal-year mapping, release timing, FIN_UNIT campus mapping and charter coverage (done 2026-11-21: see DECISIONS.md; charter archives added).
- [ ] Implement loaders, suppression handling, cohort joins and lagged features; rerun data/graph gates before model integration.
- [ ] Discipline/DAEP remains unresolved; no discipline download was supplied in this batch.

The source audit and SHA-256 inventory are reproducible with
`python src/data/audit_new_downloads.py`. Existing model inputs and results are not expanded by staging these files.

Audit outcome: AP campus coverage is about 67–70%, with unique campus/group keys; staff
2024 covers the full model cohort. Finance files contain multiple FUNDYEAR values per
archive (2018 file: 2010–2019; 2024 file: 2016–2025), so FUNDYEAR must not be
used directly as the observation year. Detailed counts are in the source audit JSON.

## PPT coverage review and integration preparation — 2026-11-16 (second download batch)

Source: `docs/references/TEA_STGNN_Architecture (1).pptx`, all 18 slide text layers reviewed,
compared with the current master/metadata schemas, feature code, split configuration and
chronological decisions (especially the additive TSI integration on 2026-10-25).
This section supersedes the earlier download-gap checklist for staff and IB.

| PPT requirement | Evidence / current status | Next step |
|---|---|---|
| SAT/ACT components, participation, TSI, dropout, retention (slides 5, 11, 14) | Present in master table | Preserve existing definitions and target masks |
| Enrollment, poverty, EL, special education, race, location and campus type (slide 5) | Present in master or metadata | Preserve lagging and document static metadata snapshot dates |
| AP/IB participation and performance (slides 2–3) | Seven AP and seven IB raw workbooks now present | All Students rows; preserve masks; keep AP and IB separate, do not sum overlapping students |
| Teacher experience and student/teacher ratio (slide 3) | Seven staff files now present; core codes checked by audit | Normalize HTML/CSV formats and verify report-year provenance |
| Finance: expenditure per student (slide 3) | Seven actual financial ZIPs present | Verify account filters, campus FIN_UNIT, denominator, charter coverage and release timing before deriving rates |
| STAAR subject performance (slides 3, 17) | Not in current master; no dedicated staged source | Source and define comparable high-school EOC measures if included in scope |
| Graduation rate (slides 3, 17) | Not in current master; dropout is not graduation | Select annual vs four-year cohort definition and audit source availability |
| Accountability domain scores/ratings (slides 3–4, 13, 17) | Not in current master; not supplied in new batch | Verify raw reference fields and year comparability; no assumption that all rating years exist |
| Discipline indicators (slides 2–3) | No new discipline source supplied; prior TAPR selector claim was unsupported | Check actual raw fields and valid downloadable campus series |
| Educator certification, RDA, FIRST (slide 2 source catalogue) | Not verified as available model fields | Catalogue items; decide whether required beyond explicit node-feature specification |
| Moving average, change, trend (slide 5) | Implemented using earlier years only | Extend explicitly to approved new features |
| Acceleration and variance (slide 5) | Not implemented in temporal_features | Proposed feature additions, not missing downloads |
| Campus–district–region node hierarchy (slides 6, 15) | Current converter uses campus nodes with district/region relations | Architecture extension; keep distinct from dataset completeness |
| Transition/student-flow edges (slides 4, 7) | Still awaiting feeder data per decisions | Remains blocked by source availability |
| District ranking, future accountability and other targets (slides 13, 17) | Broader than the agreed current target set | Resolve target scope before changing model heads |

### Existing workflow to retain for the integration plan

1. Freeze reference outputs, cohort and baseline metrics; preserve original raw bytes and register sources.
2. Audit raw inputs in `src/data/audit_raw.py`: report year, actual grain, campus keys, suppression,
   ranges, distinct years and coverage. Resolve staff year provenance and finance mapping first.
3. Add source-specific loaders to `src/data/build_master.py`; left-join on campus/year with
   explicit uniqueness checks. Keep the 2018–2024 panel and cohort; retain missingness and source provenance.
4. Extend `src/data/validate_processed.py`, including independent source-cell checks and
   regression checks that all existing columns retain their values. Run build and validation.
5. Add only past-year history features. Account for publication lag as well as observation year.
   Fit imputation/scaling on train 2018–2022; validate on 2023 and test on 2024 (`configs/split.yaml`).
6. Update graph node feature catalog in `src/graphs/build_graph_data.py`; run
   `src/graphs/validate_graph.py`. Any new similarity relation is an explicit experiment,
   not an automatic consequence of adding a raw field.
7. Update `src/graphs/to_pyg.py`; verify feature roles, masks, fixed node ordering and leakage checks.
   Run `src/models/smoke_test.py` using the established environment.
8. Evaluate the expanded features against the frozen baselines with the same split/seeds and
   comparable evaluation rows; select on validation only. Preserve the original results as reference.
9. Update pipeline, graph documentation, handoff and decision log with actual gate results.

Current action is verification and preparation. Integration implementation and model retraining
are the next phase. Older narrative sections in this plan/handoff contain stale counts and dates;
current configuration/code and the latest dated decisions control this review.

### Graduation/STAAR download update — 2026-11-16

Supersedes the earlier statement that these sources have not been supplied:
seven four-year graduation workbooks (2018–2024) and six named STAAR EOC files arrived.
However, the file labelled 2018 STAAR is byte-identical to the 2019 file and contains
only 2019 rate columns. Valid staged STAAR years: 2019, 2021, 2022, 2023, 2024.
2020 is intentionally absent; actual 2018 data must be downloaded again.
Sources are registered under `not_yet_used`; the mislabeled file is under `rejected_downloads`.
Before graduation integration, choose a consistent state-accountability calculation variant
and verify observation/release-year mapping. These are download checks, not completed model integration.

STAAR replacement check: `CSTAAR_ALL1 (1).xls` contains 2018 data (8,759 unique
campuses; 1,593/1,619 model-campus matches), but is an all-grades table without
the EOC subject codes AA1/AR1/AR2/ABI/AUS. `CSTAAR_ALL1.xls` is all-grades 2019.
Both retained as supplemental sources; 2018 EOC remains missing.


STAAR source resolution: 2018 advanced TAPR campus EOC `CAMPSTAAR5.dat`
verified (8,759 unique campus records, 2,955 columns, year-coded 2018 EOC
numerator/denominator fields). Usable years now 2018, 2019, 2021–2024;
2020 is the only gap. Raw files grouped into year folders; all-grades and
mislabeled files moved to `reference_only/`. 2018 rates require a documented
numerator/denominator transformation and suppression handling.

### Master-data implementation — 2026-11-18

The approved first milestone is complete: 179 raw-source checks, 115 independent processed checks and 11 regression tests pass. The master retains 11,546 rows, 1,670 campuses and all 59 original columns unchanged, with 97 appended value, count, status and provenance columns. See DATA_PIPELINE.md for fields, coverage, commands and receipts.

Correction to the preceding source-resolution entry: CAMPSTAAR5 has no All Students fields. CAMPSTAAR2 was downloaded from TEA, verified, registered and integrated using published rates. The original CAMPSTAAR5 remains staged and unmodified.

Continuation: the cutoff assumption and publication mapping are recorded in
`configs/feature_availability.md`. Expanded calendar histories, unchanged graph
relations and ten-target baseline/PyG wiring are implemented. Graph validation
passes 86 checks, PyG conversion 63, and both new targets pass training smoke tests.
The six approved baseline comparisons use the original split/seeds and validation
selection locks; final outcomes are recorded in `results/expanded/README.md`.
Original graph and baseline files remain separate and preserved.

### Completed evaluation — 2026-11-18

All six input comparisons and fixed test evaluations are complete (60 target/input-set results). Independent validation passed for all models, seeds, eligibility rows and selection locks. All 56 original graph/results files and seven edge files retain their checksums; 48 frozen comparisons use identical rows. The unit suite passes 16/16.

Validation-selected additions improved 2024 MAE for ACT and TSI, worsened it for SAT total and Risk Index, and left six targets unchanged versus the new existing-input comparison. The gains are small (ACT 1.1500 to 1.1479; TSI 14.4148 to 14.3335); these results do not support a general accuracy improvement from adding all sources. See `results/expanded/README.md` and `final_selection_report.csv` for the ten locked choices, coverage and errors. Finance remains staged; no full GNN performance study or historical-vintage forecasting claim is made.

### Phase 5 kickoff — expanded graph

The user reported successful Phase 4 verification on a Colab Tesla T4 with
PyTorch 2.14.1+cu130. The Phase 5 GCN + LSTM runner uses the verified
ten-target snapshot, weighted graph convolution, a 32-unit LSTM and separate
target heads trained as independent models. The validation search compares the
combined graph with each target-eligible single relation and selects an epoch
from 1–30 at seed 0. Both the combined graph and the winning single relation
are refit for seeds 0–4. The locked Phase 3 campus rows and outcomes are checked
during test scoring. See results/phase5/gcn_lstm/README.md for run status.

Phase 5 results are in [one comparison table](../results/phase5/README.md). All five completed CPU benchmarks had higher test MAE than Phase 3 on all ten targets. The uploaded GCN + LSTM Colab summary matched all 20 CPU graph choices and campus counts, with a maximum absolute MAE difference of 0.00126. The other four completed runs used the local CPU. MTGNN awaits clarification.

### Finance integration — 2026-11-21

Finance (slide 3: per-student expenditure) is integrated into the master table: 19 additive columns, 98.1–99.6% of model campuses per year, checked against
68 of TEA's own campus financial reports and by independent recomputation from the raw archives. Raw audit 200, processed validation 127, expanded graph 86/86,
tests 32 across the four test modules (2 skipped). It is **not** a model input: no experiment input, snapshot or stored result changed (`expanded_temporal_features.csv` regenerates
byte-identical; 7 edge files, the PyG snapshot and all 56 frozen artifacts unchanged). The expanded graph's node table and catalog gained 8 same-year columns, so the
`nodes.csv` fingerprint recorded in `results/expanded/validation.json` is now stale (a record only; `validate_expansion.py` passes against the new file; see DECISIONS.md 2026-11-21). Decision needed:
rerun the expanded experiment / Phase 5 with finance as an input (merge `configs/finance_availability.yaml` into `feature_availability.yaml`, add `"finance"` to
`new_sources.FEATURE_GROUPS`), or leave finance as master-only. Slide 3's teacher-similarity and funding-similarity edges are separate graph work.
