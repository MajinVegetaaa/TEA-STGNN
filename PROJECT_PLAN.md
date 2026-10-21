# TEA STGNN — Master Plan and Project Log

Last updated: 2026-10-20 · Repo: https://github.com/MajinVegetaaa/TEA-STGNN (private)

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
- **Success** = the proposed model beats every baseline on the 2023 test year, consistently across seeds, and
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

There are eight phases, from data to paper. Phases 1–2 are done; Phase 3 runs on a laptop; Phases 4–7 run on
Lightning AI GPUs. Each phase ends with a **gate**: the condition to meet before relying on its output.

### Phase 1 — Data (done)
Output: `data/processed/` (`master_school_year.csv`, `school_metadata.csv`, `temporal_features.csv`).
- [x] Clean 2018–2024 campus files (2017 on disk for later); standardise IDs; suppression codes → missing values with a flag
- [x] Select 1,670 regular high schools (regular instruction, not alternative accountability, serves grade 12, reported graduates)
- [x] Add geography (NCES coordinates), TAPR demographics, grade-level retention
- [x] History features from earlier years only
- [x] Corrected Academic Risk Index
- [ ] Optional: add 2018-19 retention to fill 2019

Gate: passed (64/64 independent checks).

### Phase 2 — Graph construction (done, 6 of 7)
Output: `data/graph/` (yearly nodes and edge files).
- [x] District, ESC region, geographic, temporal edges
- [x] Academic similarity from the previous year, one set per target with the target left out
- [x] Demographic similarity from the previous year's student make-up
- [ ] Transition edges, only if the professor keeps them in scope
- [ ] Final graph parameters after professor review (`configs/graph.yaml`)

Gate: passed (84/84 checks, including leakage tests).

### Phase 3 — Evaluation protocol and non-graph baselines (laptop)
Output: one scoring script and the "numbers to beat".
- [ ] Fix the COVID distortion for attendance and dropout: year effects, or predict year-over-year change
- [ ] Re-run baselines with all targets (incl. retention, risk index) and demographic features: persistence, LR, KNN, RF, XGBoost
- [ ] Breakdowns: urban/rural, ESC region, school size
- [ ] Freeze the protocol: train 2019–2021, validate 2022, test 2023, 5 seeds, MAE/RMSE/R², paired bootstrap

Gate: baselines beat persistence on every target, or the reason is understood.

### Phase 4 — GPU environment (Lightning AI)
Output: a cloud Studio that reproduces our data and runs a graph model.
- [ ] Studio with Python 3.11, PyTorch, PyTorch Geometric, PyTorch Geometric Temporal; pin versions in `requirements.txt`
- [ ] Clone the repo, upload the raw data, rebuild and validate (Section 4)
- [ ] Convert yearly CSVs to PyTorch Geometric snapshots (`HeteroData` per year)
- [ ] Smoke test: one small GNN trains end to end; time one run to estimate GPU-hours

Gate: validators pass in the cloud and one model trains.

### Phase 5 — Standard STGNN baselines (PPT slide 16)
Output: prediction files for each model in the shared format.
- [ ] GCN + LSTM
- [ ] T-GCN and GConvGRU
- [ ] A3TGCN
- [ ] GraphWaveNet and MTGNN (after the professor confirms which MTGNN)

These take one graph, so run them on the combined edges and on the strongest single relation.

Gate: each tuned on 2022 as carefully as our own model.

### Phase 6 — The proposed model (PPT slide 14)
Output: the model, its predictions and attention weights.
- [ ] Heterogeneous graph: campus, district, region nodes; 6 relation types
- [ ] Spatial encoder: relation-aware graph attention (one attention per edge type) or a Heterogeneous Graph Transformer
- [ ] Temporal encoder: GRU across the yearly snapshots
- [ ] Multi-task heads, one per target
- [ ] Training: masked loss for missing targets, per-target scaling, early stopping on 2022
- [ ] Tune on 2022 only; final run on 2023 with 5 seeds

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
2. Python 3.11 environment: install PyTorch with CUDA, PyTorch Geometric and PyTorch Geometric Temporal; pin the
   working versions in `requirements.txt` and commit it.
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

All results are on the 2023 test year, averaged over 5 seeds, with every model scored on the same schools.

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
- Split: train 2019–2021, validate 2022, test 2023, 5 seeds — team.
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
- [ ] **Graph settings:** preferred neighbour counts, distance limit, similarity measure, or tune on 2022?
- [ ] **Risk Index weights:** accept last year's defaults, or supply the tuned weights from the original output workbook?
- [ ] **Last year's code:** dropout and retention scored in the wrong direction in the risk index; weight tuning
  is circular; Table 6 scores missing values as 0. How should the paper refer to this?
- [ ] **MTGNN on slide 16:** Wu et al. (KDD 2020) or the student-level Multi-Topology GNN?
- [ ] **More years:** add 2016–2018 to lengthen the history?
- [ ] **Venue and timeline:** target venue, page limit, internal deadlines, authorship.
