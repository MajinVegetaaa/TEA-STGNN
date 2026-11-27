# Plan A results: 2022–2024

**Post-test ablation.** The official S6 2024 result had already been observed before this Plan A comparison. These 2024 scores do not replace or change S6.

Plan A uses a shrinkage anchor and relative-change labels (`target_mode=change_relative`, `use_anchor=True`). Each origin is selected only on the prior year; seeds are 0–4; metrics use common eligible campuses per target and MAE divided by persistence MAE. Lower is better.

Models: V0 is the temporal GRU; V1 is relation-aware GATv2; V1 Shared-4 restricts V1 to the four shared relations; V2 adds the zero-initialized ReZero gate; V3 shares an encoder across targets with separate scalar heads.

## Year × model macro results

| year | model | macro_normalized_mae_mean | macro_normalized_mae_sd | seeds |
| ---- | ----- | ------------------------- | ----------------------- | ----- |
| 2022 | v0    | 1.0281                    | 0.0321                  | 5     |
| 2022 | v1    | 0.9955                    | 0.0236                  | 5     |
| 2022 | v1s4  | 0.988                     | 0.0099                  | 5     |
| 2022 | v2    | 1.0229                    | 0.0197                  | 5     |
| 2022 | v3    | 0.9583                    | 0.0034                  | 5     |
| 2023 | v0    | 0.9975                    | 0.02                    | 5     |
| 2023 | v1    | 1.028                     | 0.0515                  | 5     |
| 2023 | v1s4  | 1.0063                    | 0.0339                  | 5     |
| 2023 | v2    | 0.9819                    | 0.0015                  | 5     |
| 2023 | v3    | 0.981                     | 0.0014                  | 5     |
| 2024 | v0    | 1.0902                    | 0.0071                  | 5     |
| 2024 | v1    | 1.1026                    | 0.0161                  | 5     |
| 2024 | v1s4  | 1.0717                    | 0.0118                  | 5     |
| 2024 | v2    | 1.0304                    | 0.0007                  | 5     |
| 2024 | v3    | 1.0198                    | 0.0002                  | 5     |

## Three-year model averages

| model | 2022   | 2023   | 2024   | mean_2022_2024 |
| ----- | ------ | ------ | ------ | -------------- |
| v3    | 0.9583 | 0.981  | 1.0198 | 0.9864         |
| v2    | 1.0229 | 0.9819 | 1.0304 | 1.0117         |
| v1s4  | 0.988  | 1.0063 | 1.0717 | 1.022          |
| v0    | 1.0281 | 0.9975 | 1.0902 | 1.0386         |
| v1    | 0.9955 | 1.028  | 1.1026 | 1.042          |

## Target-wise consolidated comparison

| year | target                  | v0     | v1     | v1s4   | v2     | v3     | best |
| ---- | ----------------------- | ------ | ------ | ------ | ------ | ------ | ---- |
| 2022 | act_composite           | 0.9204 | 0.957  | 0.9333 | 0.9169 | 0.9281 | v2   |
| 2022 | attendance_rate         | 1.3617 | 1.3543 | 1.2564 | 1.7176 | 1.3208 | v1s4 |
| 2022 | dropout_9_12            | 0.8737 | 0.8945 | 0.9263 | 0.8483 | 0.9193 | v2   |
| 2022 | graduation_4yr          | 0.939  | 0.9838 | 0.9283 | 0.917  | 0.9095 | v3   |
| 2022 | retention_rate          | 0.8955 | 0.8153 | 1.0128 | 0.7777 | 0.8484 | v2   |
| 2022 | risk_index              | 0.9109 | 0.9095 | 0.9097 | 0.9341 | 0.8945 | v3   |
| 2022 | sat_act_above_criterion | 1.205  | 1.0265 | 0.9935 | 1.0021 | 0.9417 | v3   |
| 2022 | sat_act_participation   | 1.0052 | 0.9406 | 0.9207 | 0.931  | 0.8902 | v3   |
| 2022 | sat_total               | 1.1629 | 1.1161 | 1.0011 | 1.1969 | 0.9704 | v3   |
| 2022 | tsi_both                | 1.0065 | 0.9569 | 0.9975 | 0.9878 | 0.96   | v1   |
| 2023 | act_composite           | 0.9489 | 0.9558 | 0.9331 | 0.9341 | 0.9385 | v1s4 |
| 2023 | attendance_rate         | 1.1395 | 1.2862 | 1.0904 | 1.2022 | 1.1918 | v1s4 |
| 2023 | dropout_9_12            | 1.0288 | 0.999  | 1.103  | 0.9497 | 0.9716 | v2   |
| 2023 | graduation_4yr          | 1.0232 | 1.0194 | 1.0087 | 0.9751 | 0.9576 | v3   |
| 2023 | retention_rate          | 1.0325 | 1.1043 | 1.1294 | 0.9732 | 0.9713 | v3   |
| 2023 | risk_index              | 0.8811 | 0.8811 | 0.8811 | 0.8823 | 0.8808 | v3   |
| 2023 | sat_act_above_criterion | 0.9431 | 0.9444 | 0.9335 | 0.9493 | 0.9427 | v1s4 |
| 2023 | sat_act_participation   | 1.0169 | 1.08   | 1.0377 | 0.9784 | 0.9794 | v2   |
| 2023 | sat_total               | 0.9667 | 1.0018 | 0.958  | 0.983  | 0.9798 | v1s4 |
| 2023 | tsi_both                | 0.9947 | 1.008  | 0.9878 | 0.9916 | 0.997  | v1s4 |
| 2024 | act_composite           | 0.9345 | 0.9434 | 0.9389 | 0.9489 | 0.9342 | v3   |
| 2024 | attendance_rate         | 1.4673 | 1.5682 | 1.355  | 1.0628 | 1.0646 | v2   |
| 2024 | dropout_9_12            | 1.1148 | 1.1184 | 1.1148 | 1.0543 | 1.0546 | v2   |
| 2024 | graduation_4yr          | 0.9536 | 0.9687 | 0.9437 | 0.9429 | 0.9432 | v2   |
| 2024 | retention_rate          | 1.3918 | 1.2863 | 1.3178 | 1.2153 | 1.2163 | v2   |
| 2024 | risk_index              | 0.8928 | 0.8909 | 0.8909 | 0.8885 | 0.8978 | v2   |
| 2024 | sat_act_above_criterion | 0.9456 | 0.9404 | 0.9365 | 0.9716 | 0.9255 | v3   |
| 2024 | sat_act_participation   | 1.2114 | 1.2456 | 1.2332 | 1.2006 | 1.1652 | v3   |
| 2024 | sat_total               | 0.9595 | 1.0209 | 0.9797 | 0.9845 | 0.9618 | v0   |
| 2024 | tsi_both                | 1.0307 | 1.0437 | 1.0067 | 1.0341 | 1.035  | v1s4 |

## Seed and target details

See `per_seed_metrics.csv` and `summary_by_target.csv`. Paired bootstrap tables compare seed-averaged predictions (2,000 paired draws, seed 0).

## Conclusions

Lowest three-origin descriptive macro ratio: **v3** (0.9864). Compare individual years in `summary_by_year_model.csv`; three origins are not independent test samples.

## Limitations

2022–2024 are historical rolling origins, not independent test samples. The 2024 analysis is post-test. Five seeds describe optimization variability, not independent draws from the Texas school population.
