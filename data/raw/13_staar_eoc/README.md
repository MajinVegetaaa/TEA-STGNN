# STAAR EOC campus downloads

Original TEA downloads, grouped by **data year**. Files are not edited. The
usable sources are registered under `staar_eoc` in `configs/data_sources.yaml`.
They are integrated into the master table; model-input promotion is gated separately.

| Data year | File | Format | Audit |
|---|---|---|---|
| 2018 | `2018/CAMPSTAAR2.dat` | Advanced TAPR CSV, 2,157 columns | 8,759 unique campuses; primary groups including All Students; published rates, denominators and numerators; 2017 comparison fields excluded |
| 2019 | `2019/staar_eoc_2019.xls` | SAS HTML table with `.xls` extension | EOC rates coded 2019 |
| 2020 | — | — | Spring/summer STAAR cancelled; leave missing |
| 2021 | `2021/staar_eoc_2021.csv` | CSV | EOC rates coded 2021 |
| 2022 | `2022/staar_eoc_2022.csv` | CSV | EOC rates coded 2022 |
| 2023 | `2023/staar_eoc_2023.csv` | CSV | EOC rates coded 2023 |
| 2024 | `2024/staar_eoc_2024.csv` | CSV, descriptive header followed by code header | EOC fields coded 2024; parse with second row as column codes |

`reference_only/all_grades/` contains All Grades tables for 2017–2019.
`reference_only/mislabeled/staar_eoc_2018.xls` is identical to the 2019 EOC file
and contains 2019 fields. Never join these reference files as 2018 EOC.

`2018/CAMPSTAAR5.dat` is preserved in its original location, but staged under
`not_yet_used.staar_additional_groups`: it has **no All Students fields**.
The previous inventory verified its year without verifying the needed population.

All years use published `CDA00{AA1,AR1,AR2,ABI,AUS}12{yy}R` rates. `AA0`
(Algebra I and accelerated testers) is excluded. Never reconstruct a masked
published rate from counts. See [definition and download evidence](../source_definitions/README.md).
