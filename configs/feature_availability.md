# Availability policy for the first expansion

The working forecast cutoff is **August 1 of Y−1 for target school-year end Y**.
This assumption was adopted on continuation of the approved plan. It applies to
the added inputs; existing inputs, graph relations and last-year persistence are
preserved as retrospective comparators. This experiment is not an archived-vintage
or fully deployable forecast backtest. Downloaded files can contain later revisions.

`feature_availability.yaml` contains conservative upper bounds, not fabricated
exact release dates. A source year without a bound is excluded. Staff and STAAR
start at lag 2; AP/IB and graduation at lag 3. Each individual source year must
also pass its publication bound. Thus a lag alone does not establish eligibility.
All dates within a publication month are rounded to month end.

## Evidence

- AP/IB: [TEA annual reports](https://tea.texas.gov/data-reports/school-performance/accountability-research/advanced-placement-and-international-baccalaureate)
  date the 2018–2024 editions December 2019, October 2020, September 2021,
  July 2022, August 2023, February 2025 (revised 2023 edition), and December 2025.
- Graduation: [TEA completion, graduation and dropout reports](https://tea.texas.gov/data-reports/school-performance/accountability-research/completion-graduation-and-dropout-reports)
  date the corresponding 2018–2024 reports September 2019, December 2020,
  August 2021, October 2022, August 2023, August 2024 and October 2025.
- Staff and STAAR: bounds use July 31 following the report year, deliberately
  later than the fall/winter TAPR publication cycle. Evidence includes the
  [2018 release notice](https://tea.texas.gov/school-and-district-leaders/accountability/academic-accountability/performance-reporting/december-21.pdf),
  [2019 release](https://content.govdelivery.com/accounts/TXTEA/bulletins/2711ff2),
  [2020 release calendar](https://content.govdelivery.com/accounts/TXTEA/bulletins/2ac22bf),
  [2021 accountability manual](https://tea.texas.gov/school-and-district-leaders/accountability/academic-accountability/performance-reporting/final-2021-accountability-manual.pdf),
  [2022 release calendar](https://content.govdelivery.com/accounts/TXTEA/bulletins/335cb2a),
  [2023 download notice](https://content.govdelivery.com/accounts/TXTEA/bulletins/3aa0e83),
  and [2024 release notice](https://tea2.tea.texas.gov/taa-letters/2023-24-texas-academic-performance-reports-release).
  These establish report availability, not the revision vintage of each downloaded
  campus extract. STAAR 2020 has no bound or observations.

## Interpretation

Raw lag 1–3, change, mean and trend remain available for audit. Models use only
the separate `available_*` versions of new fields. These have three consecutive
calendar lags beginning at the group's minimum lag, with complete-observation
requirements for change/mean/trend. Early-year features often have no training
observations and are dropped by training-only preprocessing.

Graduation's own model histories use the availability-filtered lags. Its raw lag 1
still determines the common evaluation mask and the retrospective persistence
benchmark. Frozen original fields and academic similarity are unchanged, so their
publication timing is a limitation of all six comparisons, not evidence that they
were known at the assumed issuance cutoff. Finance is in the master table but is not a model input; its availability rules (lag 2, bounds from each
archive's DTUPDATE) are in `finance_availability.yaml`, kept separate so this file's fingerprint in `results/expanded/validation.json` stays valid.
