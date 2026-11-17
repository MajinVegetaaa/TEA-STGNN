# Definition evidence for the November integration

Downloaded from TEA on 2026-11-17. HTML files preserve the returned dictionary
pages; the audit checks the selected five STAAR fields against these definitions.

| File | Source |
|---|---|
| `staar_2018.html` | https://rptsvr1.tea.texas.gov/perfreport/tapr/2018/download/campstaar2a.html |
| `staar_2019.html` | https://rptsvr1.tea.texas.gov/perfreport/tapr/2019/xplore/cstaar_eoc1.html |
| `staar_2021.html` | https://rptsvr1.tea.texas.gov/perfreport/tapr/2021/xplore/cstaar_eoc.html |
| `staar_2022.html` | https://rptsvr1.tea.texas.gov/perfreport/tapr/2022/xplore/cstaar_eoc.html |
| `staar_2023.html` | https://rptsvr1.tea.texas.gov/perfreport/tapr/2023/xplore/cstaar_eoc.html |
| `staff_2019.html` | https://rptsvr1.tea.texas.gov/perfreport/tapr/2019/xplore/cstaf.html |

The 2024 STAAR download includes its own descriptive header, matched by position
to the code header using a CSV parser (quoted multiline headings are preserved).
AP, IB and graduation definitions are in their original workbooks.

`CAMPSTAAR5.dat` was found to contain **additional student groups only**, with no
`CDA` (All Students) fields. The prior inventory's verification of its year did
not verify the required population. It remains unchanged and staged.

The correct primary-group file `13_staar_eoc/2018/CAMPSTAAR2.dat` was obtained by
POST to `https://rptsvr1.tea.texas.gov/cgi/sas/broker`, using the form at
https://rptsvr1.tea.texas.gov/perfreport/tapr/2018/download/DownloadData.html:

```
_service=marykay
prgopt=2018/tapr/tapr_download.sas
year4=2018
year2=18
topic=acct
_debug=0
title=Data Download
_program=perfrept.perfmast.sas
sumlev=C
setpick=STAAR2
```

SHA-256: `b7492711e19b890fc96a3492a28ae2f871d3cd3f64dce9df883f7e6416d676f0`.
It has 8,759 unique campus rows and 2,157 fields, including **published 2018
rates** for all five selected subjects. Use those rates rather than reconstructing
suppressed values from counts. The old additional-group file is not merged.

STAAR fields use `CDA00{AA1,AR1,AR2,ABI,AUS}12{yy}R`. `AA0` includes accelerated
testers and is excluded. The 2018 matching numerator suffix is `12{yy}N` and
denominator suffix `10{yy}D`. Only if a rate column is absent may an unsuppressed
numerator and positive unsuppressed denominator be used. A blank or suppressed
published rate is never reconstructed. See TEA's
[masking rules](https://rptsvr1.tea.texas.gov/perfreport/tapr/2018/masking.html).

The [STAAR redesign](https://tea.texas.gov/assessment-and-accountability/assessment-initiatives/staar-redesign)
took effect in spring 2023. Consistent field names do not remove this assessment
change; 2023/2024 comparisons need that qualification. There is no 2020 source.
