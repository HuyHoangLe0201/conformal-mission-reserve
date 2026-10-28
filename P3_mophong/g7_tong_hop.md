
### L_evidence: module lives (100 repetitions, n1 20, n2 50, alpha 0.1)

| configuration | intervals, mean (median) | failures | collapses | true damage at replacement | coverage per decision | coverage per life | thermal bound covers outer / inner | bound width outer / inner, mean | largest | 'in' / 'out' share | pulses per interval | pulse share of margined / true damage |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| stated bound 1.10 | 44.9 (40) | 0.00 | 0.00 | 0.061 | 1.000 | 1.000 | 1.000 / 0.994 | 0.100 / 0.100 | 0.100 / 0.100 | 0.00 / 0.00 | 0 | 0.0 % / 0.0 % |
| stated bound 1.10, balancing | 59.2 (48) | 0.00 | 0.00 | 0.054 | 1.000 | 1.000 | 1.000 / 0.992 | 0.100 / 0.100 | 0.100 / 0.100 | 0.50 / 0.38 | 0 | 0.0 % / 0.0 % |
| operation only | 43.6 (39) | 0.00 | 0.00 | 0.059 | 1.000 | 1.000 | 0.987 / 0.985 | 0.100 / 0.100 | 0.101 / 0.101 | 0.00 / 0.00 | 0 | 0.0 % / 0.0 % |
| operation only, balancing | 57.7 (47) | 0.00 | 0.00 | 0.052 | 1.000 | 1.000 | 0.989 / 0.985 | 0.113 / 0.113 | 0.121 / 0.124 | 0.49 / 0.38 | 0 | 0.0 % / 0.0 % |
| pulses gated by reserve (10-30) | 59.0 (48) | 0.00 | 0.00 | 0.058 | 1.000 | 1.000 | 0.991 / 0.989 | 0.098 / 0.098 | 0.269 / 0.269 | 0.53 / 0.36 | 150 | 1.4 % / 7.0 % |
| gated (10-30) + dwell minimum | 59.3 (48) | 0.00 | 0.00 | 0.059 | 1.000 | 1.000 | 0.990 / 0.985 | 0.075 / 0.075 | 0.109 / 0.109 | 0.50 / 0.37 | 170 | 1.5 % / 7.7 % |
| pulses gated by plan (30-60) | 52.6 (40) | 0.00 | 0.08 | 0.049 | 1.000 | 1.000 | 0.994 / 0.994 | 0.286 / 0.286 | 1.204 / 1.204 | 0.52 / 0.37 | 75 | 0.7 % / 3.6 % |
| gated (30-60) + dwell minimum | 58.8 (47) | 0.00 | 0.00 | 0.056 | 1.000 | 1.000 | 0.989 / 0.985 | 0.094 / 0.094 | 0.112 / 0.112 | 0.50 / 0.37 | 107 | 1.0 % / 5.0 % |
| pulses at full rate | 59.5 (48) | 0.00 | 0.00 | 0.061 | 1.000 | 1.000 | 0.987 / 0.982 | 0.055 / 0.055 | 0.100 / 0.100 | 0.50 / 0.37 | 294 | 2.2 % / 12.3 % |

Paired differences in served intervals (mean; lives longer / shorter):

- operation only, balancing minus operation only: +14.0 (85 / 0)
- stated bound 1.10, balancing minus stated bound 1.10: +14.2 (85 / 0)
- operation only minus stated bound 1.10: -1.3 (14 / 44)
- gated (10-30) + dwell minimum minus pulses gated by reserve (10-30): +0.3 (29 / 5)
- gated (30-60) + dwell minimum minus pulses gated by plan (30-60): +6.2 (78 / 5)
- pulses at full rate minus pulses gated by reserve (10-30): +0.5 (48 / 6)
- pulses at full rate minus pulses gated by plan (30-60): +6.9 (78 / 7)
- pulses at full rate minus gated (10-30) + dwell minimum: +0.2 (24 / 1)
- pulses at full rate minus stated bound 1.10, balancing: +0.3 (32 / 18)

### L_evidence: phase map of the information of the inner module (median over 100 worlds)

| age | configuration | Phi(F) - F at F = 5 | fixed points of the median map (F) | bound width at the largest fixed point |
|---|---|---|---|---|
| start | pulses gated by reserve (10-30) | +203.6 | 1082 | 0.050 |
| start | gated (10-30) + dwell minimum | +203.6 | 1082 | 0.050 |
| start | pulses gated by plan (30-60) | -1.0 | 5, 620 | 0.066 |
| start | gated (30-60) + dwell minimum | +42.3 | 620 | 0.066 |
| start | operation only | +53.1 | 271 | 0.100 |
| start | operation only, balancing | +27.3 | 142 | 0.138 |
| mid | pulses gated by reserve (10-30) | +112.3 | 1066 | 0.050 |
| mid | gated (10-30) + dwell minimum | +112.3 | 1066 | 0.050 |
| mid | pulses gated by plan (30-60) | -1.0 | none | -- |
| mid | gated (30-60) + dwell minimum | +42.3 | 216 | 0.112 |
| mid | operation only | +53.1 | 271 | 0.100 |
| mid | operation only, balancing | +27.3 | 142 | 0.138 |

### L_boundary: module lives (100 repetitions, n1 20, n2 50, alpha 0.1)

| configuration | intervals, mean (median) | failures | collapses | true damage at replacement | coverage per decision | coverage per life | thermal bound covers outer / inner | bound width outer / inner, mean | largest | 'in' / 'out' share | pulses per interval | pulse share of margined / true damage |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| stated bound 1.10 | 42.7 (42) | 0.05 | 0.00 | 0.380 | 0.967 | 0.948 | 0.036 / 0.036 | 0.100 / 0.100 | 0.100 / 0.100 | 0.00 / 0.00 | 0 | 0.0 % / 0.0 % |
| stated bound 1.10, balancing | 55.6 (52) | 0.05 | 0.00 | 0.360 | 0.966 | 0.948 | 0.032 / 0.032 | 0.100 / 0.100 | 0.100 / 0.100 | 0.56 / 0.31 | 0 | 0.0 % / 0.0 % |
| operation only | 38.2 (38) | 0.03 | 0.00 | 0.338 | 0.980 | 0.969 | 0.978 / 0.976 | 0.100 / 0.100 | 0.101 / 0.101 | 0.00 / 0.00 | 0 | 0.0 % / 0.0 % |
| operation only, balancing | 50.8 (50) | 0.03 | 0.00 | 0.320 | 0.979 | 0.959 | 0.979 / 0.981 | 0.113 / 0.113 | 0.120 / 0.122 | 0.54 / 0.34 | 0 | 0.0 % / 0.0 % |
| pulses gated by reserve (10-30) | 51.3 (50) | 0.04 | 0.00 | 0.331 | 0.974 | 0.959 | 0.977 / 0.980 | 0.100 / 0.100 | 0.284 / 0.284 | 0.54 / 0.34 | 146 | 1.4 % / 1.9 % |
| gated (10-30) + dwell minimum | 51.6 (50) | 0.04 | 0.00 | 0.335 | 0.974 | 0.959 | 0.973 / 0.973 | 0.076 / 0.076 | 0.109 / 0.109 | 0.55 / 0.33 | 165 | 1.5 % / 2.2 % |
| pulses gated by plan (30-60) | 44.2 (40) | 0.02 | 0.14 | 0.280 | 0.975 | 0.959 | 0.987 / 0.991 | 0.309 / 0.309 | 1.311 / 1.311 | 0.49 / 0.40 | 63 | 0.6 % / 0.8 % |
| gated (30-60) + dwell minimum | 51.2 (50) | 0.04 | 0.00 | 0.328 | 0.974 | 0.959 | 0.979 / 0.976 | 0.097 / 0.097 | 0.112 / 0.112 | 0.54 / 0.34 | 96 | 0.9 % / 1.3 % |
| pulses at full rate | 51.7 (50) | 0.04 | 0.00 | 0.339 | 0.974 | 0.959 | 0.962 / 0.962 | 0.055 / 0.055 | 0.100 / 0.100 | 0.55 / 0.32 | 291 | 2.3 % / 3.7 % |

Paired differences in served intervals (mean; lives longer / shorter):

- operation only, balancing minus operation only: +12.6 (87 / 0)
- stated bound 1.10, balancing minus stated bound 1.10: +12.9 (86 / 0)
- operation only minus stated bound 1.10: -4.5 (3 / 72)
- gated (10-30) + dwell minimum minus pulses gated by reserve (10-30): +0.3 (33 / 2)
- gated (30-60) + dwell minimum minus pulses gated by plan (30-60): +7.0 (77 / 2)
- pulses at full rate minus pulses gated by reserve (10-30): +0.5 (44 / 7)
- pulses at full rate minus pulses gated by plan (30-60): +7.5 (77 / 7)
- pulses at full rate minus gated (10-30) + dwell minimum: +0.1 (16 / 5)
- pulses at full rate minus stated bound 1.10, balancing: -3.8 (4 / 60)

### L_boundary: phase map of the information of the inner module (median over 100 worlds)

| age | configuration | Phi(F) - F at F = 5 | fixed points of the median map (F) | bound width at the largest fixed point |
|---|---|---|---|---|
| start | pulses gated by reserve (10-30) | +202.1 | 1082 | 0.050 |
| start | gated (10-30) + dwell minimum | +202.1 | 1082 | 0.050 |
| start | pulses gated by plan (30-60) | -1.0 | 6, 722 | 0.061 |
| start | gated (30-60) + dwell minimum | +42.3 | 722 | 0.061 |
| start | operation only | +53.1 | 271 | 0.100 |
| start | operation only, balancing | +53.1 | 142 | 0.138 |
| mid | pulses gated by reserve (10-30) | +104.4 | 1082 | 0.050 |
| mid | gated (10-30) + dwell minimum | +104.4 | 1082 | 0.050 |
| mid | pulses gated by plan (30-60) | -1.0 | none | -- |
| mid | gated (30-60) + dwell minimum | +42.3 | 216 | 0.112 |
| mid | operation only | +53.1 | 271 | 0.100 |
| mid | operation only, balancing | +27.3 | 142 | 0.138 |

### Forgetting factor (evidence truth, the same 30 worlds; rates recalibrated to the same bound widths)

| lambda | configuration | intervals, mean | largest bound width, median | phase map at mid-life: fixed points of the inner module |
|---|---|---|---|---|
| 0.6 | pulses gated by reserve (10-30) | 47.6 | 0.938 | 981 |
| 0.6 | pulses gated by plan (30-60) | 32.7 | 1.660 | none (collapse) |
| 0.6 | gated (30-60) + dwell minimum | 48.4 | 0.112 | 216 |
| 0.6 | pulses at full rate | 49.1 | 0.100 | -- |
| 0.8 | pulses gated by reserve (10-30) | 48.7 | 0.277 | 981 |
| 0.8 | pulses gated by plan (30-60) | 43.3 | 1.195 | none (collapse) |
| 0.8 | gated (30-60) + dwell minimum | 48.4 | 0.112 | 216 |
| 0.8 | pulses at full rate | 49.1 | 0.100 | -- |
| 0.9 | pulses gated by reserve (10-30) | 48.8 | 0.128 | 981 |
| 0.9 | pulses gated by plan (30-60) | 47.4 | 0.389 | none (collapse) |
| 0.9 | gated (30-60) + dwell minimum | 48.5 | 0.110 | 216 |
| 0.9 | pulses at full rate | 49.1 | 0.100 | -- |

### Sensitivity: z of the thermal bound (the same 30 worlds)

| value | configuration | intervals, mean | failures | coverage per life | thermal bound covers outer / inner | bound width inner, mean / largest (median) | pulse share of margined damage |
|---|---|---|---|---|---|---|---|
| 1.28 | operation only | 38.3 | 0.00 | 1.000 | 0.941 / 0.948 | 0.078 / 0.079 | 0.0 % |
| 1.28 | pulses gated by plan (30-60) | 44.5 | 0.00 | 1.000 | 0.978 / 0.983 | 0.258 / 1.066 | 0.5 % |
| 1.28 | gated (30-60) + dwell minimum | 48.8 | 0.00 | 1.000 | 0.947 / 0.957 | 0.076 / 0.087 | 0.7 % |
| 1.28 | pulses at full rate | 49.4 | 0.00 | 1.000 | 0.939 / 0.944 | 0.043 / 0.078 | 1.9 % |
| 1.645 | operation only | 37.7 | 0.00 | 1.000 | 0.980 / 0.977 | 0.100 / 0.101 | 0.0 % |
| 1.645 | pulses gated by plan (30-60) | 43.3 | 0.00 | 1.000 | 0.993 / 0.994 | 0.312 / 1.195 | 0.5 % |
| 1.645 | gated (30-60) + dwell minimum | 48.4 | 0.00 | 1.000 | 0.985 / 0.981 | 0.097 / 0.112 | 0.7 % |
| 1.645 | pulses at full rate | 49.1 | 0.00 | 1.000 | 0.979 / 0.978 | 0.055 / 0.100 | 1.9 % |
| 2.33 | operation only | 36.7 | 0.00 | 1.000 | 0.997 / 1.000 | 0.141 / 0.143 | 0.0 % |
| 2.33 | pulses gated by plan (30-60) | 41.0 | 0.00 | 1.000 | 0.998 / 0.998 | 0.401 / 1.518 | 0.5 % |
| 2.33 | gated (30-60) + dwell minimum | 47.6 | 0.00 | 1.000 | 0.998 / 0.999 | 0.138 / 0.158 | 0.8 % |
| 2.33 | pulses at full rate | 48.4 | 0.00 | 1.000 | 0.998 / 0.999 | 0.078 / 0.141 | 2.0 % |

### Sensitivity: drift of the thermal path per unit damage (boundary) (the same 30 worlds)

| value | configuration | intervals, mean | failures | coverage per life | thermal bound covers outer / inner | bound width inner, mean / largest (median) | pulse share of margined damage |
|---|---|---|---|---|---|---|---|
| 0 | stated bound 1.10 | 39.3 | 0.07 | 0.967 | 1.000 / 1.000 | 0.100 / 0.100 | 0.0 % |
| 0 | operation only | 36.7 | 0.03 | 0.967 | 0.979 / 0.995 | 0.100 / 0.101 | 0.0 % |
| 0 | gated (30-60) + dwell minimum | 47.7 | 0.03 | 0.967 | 0.974 / 0.990 | 0.099 / 0.112 | 0.8 % |
| 0 | pulses at full rate | 48.3 | 0.07 | 0.967 | 0.968 / 0.990 | 0.055 / 0.100 | 2.1 % |
| 0.2 | stated bound 1.10 | 38.4 | 0.07 | 0.967 | 0.038 / 0.038 | 0.100 / 0.100 | 0.0 % |
| 0.2 | operation only | 35.2 | 0.03 | 0.967 | 0.974 / 0.991 | 0.100 / 0.101 | 0.0 % |
| 0.2 | gated (30-60) + dwell minimum | 45.9 | 0.07 | 0.967 | 0.968 / 0.992 | 0.099 / 0.112 | 0.8 % |
| 0.2 | pulses at full rate | 46.4 | 0.07 | 0.967 | 0.953 / 0.986 | 0.055 / 0.100 | 2.2 % |
| 0.4 | stated bound 1.10 | 37.8 | 0.07 | 0.933 | 0.038 / 0.038 | 0.100 / 0.100 | 0.0 % |
| 0.4 | operation only | 34.0 | 0.07 | 0.967 | 0.956 / 0.986 | 0.100 / 0.101 | 0.0 % |
| 0.4 | gated (30-60) + dwell minimum | 44.4 | 0.07 | 0.967 | 0.956 / 0.987 | 0.100 / 0.112 | 0.8 % |
| 0.4 | pulses at full rate | 44.7 | 0.07 | 0.967 | 0.939 / 0.968 | 0.055 / 0.100 | 2.2 % |

### Sensitivity: minimum dwell pulses per interval (the same 30 worlds)

| value | configuration | intervals, mean | failures | coverage per life | thermal bound covers outer / inner | bound width inner, mean / largest (median) | pulse share of margined damage |
|---|---|---|---|---|---|---|---|
| 20 | gated (10-30) + dwell minimum | 48.8 | 0.00 | 1.000 | 0.987 / 0.985 | 0.089 / 0.168 | 1.2 % |
| 20 | pulses gated by plan (30-60) | 43.3 | 0.00 | 1.000 | 0.993 / 0.994 | 0.312 / 1.195 | 0.5 % |
| 20 | gated (30-60) + dwell minimum | 47.8 | 0.00 | 1.000 | 0.988 / 0.986 | 0.136 / 0.193 | 0.6 % |
| 60 | gated (10-30) + dwell minimum | 48.8 | 0.00 | 1.000 | 0.986 / 0.981 | 0.077 / 0.109 | 1.2 % |
| 60 | pulses gated by plan (30-60) | 43.3 | 0.00 | 1.000 | 0.993 / 0.994 | 0.312 / 1.195 | 0.5 % |
| 60 | gated (30-60) + dwell minimum | 48.4 | 0.00 | 1.000 | 0.985 / 0.981 | 0.097 / 0.112 | 0.7 % |
| 120 | gated (10-30) + dwell minimum | 49.0 | 0.00 | 1.000 | 0.984 / 0.981 | 0.068 / 0.100 | 1.4 % |
| 120 | pulses gated by plan (30-60) | 43.3 | 0.00 | 1.000 | 0.993 / 0.994 | 0.312 / 1.195 | 0.5 % |
| 120 | gated (30-60) + dwell minimum | 48.7 | 0.00 | 1.000 | 0.983 / 0.980 | 0.077 / 0.100 | 1.0 % |
