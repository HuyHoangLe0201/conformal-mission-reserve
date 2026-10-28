#!/bin/sh
# Fourth round (20/10/2026): experiments that test the remaining weaknesses of P3 directly.
#   P_*_weak     weak subpopulation (strength law outside the Weibull family): static coverage
#   P_*_a001     alpha = 0.01 with benches of 200 and 400 modules
#   M_*_dl       mission change with the replanned ages computed only after a detection delay (2, 6, 12 intervals)
#   M_*_weak     the same under the weak subpopulation
#   C_*_weak     closed loop, replacement only, conformal against parametric certificates, weak subpopulation
#   V_*          value of anchor tests in the closed loop (ideal and five realistic anchor tests at 30 and 300 s)
# Run from P3_mophong: sh chay_vong4.sh > chay_vong4_log.txt 2>&1
set -e
for t in M_boundary_cruise_coast_dl M_boundary_cruise_coast_weak; do python -I thu_nghiem_g6.py $t 10; done
for t in P_boundary_weak P_evidence_weak P_boundary_a001 P_boundary_weak_a001; do python -I thu_nghiem_g6.py $t 10; done
for t in C_boundary_cruise_weak C_boundary_coast_weak; do python -I thu_nghiem_g6.py $t 6; done
for sc in evidence boundary; do
  for st in cruise coast; do
    for an in ideal real; do python -I thu_nghiem_g6.py V_${sc}_${st}_${an} 6; done
  done
done
echo "fourth-round block done $(date '+%d/%m %H:%M')"
