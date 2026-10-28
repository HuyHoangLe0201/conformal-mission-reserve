#!/bin/sh
# P3, third review (19/10/2026): repeated temporal check, union with a small mission level, truths outside the family
# of the stated margin, mission change within a life, and the closed loop at alpha = 0.05.
#   sh chay_pb3.sh [workers] >> chay_pb3_log.txt 2>&1
cd "$(dirname "$0")"
W=${1:-6}
for t in T_evidence_b U_boundary U_evidence Q_boundary_int Q_evidence_int Q_boundary_lutz \
         M_boundary_cruise_coast M_boundary_coast_cruise M_evidence_cruise_coast M_evidence_coast_cruise \
         C_boundary_cruise_a05 C_boundary_coast_a05; do
  python -I thu_nghiem_g6.py $t "$W" || echo "FAILED $t"
done
echo "third-review block done $(date '+%d/%m %H:%M')"
