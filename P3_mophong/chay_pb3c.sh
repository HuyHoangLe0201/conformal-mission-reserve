#!/bin/sh
# P3, editor re-review (19/10/2026): fleets of 20 legs sharing one bench (F), and the closed loop at n1 = 40 with the
# PAC certificate (C_*_n40).
#   sh chay_pb3c.sh [workers] >> chay_pb3c_log.txt 2>&1
cd "$(dirname "$0")"
W=${1:-6}
for t in F_boundary_cruise F_boundary_coast F_beyond_cruise C_boundary_cruise_n40 C_boundary_coast_n40; do
  python -I thu_nghiem_g6.py $t "$W" || echo "FAILED $t"
done
echo "fleet and PAC block done $(date '+%d/%m %H:%M')"
