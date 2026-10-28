#!/bin/sh
# P3, editor re-review (19/10/2026): baseplate-solder fatigue (Supplementary Table S19). Case-temperature cycles of the
# certified modules on the vehicle days of both strategies, solder damage against the junction damage (han_de.py), and
# the static coverage with a solder channel added to the truth (S_*: as A_*_k0, cruising with its seeds, and coasting).
#   sh chay_han.sh [workers] >> chay_han_log.txt 2>&1
cd "$(dirname "$0")"
W=${1:-6}
for st in cruise coast; do
  [ -f "chu_ky_vo_$st.npz" ] || python -I chu_ky_vo.py hsr_runs_jinghu.csv $st "$W" || echo "FAILED chu_ky_vo $st"
done
python -I han_de.py > /dev/null || echo "FAILED han_de"
for t in S_boundary_cruise S_evidence_cruise S_boundary_coast S_evidence_coast; do
  python -I thu_nghiem_g6.py $t "$W" || echo "FAILED $t"
done
echo "solder block done $(date '+%d/%m %H:%M')"
