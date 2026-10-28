#!/bin/sh
# Fourth round, second block (20/10/2026): electro-thermal truth. Waits for the first block (chay_vong4.sh), builds the
# electro-thermal cycle databases of both driving strategies, compares their true damage with the case-study response,
# and runs the static coverage with that truth.
# Run from P3_mophong: sh chay_vong4b.sh > chay_vong4b_log.txt 2>&1
set -e
until grep -q "fourth-round block done" chay_vong4_log.txt; do sleep 60; done
for st in cruise coast; do
  [ -f "chu_ky_${st}_d00_base_et.npz" ] || python -I dien_nhiet.py build $st 8
done
python -I so_sanh_et.py
for t in E_boundary_et_cruise E_boundary_et_coast E_evidence_et_cruise E_evidence_et_coast; do
  python -I thu_nghiem_g6.py $t 8
done
echo "electro-thermal block done $(date '+%d/%m %H:%M')"
