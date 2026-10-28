#!/bin/sh
# P3, editor re-review (19/10/2026): truths with a non-common strength, a thermal error growing with the heating time,
# and a Miner factor depending on the share of small cycles (static coverage as A).
#   sh chay_pb3b.sh [workers] >> chay_pb3b_log.txt 2>&1
cd "$(dirname "$0")"
W=${1:-6}
for t in Q_boundary_nc Q_evidence_nc Q_boundary_th Q_evidence_th Q_boundary_seq Q_evidence_seq; do
  python -I thu_nghiem_g6.py $t "$W" || echo "FAILED $t"
done
echo "second structural block done $(date '+%d/%m %H:%M')"
