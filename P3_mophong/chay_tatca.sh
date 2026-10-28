#!/bin/sh
# P3: the whole chain of the case study in order, from the timetable data to the final figures and tables (3.3 kV
# converter and per-module certificates since 18/10/2026; the 1.7 kV chain is kept in luu_1700V_20261018/). It replaces
# the staged scripts chay_csdl*.sh, chay_g6*.sh, chay_g7*.sh, chay_mo.sh and chay_hv_*.sh, which ran the same commands
# on a shared machine (waits for free memory, reruns after a stall). Every step reads and writes in this folder; the
# experiment drivers are seeded per repetition and resumable (finished repetitions are skipped), and the number of
# worker processes (last argument) only sets the parallelism. With the stored g6_*.jsonl and g7_*.jsonl in place, steps
# 4 and 5 therefore add nothing: move these files away to compute the runs again.
#
#   sh chay_tatca.sh "<high-speed trains operation data.csv>" "<adjacent railway stations mileage data.csv>"
#
# The two files are those of Zhang et al. (2022), figshare doi:10.6084/m9.figshare.15087882.v4 (CC0). Without arguments
# the mission files hsr_*.csv of this folder are kept and the chain starts at step 2.
# Time on the desktop used for the paper: 12-25 min per cycle database with 8 processes (14 databases), and about a day
# for the experiments, most of it in the closed loop (12 configurations of 300 lives). The database builds and the
# closed-loop runs (cycle pools of six derating levels per process, about 1.6 GB) need several GB of memory; lower the
# number of processes if memory is short.
set -e
cd "$(dirname "$0")"

# 1. Missions (Section 6.3): cleaned stop-to-stop runs, modelled runs and vehicle days, on-time share under derating
if [ $# -eq 2 ]; then
  python -I du_lieu_hsr.py build "$1" "$2" "Beijingnan Railway Station" "Shanghaihongqiao Railway Station" jinghu
  python -I ho_so.py hsr_runs_jinghu.csv jinghu
  python -I giam_tai_kha_thi.py hsr_runs_model_jinghu.csv
fi

# 2. Losses of the NPC leg of 3.3 kV single-switch modules and cycle databases (derating 0-0.5 under both driving
#    strategies; balancing patterns). Existing files are kept (for instance the databases of a release asset).
[ -f bang_ton_hao.npz ] || python -I ton_hao.py > ton_hao_log.txt
for cfg in "cruise 0.0 base 00" "coast 0.0 base 00" "cruise 0.0 in 00" "cruise 0.0 out 00" \
           "cruise 0.1 base 10" "cruise 0.2 base 20" "cruise 0.3 base 30" "cruise 0.4 base 40" "cruise 0.5 base 50" \
           "coast 0.1 base 10" "coast 0.2 base 20" "coast 0.3 base 30" "coast 0.4 base 40" "coast 0.5 base 50"; do
  set -- $cfg
  [ -f "chu_ky_$1_d$4_$3.npz" ] || python -I nhiem_vu.py hsr_runs_jinghu.csv "$1" "$2" "$3" all 8
done

# 3. Budget of the margined damage per chip (paper: Table 5b, Section 6.3) under the margin rules, both driving
#    strategies and both balancing patterns; collapse by caution E3 on both driving strategies (Fig. 7, with the
#    bandwidth sensitivity of its test weight on the cruising mission: Supplementary Table S2)
python -I g5_ngan_sach.py chu_ky_cruise_d00_base.npz
python -I g5_ngan_sach.py chu_ky_cruise_d00_base.npz g5_ngan_sach_chienluoc_cruise.json base,bare
python -I g5_ngan_sach.py chu_ky_coast_d00_base.npz g5_ngan_sach_chienluoc_coast.json base,bare
python -I g5_ngan_sach.py chu_ky_cruise_d00_in.npz g5_ngan_sach_mau_in.json base,bare
python -I g5_ngan_sach.py chu_ky_cruise_d00_out.npz g5_ngan_sach_mau_out.json base,bare
python -I thu_nghiem_g5.py e3 200
python -I thu_nghiem_g5.py e3 200 coast
python -I e3_bang_thong.py

# 4. Case study (thu_nghiem_g6.py; paper: Sections 6.5-6.9): A coverage and tightness (four truths), D margin
#    sensitivity, E season shift, N anchor tests, X exceedance of the heating-time margin, T temporal structure, L lot
#    shift, Y stated components exceeded at once, R re-simulation necessity, C closed loop over leg lives (every policy,
#    method and baseline per truth and driving strategy, and the drifting thermal scale)
for t in A_boundary_k0 A_evidence_k0 A_adverse_strict_k0 A_beyond_k0 A_evidence_k20 A_evidence_n500 D_cov E_shift \
         E_shift_evidence N_evidence N_boundary N_adverse_strict X_ton T_boundary T_evidence L_boundary L_evidence \
         Y_stack R_boundary R_evidence; do
  python -I thu_nghiem_g6.py $t 3
done
python -I thu_nghiem_g6.py D_det
for t in C_boundary_cruise C_boundary_coast C_evidence_cruise C_evidence_coast C_adverse_strict_cruise \
         C_adverse_strict_coast C_beyond_cruise C_beyond_coast C_drift C_drift_allow C_drift04 C_drift04_allow; do
  python -I thu_nghiem_g6.py $t 3
done
# the worked example and the certificates on the real bench data (Section 6.1)
python -I vi_du_so.py > vi_du_so_log.txt
# further tests (Supplementary Section S13): T_evidence repeated, union with a small mission level, truths outside
# the family of the stated margin, mission change within a life, closed loop at alpha = 0.05
sh chay_pb3.sh 3
sh chay_pb3b.sh 3
sh chay_pb3c.sh 3
# baseplate-solder fatigue, a second mechanism outside the certificate (Table S19)
sh chay_han.sh 3
# multi-split paired certificate: 15 splits of the same bench (Table S20), and on the real bench data
for t in K_boundary K_evidence; do
  python -I thu_nghiem_g6.py $t 3
done
python -I chia_nhieu.py
# coverage of the statistical part of the margin alone (Table S21)
python -I thu_nghiem_g6.py W_boundary 3
# out-of-sample check of the heating-time rule on real lifetime data (Table S22)
(cd ../P3_kiemtoan && python g27_ton_ngoai_mau.py > /dev/null)
# heating time timed from the valley or from three quarters of the swing (600 vehicle days, Section S14)
for lv in 0.0 0.75; do for st in cruise coast; do
  P3_TON_LEVEL=$lv python -I nhiem_vu.py hsr_runs_jinghu.csv $st 0 base 600 8
done; done
python -I dinh_nghia_ton.py

# 5. Information loop of the thermal bound (thu_nghiem_g7.py; paper: Section 6.8, Fig. 8)
for t in L_evidence L_boundary S_lam06 S_lam09 S_z128 S_z233 S_rho0 S_rho04 S_nmin20 S_nmin120; do
  python -I thu_nghiem_g7.py $t 5
done

# 6. Summaries, final figures and tables
python -I g6_tong_hop.py
python -I g6_vuot_nhieu.py
python -I g7_tong_hop.py
python -I hinh_cuoi.py
python -I hinh_ga.py
python -I bang_cuoi.py
python -I tong_hop_pb3.py
python -I chan_doan_am.py
python -I phan_ra_hu_hai.py
python -I hinh_quy_dao.py
python -I bang_pb3.py
python -I ../P3_kiemtoan/hinh_cuoi_f3.py

# 7. Fourth round (20/10/2026, supplementary Section S15): weak subpopulation, alpha = 0.01 with large benches,
# detection delay, value of anchor tests, weighted certificate with anchors at full load, electro-thermal truth
for t in M_boundary_cruise_coast_dl M_boundary_cruise_coast_weak P_boundary_weak P_evidence_weak P_boundary_a001 \
         P_boundary_weak_a001; do python -I thu_nghiem_g6.py $t 8; done
for t in C_boundary_cruise_weak C_boundary_coast_weak; do python -I thu_nghiem_g6.py $t 6; done
for sc in evidence boundary; do for st in cruise coast; do for an in ideal real; do
  python -I thu_nghiem_g6.py V_${sc}_${st}_${an} 6
done; done; done
python -I e3_thiet_ke.py
python -I e3_neo_co_dinh.py
python -I dien_nhiet.py tables
for st in cruise coast; do python -I dien_nhiet.py build $st 8; done
python -I so_sanh_et.py
for t in E_boundary_et_cruise E_boundary_et_coast E_evidence_et_cruise E_evidence_et_coast; do
  python -I thu_nghiem_g6.py $t 8
done
python -I et_chung_chi.py
python -I vong4_tong_hop.py
python -I bang_vong4.py
