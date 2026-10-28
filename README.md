# Reproduction code: conformal mission reserve certificates for power modules

Code, digitized data and result files for the manuscript

> *Conformal mission reserve certificates for health-aware control and replacement of power modules*, prepared for Reliability Engineering & System Safety.

The paper certifies the damage that a power module will have reached at the end of the coming maintenance interval,
with finite-sample coverage, when a health-aware control policy acts on the certificate. The case study combines
power-cycling data digitized from published tests, the operation records of the Beijing–Shanghai high-speed line
(Zhang et al., 2022), a train and converter model (three-level NPC leg of 3.3 kV/1 kA single-switch silicon IGBT
modules, with a certificate per module) and simulated truths against which the coverage is measured. All results are obtained by simulation; the absolute lives illustrate the
method only.

## Library: `p3cert`

A small, dependency-light (NumPy only) implementation of the certificates and the decision rule, for use on your own
bench and mission data. It does not contain the railway case study; that is in the folders below.

```bash
pip install -e .
python -m unittest discover -s tests -v      # 12 tests, including the worked example of Section 7.1
python examples/quickstart.py                # synthetic bench -> certificate -> derating or replacement decision
python examples/holdout_real_modules.py      # Table S31: disjoint hold-out on the digitized Hernes et al. modules
```

```python
from p3cert import fit_lifetime, log_strength, paired_certificate, choose_level

model = fit_lifetime(log_life_fit, dT_fit, t_on_fit, I_fit)            # lifetime model on the fit part of the bench
S = log_strength(model, N_obs_score, dT_score, t_on_score, I_score)    # log-strengths of the other part
cert = paired_certificate(S, V, D0=0.0, alpha=0.10)                    # V: margined damage of recorded intervals,
cert.value, cert.reserve, cert.passes                                  # re-simulated under the policy
```

| Module | Content |
|---|---|
| `p3cert.conformal` | Conformal ranks (upper, lower, minimum number of scores, finite-sample coverage interval) |
| `p3cert.certificate` | Paired and union certificates and the certified reserve (Eqs. 1 and 2) |
| `p3cert.lifetime` | Log-linear lifetime model, log-strengths, margined cycle life, Miner sum |
| `p3cert.policy` | Reserve rule of Algorithm 1: derating level or replacement from the certificates |
| `p3cert.holdout` | Disjoint hold-out of the calibrated part on your own bench modules (Section 7.1, Table S31) |

What you supply, and what the library does not do: the rainflow counting of your junction-temperature history, the
thermal model, the stated extrapolation margin and Miner factor, and the re-simulation of recorded intervals under each
derating level. The coverage guarantee holds for the calibrated part when the bench scores and the field device are
exchangeable and the stated components hold (Assumptions A1-A5 of the manuscript); see Section 8 of the manuscript for
what the certificate does and does not certify. The statistics of Section 7.1 show that the calibrated part is valid out
of sample inside the tested conditions and loses coverage when a whole condition is withheld.

## Contents

| Path | Content |
|---|---|
| `P3_kiemtoan/G2_*.csv`, `G3_*.csv`, `G10_*.csv` | Power-cycling data digitized from the publications listed under *Data sources* |
| `P3_kiemtoan/thongke_G2.py`, `g3_tref_bien.py`, `g10_tong_hop.py` and their `.json` | Statistics on the digitized data: lifetime model and reference set, tests of the scatter assumption, evidence for the stated margin, Miner factor (Sections 6.1–6.2) |
| `P3_kiemtoan/hinh_cuoi_f3.py` | Figure 2 (F3) |
| `P3_kiemtoan/g31_giu_lai_that.py`, `g31_giu_lai_that.json` | Disjoint hold-out of the calibrated part on the real modules (Section 7.1, Supplementary Table S31); NumPy only |
| `P3_kiemtoan/kiem_ghepcap.py`, `results.json` | Numerical check of the coverage results on an illustrative model (Supplementary Material, Section S1, Table S1) |
| `P3_kiemtoan/sohoa_*.py` | Digitization scripts; they document the method and need the source figures, which are not redistributed |
| `P3_mophong/*.py` | Train, converter, thermal and lifetime models, certificates, experiment drivers, summaries, final figures and tables (flat layout; every script reads and writes in its own folder) |
| `P3_mophong/chay_tatca.sh` | The whole chain in order, from the operation records to the figures and tables |
| `P3_mophong/hsr_*.csv`, `ho_so_jinghu.json`, `giam_tai_kha_thi.json` | Missions: cleaned stop-to-stop runs, modeled runs under both driving strategies, vehicle days, on-time share under derating |
| `P3_mophong/g5_ngan_sach*.json`, `e3_cruise_base_delta.json`, `e3_coast_base_delta.json` | Budget of the margined damage and experiment E3 (collapse by caution) on both driving strategies |
| `P3_mophong/g6_*.jsonl`, `g6_D_det.json` | Case-study runs, one line per repetition, and the deterministic margin sensitivity; `g6_C_*.jsonl` is the closed loop (every policy, certificate and maintenance baseline per truth and driving strategy, 300 lives, and the drift of the thermal path), `g6_R_*.jsonl` the test of re-simulation, `g6_T_*.jsonl` the temporal structure of the missions, `g6_L_*.jsonl` the lot shift and `g6_Y_stack.jsonl` the stated components exceeded at once |
| `P3_mophong/vi_du_so.py`, `vi_du_so.json` | Worked example of Section 6.1: the paired certificate and its reserve on the real bench data, step by step |
| `P3_mophong/g6_vuot_nhieu.py`, `g6_vuot_nhieu.json` | Results of the adverse and beyond truths by the number of exceeded stated components (Supplementary Material, Section S7) |
| `P3_mophong/e3_bang_thong.py`, `e3_bang_thong.json` | Test weight of the weighted certificate for other kernel bandwidths (Supplementary Table S2) |
| `P3_mophong/g7_*.jsonl` | Information-loop runs, one line per repetition |
| `P3_mophong/g6_tong_hop.json`, `.md`, `g7_tong_hop.json`, `.md` | Summaries read by the final figures and tables |
| `P3_hinh_bang/F1.pdf` … `F9.pdf` (and `.png`), `T1.tex`, `T3.tex`, `T5.tex`, `TS2.tex` … `TS10.tex` | Final figures (vector PDF at print size, 300 dpi PNG) and tables (LaTeX, booktabs; `TS`: Supplementary Material) |
| `P3_hinh_bang/GA.pdf` (and `.png`) | Graphical abstract, drawn by `P3_mophong/hinh_ga.py` from the data of Figs. 4 and 7b |
| `P3_hinh_bang/F1_cau_truc.tex`, `P3_hinh_bang_cuoi.tex` | Source of F1 (TikZ); overview of all figures and tables with their captions |
| `tailieu.bib` | References cited in the captions and tables, read by the overview |
| Release asset `P3_code_cycle_databases.zip` | Cycle databases `chu_ky_*.npz` (14 files, about 0.9 GB) and the loss tables `bang_ton_hao.npz`, needed to re-run the experiments; they can also be rebuilt (see below) |

The folder names are those of the project (*mô phỏng*: simulation; *kiểm toán*: audit; *hình, bảng*: figures and
tables), and the scripts refer to one another through them. File names and some identifiers are Vietnamese; docstrings,
comments and outputs are in English.

### Numbering

| Manuscript | Repository |
|---|---|
| Figures 1–9 | F1, F3, F2, F4, F5, F9, F6, F7, F8 |
| Tables 2, 3, 4 | T1, T5, T3 |
| Supplementary Tables S1–S11 | S1: `P3_kiemtoan/kiem_ghepcap.py` (`results.json`); S2: `P3_mophong/e3_bang_thong.py`; S3: experiment T; S4: experiments D, E and N; S5: experiment C at the margins; S6: experiments Y, L and A (adverse and beyond truths); S7: experiment C, every truth and the drift runs; S8: data sets; S9: `P3_mophong/thu_nghiem_g7.py`; S10: experiment R; tables by `bang_cuoi.py` (`TS2.tex` … `TS10.tex`); S11: `Scenario` in `P3_mophong/tuoi_tho.py` |

Comments in the code use the numbering of the project outline. The manuscript states one theorem and one proposition
and gives the other results in the text, with all proofs in Section S4 of the Supplementary Material:
- Assumptions P3.1–P3.5 are Assumptions A1–A5.
- Definition P3.6 is Section 2.4 (Eqs. 1–2), and Remark P3.7 is Section 2.5.
- Theorem P3.10 and Proposition P3.11 are Theorem 1 (a) and (b).
- Propositions P3.13–P3.14 are Section 3.2 (Eq. 5), and Remarks P3.15–P3.16 are Section 3.3.
- Proposition P3.17 is Proposition 1, Remark P3.18 is Section 4.1, and Proposition P3.19 is Section 4.2.
- Proposition P3.20 and Remark P3.21 are Section 4.3 (Eq. 6).
- Proposition P3.22 and Corollary P3.23 are Section 5.1 (Eqs. 7–8).
- Propositions P3.24–P3.26 and Remarks P3.25 and P3.27 are Sections 5.2 and 5.3 (Eqs. 9–11).

"Section 10 of the outline" is Section S12 of the Supplementary Material. G2–G11 are work packages of the project.

The code labels the certificates U1–U10; the manuscript names them without numbers, since $U_i$ denotes the conformal
scores there: U1 box, U2 Monte Carlo, U3 Bayes, U4 split, U5 union, U6 paired, U7 weighted, U8 adaptive, U9 PAC and U10
oracle (`reuse` and `anchored` are the variants of U6 with re-used calibration data and with ideal anchor tests; `mc`,
`bayes`, `tbr60` and `rtf` in the closed loop are U2, U3, replacement at the planned 60 intervals and run to failure;
`fix20_tbr60` adds a fixed 20 % derating to the planned replacement, `cert_age` replaces at the reserve of the paired
certificate of a new leg on all bench scores, `mc_age` at the B10 life of the Monte Carlo model and `mc_opt10`,
`mc_opt50` at the age that minimizes the cost of age replacement for r = 10 and 50).
Truth scenarios: `evidence` is the evidence truth, `boundary` the truth at the margins, `adverse_strict` the adverse
truth and `beyond` the beyond truth; `adverse` in `tuoi_tho.py`, which keeps the benefit of minimum temperatures below
60 °C, is not used in the manuscript. Experiment `X` is the exceedance of the heating-time margin,
`Y` the stated components exceeded at once and `L` the lot shift (not the `L` runs of `thu_nghiem_g7.py`).

## Data sources

* Operation records: Zhang, D. et al. (2022), A high-speed railway network dataset from train operation records and
  weather data, *Scientific Data* 9, 244, doi:10.1038/s41597-022-01349-8; data on figshare,
  doi:10.6084/m9.figshare.15087882.v4 (CC0). Two files are used, `high-speed trains operation data.csv` (MD5
  f8f5ee9dab655ca59fcc6ab9756b2cd1) and `adjacent railway stations mileage data.csv` (MD5
  c098b0e138dd6cc3fc80b575191e3695). They are not redistributed here; `hsr_runs_jinghu.csv` holds the runs of the
  Beijing–Shanghai line cleaned from them by `du_lieu_hsr.py`, with the rules stated in its docstring.
* Power-cycling data, digitized from figures or transcribed from tables, and checked against the values printed in each
  paper:
  * `G2_hernes2021_sohoa.csv`: cycles to failure of 34 of 36 modules, five constant-stress runs; Hernes, M. et al.
    (2021), *IET Power Electronics* 14(7), 1271–1283, doi:10.1049/pel2.12083;
  * `G2_hernes2021pcim_sohoa.csv`: Miner damage at failure of 48 of 49 modules, five constant and two mixed-stress runs;
    Hernes, M. et al. (2021), Proc. PCIM Europe digital days, 1202–1207, Fig. 4;
  * `G2_choi2017_sohoa.csv`: cycles to failure of 39 modules, six swing durations; Choi, U.-M. et al. (2017), *IEEE
    Transactions on Power Electronics* 32(8), 6434–6443, doi:10.1109/TPEL.2016.2618917, Fig. 7;
  * `G2_choi2018_bang2.csv`: cycles to failure of 18 modules, three swings; Choi, U.-M. et al. (2018),
    *Microelectronics Reliability* 88–90, 788–794, doi:10.1016/j.microrel.2018.07.002, Tables 1–2;
  * `G10_junghaenel2017_bang1.csv`: lives against load-pulse duration; Junghaenel, M., Scheuermann, U. (2017),
    *Microelectronics Reliability* 76–77, 480–484, doi:10.1016/j.microrel.2017.06.081, Table 1;
  * `G10_zeng2019_bang.csv`: lives under constant and block-mixed tests; Zeng, G. et al. (2019), *IEEE Transactions on
    Power Electronics* 34(5), 4722–4728, doi:10.1109/TPEL.2018.2859479, Tables II–VIII;
  * `G3_scheuermann2018_ton.csv`: the pulse-duration data as re-plotted in Scheuermann, U., Junghaenel, M. (2018), Proc.
    CIPS, 78–87, Figs. 6–7 (comparison only).

  The source figures and papers are not redistributed. Please cite the original publications when using these data.
* Device characteristics: digitized from the datasheet of the Mitsubishi Electric CM1000HC-66R module (3.3 kV, 1 kA,
  single switch) and contained in `P3_mophong/linh_kien_hv.py`; the datasheet is not redistributed.

## Requirements

Python 3.12 (tested with 3.12.6) with the packages in `requirements.txt`; PyMuPDF is needed only by the digitization
scripts. F1 and the overview need XeLaTeX with TikZ, the TeX Gyre Termes fonts, biblatex and biber (tested with TeX
Live 2026).

```bash
pip install -r requirements.txt
```

## Reproducing the figures and tables from the stored results (minutes)

In `P3_mophong/`:

```bash
python hinh_cuoi.py
```

```bash
python hinh_ga.py
```

```bash
python bang_cuoi.py
```

In `P3_kiemtoan/`:

```bash
python hinh_cuoi_f3.py
```

In `P3_hinh_bang/`:

```bash
xelatex F1_cau_truc.tex
```

`hinh_cuoi.py` writes F2 and F4–F9, `hinh_ga.py` the graphical abstract, `hinh_cuoi_f3.py` F3, and `bang_cuoi.py` T1, T3, T5, TS2–TS10 and the overview
`P3_hinh_bang_cuoi.tex`, all into `P3_hinh_bang/`; `F1_cau_truc.pdf` is F1. The overview compiles in `P3_hinh_bang/`
with `xelatex`, `biber`, `xelatex`, `xelatex`.

## Regenerating the summaries and the statistics (minutes)

In `P3_mophong/`, from the per-repetition results: `python g6_tong_hop.py` and `python g7_tong_hop.py` write
`g6_tong_hop.json`, `g7_tong_hop.json` and their markdown tables (and draft figures `*_nhap.pdf`).

In `P3_kiemtoan/`, from the digitized data: `python kiem_ghepcap.py`, `python thongke_G2.py` (about 90 s, parametric
bootstrap), `python g3_tref_bien.py` and `python g10_tong_hop.py` write `results.json`, `thongke_G2.json`,
`g3_tref_bien.json` and `g10_tong_hop.json`.

## Re-running the simulations (hours)

Download the two files of Zhang et al. (2022) from figshare and run, from the repository root:

```bash
sh P3_mophong/chay_tatca.sh "<folder>/high-speed trains operation data.csv" "<folder>/adjacent railway stations mileage data.csv"
```

The script runs, in order:

1. `du_lieu_hsr.py`, `ho_so.py`, `giam_tai_kha_thi.py`: missions (Section 6.3); about 3 min;
2. `ton_hao.py` and `nhiem_vu.py`: losses of the NPC leg and the 14 cycle databases (derating 0–0.5 under both driving
   strategies, and the two balancing patterns), about 15 min each with 10 processes;
3. `g5_ngan_sach.py` and `thu_nghiem_g5.py`: budget of the margined damage (Table 3b, Section 6.2) and experiment E3
   on both driving strategies (Fig. 8);
4. `thu_nghiem_g6.py` and `vi_du_so.py`: the case study (Sections 6.1 and 7.1–7.5; Table 4; Figs. 3–7 and 10;
   Supplementary Tables S3–S7 and S10–S22);
5. `thu_nghiem_g7.py`: the information loop of the thermal bound (Section 7.4, Fig. 9);
6. the summaries, figures and tables as above;
7. the fourth-round experiments of Supplementary Section S15 (Sections 7.2–7.6; Tables S23–S27): weak subpopulation,
   large benches at alpha = 0.01, late detection of a mission change, value of anchor tests, weighted certificate with
   anchors at full load, and the electro-thermal truth (two further cycle databases, about 45 min each with 8
   processes).

Without arguments the script keeps the mission files of the repository and starts at step 2. Cycle databases that
already exist are kept, so with the release asset extracted into `P3_mophong/` step 2 is skipped. The drivers of steps 4
and 5 are seeded per repetition and resumable: they skip the repetitions already written to `g6_*.jsonl` and
`g7_*.jsonl`, so move the stored files away to compute the runs again. The number of worker processes (last argument
of each call) only sets the parallelism. The database builds and the derating runs need several GB of memory.

## Map from the manuscript to the code

| Manuscript | Script |
|---|---|
| Section 3, Theorem 1, Supplementary Material S1 | `P3_kiemtoan/kiem_ghepcap.py` |
| Section 2.3, monotonicity in the thermal scale, Supplementary S4 and S14 | `P3_kiemtoan/g28_don_dieu_nhiet.py`, `g28b_don_dieu_khong_nen.py` |
| Section 2.4 and the certificates compared in Section 6.4 (U1–U9 in the code) | `P3_mophong/chung_chi.py` |
| Section 6.1, Fig. 3, Supplementary Table S8 | `P3_kiemtoan/thongke_G2.py`, `g3_tref_bien.py`, `hinh_cuoi_f3.py`; real-data coverage of the calibrated part (leave one condition out, across campaigns): `g19_kiem_du_lieu_that.py` |
| Section 6.2, Table 3, Supplementary Tables S19 and S22 | `P3_mophong/tuoi_tho.py` (stated margin), `P3_kiemtoan/g3_tref_bien.py`, `g10_tong_hop.py`, `g27_ton_ngoai_mau.py`, `P3_mophong/g5_ngan_sach.py` (budget), `han_de.py` (baseplate solder) |
| Section 6.3, missions and converter | `P3_mophong/du_lieu_hsr.py`, `ho_so.py`, `doan_tau.py`, `giam_tai_kha_thi.py`, `linh_kien_hv.py`, `ton_hao.py`, `nhiet.py`, `rainflow_ton.py`, `nhiem_vu.py` |
| Section 6.4, simulation design, Supplementary Section S12 | `P3_mophong/tuoi_tho.py` (`Scenario`, `World`), `thu_nghiem_g6.py` |
| Section 7.1, worked example, Supplementary Table S20 | `P3_mophong/vi_du_so.py`, `chia_nhieu.py` |
| Section 7.2, Fig. 4, Supplementary Tables S23 and S28 | `thu_nghiem_g6.py` (experiments A and P), `g6_tong_hop.py`, `vong4_tong_hop.py`, `hinh_cuoi.py`, `bang_cuoi.py`, `bang_vong4.py` |
| Algorithm 1, Section 7.3, Figs. 5–7, Table 4, Supplementary Tables S5, S7, S10, S14, S18 and S24 | `thu_nghiem_g6.py` (experiments C, R, M and F; `rep_C`, `rep_R`, `rep_M`, `rep_F`), `thu_nghiem_g5.py` (`reserve`), `hinh_quy_dao.py`, `bang_pb3.py` |
| Section 7.4, Figs. 8–9, Supplementary S2, Tables S9 and S27 | `thu_nghiem_g5.py` (experiment E3), `e3_bang_thong.py`, `e3_thiet_ke.py`, `e3_neo_co_dinh.py`, `e3_neo_bang_thong.py`, `thu_nghiem_g7.py`, `g7_tong_hop.py` |
| Section 7.5, Fig. 10, Supplementary S3, S4, S6, S13 and S14 | `thu_nghiem_g6.py` (experiments A, D, E, N, X, T, Y, L, Q and S), `g6_vuot_nhieu.py`, `dinh_nghia_ton.py` |
| Section 7.6, Supplementary Tables S25 and S26 | `thu_nghiem_g6.py` (experiments V and E), `dien_nhiet.py` (electro-thermal cycle databases), `so_sanh_et.py`, `et_chung_chi.py` |
| Tables 1–3, Supplementary Tables S2–S10 and S28 | `bang_cuoi.py` |
| Fig. 1, Fig. 2 | `P3_hinh_bang/F0_khung.tex`, `P3_hinh_bang/F1_cau_truc.tex` |

The fourth-round runs (Supplementary Section S15) are `P3_mophong/chay_vong4.sh` and `chay_vong4b.sh`; the
electro-thermal databases `chu_ky_*_d00_base_et.npz` are rebuilt by `dien_nhiet.py tables` and `dien_nhiet.py build`.

## License

Code: MIT, see `LICENSE`. The data files are derived from the sources above; please cite them.
