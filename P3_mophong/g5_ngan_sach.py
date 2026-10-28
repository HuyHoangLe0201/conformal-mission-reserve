"""P3 case study (G5): where the margined damage of the mission comes from, under the candidate margin rules.

Input: a cycle database of nhiem_vu.py. Lifetime model: the full fit on the 34 Hernes modules. For every chip, the mean
margined damage per vehicle day is split by kind (slow rainflow cycles, ripple with heating time >= 0.15 s, shorter
ripple) and, for the slow cycles, by swing (above or below dT_T = 26 K) and heating time (<= 4.5 s, 4.5-60 s, > 60 s).
The statistical part is left out here ('none'): it is a common factor of the interval and is added at interval level.

Run: python g5_ngan_sach.py chu_ky_cruise_d00_base.npz -> g5_ngan_sach.json and printed tables
     python g5_ngan_sach.py <npz> <out.json> base[,bare]   only the rules of the case study: 'base', the stated margin
                                                           with short pulses out of scope, and 'bare', no stated margin
  (g5_ngan_sach_chienluoc_cruise.json, g5_ngan_sach_chienluoc_coast.json: base,bare; g5_ngan_sach_mau_in.json: base)
"""
import json
import site
import sys
from dataclasses import replace
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
sys.path.append(site.getusersitepackages())
import numpy as np  # noqa: E402

import tuoi_tho as tt  # noqa: E402

RULES = {
    "G3 rules (t_on uncapped, short pulses clipped)": tt.Margin(stat="none", short_rule="clip"),
    "short pulses credited 0.59, t_on uncapped": tt.Margin(stat="none", short_rule="credit"),
    "short pulses credited 0.59, t_on capped at 60 s": tt.Margin(stat="none", short_rule="credit", ton_rule="cap"),
    "short pulses credited 0.59, t_on slope 0.27 above 60 s": tt.Margin(stat="none", short_rule="credit", ton_rule="slope"),
    "short pulses excluded, t_on uncapped": tt.Margin(stat="none", short_rule="exclude"),
    "short pulses excluded, t_on capped at 60 s": tt.Margin(stat="none", short_rule="exclude", ton_rule="cap"),
    "no stated margin (model only, kappa 1), short pulses clipped": tt.Margin(stat="none", gamma=0.0, gamma_t=0.0,
                                                                            ton_rule="cap", Ea=0.0, kappa_M=1.0,
                                                                            short_rule="clip"),
    "no stated margin, short pulses excluded": tt.Margin(stat="none", gamma=0.0, gamma_t=0.0, ton_rule="cap", Ea=0.0,
                                                         kappa_M=1.0, short_rule="exclude"),
    "no stated margin, short pulses credited 0.59": tt.Margin(stat="none", gamma=0.0, gamma_t=0.0, ton_rule="cap",
                                                              Ea=0.0, kappa_M=1.0, short_rule="credit"),
}
# Rules of the case study, under the names stored in the output: command-line name -> (output key, rule)
CASE = {"base": ("base", RULES["short pulses excluded, t_on uncapped"]),
        "bare": ("no stated margin", RULES["no stated margin, short pulses excluded"])}


def main(path, out_name="g5_ngan_sach.json", rules=RULES):
    d = np.load(HERE / path)
    cyc, T_amb, names = d["cycles"], d["T_amb"], [str(x) for x in d["names"]]
    nday = len(d["day"])
    h = tt.hernes()
    f = tt.fit(h["dT"], h["ton"], h["I"], h["Nf"], h["run"])
    tamb = np.zeros(int(cyc[:, 0].max()) + 1)
    tamb[d["day"]] = T_amb
    out = {"file": path, "days": nday}
    for rname, mg in rules.items():
        res = {}
        for c in sorted(set(cyc[:, 1].astype(int))):
            s = cyc[cyc[:, 1] == c]
            cc = dict(dTheta=s[:, 2].astype(float), theta_min=s[:, 3].astype(float), ton=s[:, 4].astype(float),
                      n=s[:, 5].astype(float), T_amb=tamb[s[:, 0].astype(int)])
            tot, parts = tt.model_damage(cc, f, mg, return_parts=True)
            keep = np.ones(len(s), bool) if mg.short_rule != "exclude" else cc["ton"] >= tt.TON_LO
            kind, dT, ton = s[keep, 6], cc["dTheta"][keep], cc["ton"][keep]
            per_day = lambda m: float(parts[m].sum() / nday)
            k0 = kind == 0
            res[names[c]] = {
                "per_day": tot / nday,
                "slow": per_day(k0), "ripple_ge_0.15s": per_day(kind == 1), "ripple_short": per_day(kind == 2),
                "slow_dT_below_26": per_day(k0 & (dT < 26)),
                "slow_ton_le_4.5": per_day(k0 & (ton <= 4.5)), "slow_ton_4.5_60": per_day(k0 & (ton > 4.5) & (ton <= 60)),
                "slow_ton_gt_60": per_day(k0 & (ton > 60)),
                "cycles_per_day_slow": float(s[s[:, 6] == 0, 5].sum() / nday),
            }
        crit = max(res, key=lambda k: res[k]["per_day"])
        c = names.index(crit)
        s_ = cyc[cyc[:, 1] == c]
        cc = dict(dTheta=s_[:, 2].astype(float), theta_min=s_[:, 3].astype(float), ton=s_[:, 4].astype(float),
                  n=s_[:, 5].astype(float), T_amb=tamb[s_[:, 0].astype(int)])
        _, parts = tt.model_damage(cc, f, mg, return_parts=True)
        keep = np.ones(len(s_), bool) if mg.short_rule != "exclude" else cc["ton"] >= tt.TON_LO
        dT, ton = cc["dTheta"][keep], cc["ton"][keep]
        Tmin = cc["T_amb"][keep] + cc["theta_min"][keep]
        w = parts / parts.sum()
        comp = {
            "swing gamma ln(dT_T/dT)": mg.gamma * np.log(np.maximum(mg.dT_T / np.maximum(dT, 1e-3), 1.0)),
            "heating 4.5-60 s": mg.gamma_t * np.log(np.clip(ton, 4.5, 60) / 4.5),
            "heating above 60 s": (mg.gamma_t if mg.ton_rule == "uncapped" else mg.gamma_h if mg.ton_rule == "slope"
                                   else 0.0) * np.log(np.maximum(ton / 60, 1.0)),
            "short-pulse credit": -(mg.gamma_s if mg.short_rule == "credit" else 0.0) * np.log(np.maximum(0.15 / ton, 1.0)),
            "T_min Arrhenius": mg.Ea / tt.K_B * np.maximum(1 / 333.15 - 1 / (Tmin + 273.15), 0.0),
        }
        budget = {k: float(np.sum(w * v)) for k, v in comp.items()}
        budget["Miner"] = mg.eps_M()
        print("  log budget of the critical chip (damage-weighted mean of each component): "
              + "; ".join(f"{k} {v:+.2f}" for k, v in budget.items()))
        out[rname] = {"critical_chip": crit, "life_days_critical": 1.0 / res[crit]["per_day"], "chips": res,
                      "log_budget_critical": budget}
        print(f"\n{rname}: critical chip {crit}, margined damage {res[crit]['per_day']:.3e} per day "
              f"(life {1 / res[crit]['per_day']:.0f} days)")
        print(f"  {'chip':4s} {'per day':>9s} {'slow':>6s} {'rip>=.15':>8s} {'rip<.15':>8s} | slow: {'dT<26':>6s} "
              f"{'t<=4.5':>6s} {'4.5-60':>6s} {'>60':>6s}  cycles/day")
        for k, v in res.items():
            tot = v["per_day"]
            print(f"  {k:4s} {tot:9.2e} {v['slow'] / tot:6.2f} {v['ripple_ge_0.15s'] / tot:8.2f} {v['ripple_short'] / tot:8.2f}"
                  f" | {v['slow_dT_below_26'] / tot:12.2f} {v['slow_ton_le_4.5'] / tot:6.2f} {v['slow_ton_4.5_60'] / tot:6.2f}"
                  f" {v['slow_ton_gt_60'] / tot:6.2f}  {v['cycles_per_day_slow']:6.1f}")
    (HERE / out_name).write_text(json.dumps(out, indent=1), encoding="utf-8", newline="\n")


if __name__ == "__main__":
    if len(sys.argv) > 3:
        main(sys.argv[1], sys.argv[2], dict(CASE[k] for k in sys.argv[3].split(",")))
    else:
        main(*sys.argv[1:3])
