"""P3: one leg through its life under a mission change (figure F10, 19/10/2026, after the editor re-review).

Replays repetition r of M_boundary_cruise_coast (thu_nghiem_g6.rep_M, same random stream) and records, interval by
interval, the true damage of the leg, the paired certificate of the coming interval on the certification data, the
certified reserve on the policy data, the change of driving strategy and the static ages set at commissioning. The
repetition is the first one in which the commissioning certificate age (at most 60 intervals) fails while the paired
certificate does not.

Run: python hinh_quy_dao.py -> ../P3_hinh_bang/F10.pdf, F10.png, hinh_quy_dao.json
"""
import json
import math
import site
import sys
from pathlib import Path

sys.path.insert(0, site.getusersitepackages())
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np  # noqa: E402

import chung_chi as cc  # noqa: E402
import kieu_hinh as kh  # noqa: E402
import thu_nghiem_g5 as g5  # noqa: E402
import thu_nghiem_g6 as g6  # noqa: E402
import tuoi_tho as tt  # noqa: E402

HERE = Path(__file__).resolve().parent
TAG = "M_boundary_cruise_coast"


def pick():
    for line in (HERE / f"g6_{TAG}.jsonl").read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        ru = r["runs"]
        if ru["none|cert_age"]["failed"] and not ru["cert|paired"]["failed"] and 15 <= r["t_sw"] <= 25 \
                and r["t_sw"] < r["ages"]["cert_age"] <= 60:
            return r
    raise SystemExit("no suitable repetition")


def replay(cfg, r):
    """The state of rep_M up to the life, in the same order of random draws."""
    rng = np.random.default_rng([cfg["seed"], r])
    sc = g6.scenario(cfg["scen"])
    pools, b_full = g6._W["pools"], g6._W["b_full"]
    pa, pb = pools[cfg["first"]], pools[cfg["second"]]
    alpha, n1, n2 = cfg["alpha"], cfg["n1"], cfg["n2"]
    world = tt.World.draw(rng, b_full, sc)
    bn, fi, si, fitd, X, S_all = g6.bench_scores(world, rng, 3 * n1)
    perm = rng.permutation(len(si))
    S_pol, S_cer = S_all[perm[:len(si) // 2]], S_all[perm[len(si) // 2:]]
    J1 = [pa.sample(rng) for _ in range(n2)]
    J2 = [pa.sample(rng) for _ in range(n2)]
    S_new, kap, s_th = g6.field_leg(world, sc, rng, pa.K)
    life = [pa.sample(rng) for _ in range(g6.T_MAX)]
    t_sw = int(rng.integers(cfg["t_sw"][0], cfg["t_sw"][1] + 1))
    VA1, VB1 = (np.array([p.model(q, fitd, g6.BASE) for q in J1]) for p in (pa, pb))
    VA2, VB2 = (np.array([p.model(q, fitd, g6.BASE) for q in J2]) for p in (pa, pb))
    return dict(world=world, fitd=fitd, S_pol=S_pol, S_cer=S_cer, life=life, t_sw=t_sw, VA1=VA1, VB1=VB1, VA2=VA2,
                VB2=VB2, S_new=S_new, kap=kap, s_th=s_th, pa=pa, pb=pb, alpha=alpha)


def trajectory(st):
    pa, pb, alpha, t_sw = st["pa"], st["pb"], st["alpha"], st["t_sw"]
    groups, nc = pa.groups, len(pa.chips)
    D0, Dt = np.zeros(nc), np.zeros(nc)
    rows = []
    for t in range(g6.T_MAX):
        V1, V2 = (st["VB1"], st["VB2"]) if t >= t_sw else (st["VA1"], st["VA2"])
        pol = [g5.reserve("paired", st["S_pol"], V1[:, g], D0[g], alpha) for g in groups]
        C_pol, R = max(c for c, _ in pol), min(r for _, r in pol)
        m = min(len(st["S_cer"]), len(V2))
        Cc = max(cc.u6_paired(st["S_cer"][:m], V2[:m][:, g], D0[g], alpha) for g in groups)
        if C_pol > 1:
            rows.append(dict(t=t, true=float(pa.mod(Dt).max()), C=None, R=R, replaced=True))
            break
        p = pa if t < t_sw else pb
        Dn = p.true(st["life"][t], st["world"], st["S_new"], st["kap"], st["s_th"])
        rows.append(dict(t=t, true_start=float(pa.mod(Dt).max()), true=float(pa.mod(Dt + Dn).max()), C=float(Cc), R=R,
                         replaced=False))
        D0, Dt = D0 + p.model(st["life"][t], st["fitd"], g6.BASE), Dt + Dn
        if Dt.max() >= 1:
            break
    # the same leg run on without replacement, to show where a static age would have ended
    Dr, run_on = Dt.copy(), []
    for t in range(len(rows), g6.T_MAX):
        p = pa if t < t_sw else pb
        Dr = Dr + p.true(st["life"][t], st["world"], st["S_new"], st["kap"], st["s_th"])
        run_on.append(dict(t=t, true=float(pa.mod(Dr).max())))
        if Dr.max() >= 1:
            break
    return rows, run_on


def main():
    g6._init("M")
    cfg = g6.CONFIGS[TAG]
    rec = pick()
    st = replay(cfg, rec["rep"])
    rows, run_on = trajectory(st)
    out = dict(rep=rec["rep"], t_sw=rec["t_sw"], ages=rec["ages"], replanned=rec["replanned"], rows=rows, run_on=run_on,
               stored=rec["runs"]["cert|paired"])
    (HERE / "hinh_quy_dao.json").write_text(json.dumps(out, indent=1), encoding="utf-8", newline="\n")

    fig, ax = kh.figure(kh.W2, 62, 1, 2)
    a, b = ax
    live = [x for x in rows if not x["replaced"]]
    t_end = live[-1]["t"] + 1
    tt_ = np.arange(1, t_end + 1)
    a.step(tt_, [x["C"] for x in live], where="pre", color=kh.BLUE, lw=1.0, label="paired certificate of the coming interval")
    a.plot(tt_, [x["true"] for x in live], color=kh.BLACK, lw=1.2, label="true damage of the leg")
    ro = [x for x in run_on]
    if ro:
        a.plot([t_end] + [x["t"] + 1 for x in ro], [live[-1]["true"]] + [x["true"] for x in ro], color=kh.BLACK, lw=0.8,
               ls=":", label="true damage if run on")
    a.axhline(1.0, color=kh.GREY, lw=0.6)
    a.text(1, 1.02, "damage budget", fontsize=6.5, color=kh.GREY, va="bottom")
    a.axvline(rec["t_sw"], color=kh.GREY, lw=0.6, ls="--")
    a.text(rec["t_sw"] - 0.5, 1.17, "cruising $\\to$ coasting", fontsize=6.5, color=kh.GREY, va="top", ha="right")
    a.axvline(t_end, color=kh.BLUE, lw=0.8)
    a.text(t_end - 0.5, 0.05, "replaced by\nthe certificate", fontsize=6.5, color=kh.BLUE, ha="right", va="bottom")
    for key, lab, col in (("cert_age", "certificate age at commissioning", kh.ORANGE),
                          ("mc_opt10", "Monte Carlo optimal age", kh.SKY)):
        ag = rec["ages"][key]
        a.axvline(ag, color=col, lw=0.8, ls="-.", label=f"{lab} ({ag})")
    a.set_xlim(0, max(t_end, max(x["t"] for x in ro) + 1 if ro else t_end, rec["ages"]["cert_age"]) + 2)
    a.set_ylim(0, 1.2)
    a.set_xlabel("maintenance interval")
    a.set_ylabel("damage")
    b.step(tt_, [x["R"] for x in live], where="pre", color=kh.BLUE, lw=1.0, label="certified reserve (policy data)")
    b.plot(tt_, np.maximum(0, cfg["t_plan"] - (tt_ - 1)), color=kh.GREY, lw=0.6, ls="--",
           label="intervals left to the planned 60")
    b.axvline(rec["t_sw"], color=kh.GREY, lw=0.6, ls="--")
    b.axhline(1.0, color=kh.BLUE, lw=0.5, ls=":")
    b.set_xlim(a.get_xlim())
    b.set_ylim(0, max(x["R"] for x in live) * 1.1)
    b.set_xlabel("maintenance interval")
    b.set_ylabel("intervals")
    ha, la = a.get_legend_handles_labels()
    hb, lb = b.get_legend_handles_labels()
    fig.legend(ha + hb, la + lb, loc="lower center", ncol=4, fontsize=6.5, bbox_to_anchor=(0.5, 0.0),
               columnspacing=1.2, handlelength=2.2)
    for ax_, letter in zip(ax, "ab"):
        kh.label(ax_, letter)
    fig.tight_layout(w_pad=1.5, rect=(0, 0.17, 1, 1))
    kh.save(fig, "F10")
    print("rep", rec["rep"], "t_sw", rec["t_sw"], "ages", rec["ages"], "replaced at", t_end,
          "stored paired intervals", rec["runs"]["cert|paired"]["intervals"])


if __name__ == "__main__":
    main()
