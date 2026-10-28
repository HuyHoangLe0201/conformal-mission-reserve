"""P3 case study (G4): vehicle days, dwell windows and maintenance intervals on the Beijing-Shanghai corridor.

Input: hsr_runs_<tag>.csv from du_lieu_hsr.py (cleaned stop-to-stop runs); train model doan_tau.py.

Run model. Each run is driven by doan_tau with one of two strategies that bracket the inverter load ('cruise': lowest
cruise speed meeting the run time; 'coast': line speed, then coasting). Run times are checked against the minimum at
line speed, T_fast:
  ok          recorded time >= T_fast;
  rounding    within ROUND_MIN below T_fast (one-minute resolution of the data): driven at line speed;
  act->sched  actual time more than ROUND_MIN below T_fast while the scheduled time is not: the actual time is a
              recording error that passed the 30-min rule of du_lieu_hsr.py, and the scheduled time is used;
  short       both times below T_fast - ROUND_MIN: driven at line speed.
When a modelled arrival is later than the recorded departure from that stop, the departure moves to the arrival plus
DWELL_MIN, and the delay carries on.

Vehicle days. The data give train-days (one train number on one date), not the units that run them. A unit's day is
built by the greedy rule of vehicle scheduling: on each date trips are taken in order of departure; a trip goes to
the unit idle at its origin terminus that has waited longest, if the turnaround is at least T_TURN_MIN and the unit has
run fewer than MAX_TRIPS trips that day; otherwise a new unit starts. Two trips (2636 km) are about the daily distance
of a 350 km/h unit between night inspections. Rolling-stock diagrams are not in the data.

Dwell windows. Intervals at standstill with no traction current: intermediate dwells and turnarounds (the rest between
days is longer than any of them). They are the windows of Remark P3.27 in which test pulses can identify the thermal
impedance. Coasting phases also carry no current but the train is moving; they are counted separately.

Maintenance intervals. An interval of N days is N vehicle days drawn with replacement from the pool, so intervals are
exchangeable by construction. A date filter gives shifted pools (for example December-January against
October-November) for the weighted-conformal experiments.

Run: python ho_so.py <hsr_runs_tag.csv> <tag>
  -> hsr_runs_model_<tag>.csv, hsr_days_<tag>.csv, ho_so_<tag>.json, ho_so_<tag>_nhap.pdf/.png
"""
import collections
import csv
import json
import math
import site
import sys
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
sys.path.append(site.getusersitepackages())
import numpy as np  # noqa: E402

import doan_tau as dt  # noqa: E402

T_TURN_MIN = 30.0      # minimum turnaround at a terminus, min
MAX_TRIPS = 2          # trips per vehicle day
DWELL_MIN = 1.0        # shortest dwell kept when a modelled run arrives late, min
ROUND_MIN = 1.0        # one-minute resolution of the timetable data
INTERVAL_DAYS = 30
STRATEGIES = ("cruise", "coast")


def load_trips(path):
    """Train-days of the run table as trips, each with its runs in order."""
    trips = collections.defaultdict(list)
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            for k in ("L_km", "T_run_min", "T_sched_min", "dep_min", "arr_min"):
                r[k] = float(r[k])
            r["dwell_min"] = float(r["dwell_min"]) if r["dwell_min"] else None
            r["seq"] = int(r["seq"])
            r["temperature_C"] = float(r["temperature_C"]) if r["temperature_C"] not in ("", None) else math.nan
            trips[(r["date"], r["train"], r["direction"])].append(r)
    out = []
    for (date, train, direction), runs in trips.items():
        runs.sort(key=lambda r: r["seq"])
        out.append(dict(date=date, train=train, direction=direction, runs=runs, origin=runs[0]["from_station"],
                        terminus=runs[-1]["to_station"], dep=runs[0]["dep_min"], arr=runs[-1]["arr_min"]))
    return out


class RunModel:
    """Model of single runs for one parameter set and one driving strategy, cached by (distance, run time)."""

    def __init__(self, strategy="cruise", p=dt.P):
        self.p, self.strategy = p, strategy
        self.acc = dt.Acceleration(p)
        self.coast = dt.Coasting(p) if strategy == "coast" else None
        self.cache, self.fast = {}, {}

    def t_fast_min(self, L_km):
        if L_km not in self.fast:
            self.fast[L_km] = dt.run_time(L_km * 1e3, self.p["v_line"], self.acc, self.p)[0] / 60
        return self.fast[L_km]

    def time_used(self, r):
        """Run time used by the model (min) and the check flag."""
        tf = self.t_fast_min(r["L_km"])
        T = r["T_run_min"]
        if T >= tf:
            return T, "ok"
        if T >= tf - ROUND_MIN:
            return tf, "rounding"
        if r["T_src"] == "act" and r["T_sched_min"] >= tf - ROUND_MIN:
            return max(r["T_sched_min"], tf), "act->sched"
        return tf, "short"

    def profile(self, L_km, T_min):
        return dt.profile(L_km * 1e3, max(T_min, self.t_fast_min(L_km)) * 60, self.acc, self.p, self.coast)

    def summary(self, L_km, T_min):
        key = (round(L_km, 3), round(T_min, 3))
        if key in self.cache:
            return self.cache[key]
        prof, s = self.profile(L_km, T_min)
        el = dt.electrical(prof["v"], prof["F_e"], dt.P)
        h = np.diff(prof["t"], append=prof["t"][-1])
        I, Pe = el["I_pk"], el["P_e"]
        res = dict(T_model_min=s["T_model_s"] / 60, v_top_kmh=s["v_peak_kmh"], v_brake_kmh=s["v_brake_kmh"],
                   t_accel_s=s["t_accel_s"], t_cruise_s=s["t_cruise_s"], t_coast_s=s["t_coast_s"],
                   t_brake_s=s["t_brake_s"],
                   E_trac_kWh=float(np.sum(np.maximum(Pe, 0) * h)) / 3.6e6,
                   E_regen_kWh=float(np.sum(np.minimum(Pe, 0) * h)) / 3.6e6,
                   t_rated_s=float(np.sum((I >= 0.95 * dt.P["I_rated"]) * h)),
                   I_rms_A=float(np.sqrt(np.sum(I ** 2 * h) / np.sum(h))))
        self.cache[key] = res
        return res

    def timeline(self, trip):
        """Modelled departure and arrival of each run (min after midnight of the date), run summary and flag."""
        out, t_free = [], -1e9
        for r in trip["runs"]:
            T, flag = self.time_used(r)
            s = self.summary(r["L_km"], T)
            dep = max(r["dep_min"], t_free)
            arr = dep + s["T_model_min"]
            out.append((dep, arr, s, flag))
            t_free = arr + DWELL_MIN
        return out


def vehicle_days(trips, t_turn=T_TURN_MIN, max_trips=MAX_TRIPS):
    """Greedy first-in-first-out assignment of trips to units, date by date."""
    by_date = collections.defaultdict(list)
    for x in trips:
        by_date[x["date"]].append(x)
    days = []
    for date in sorted(by_date):
        tr = sorted(by_date[date], key=lambda x: x["dep"])
        idle = collections.defaultdict(list)
        units = []
        for x in tr:
            pool = idle[x["origin"]]
            cand = [u for u in pool if u[0] + t_turn <= x["dep"] and len(units[u[1]]) < max_trips]
            if cand:
                u = min(cand, key=lambda u: u[0])
                pool.remove(u)
                ui = u[1]
            else:
                ui = len(units)
                units.append([])
            units[ui].append(x)
            idle[x["terminus"]].append([x["arr"], ui])
        days.extend(dict(date=date, unit=i, trips=u) for i, u in enumerate(units))
    return days


def day_windows(day, model):
    """Standstill windows of a vehicle day: (start_min, length_min, kind, station)."""
    win = []
    prev_arr = None
    for j, trip in enumerate(day["trips"]):
        tl = model.timeline(trip)
        if j > 0:
            win.append((prev_arr, tl[0][0] - prev_arr, "turnaround", trip["origin"]))
        for k in range(len(tl) - 1):
            win.append((tl[k][1], tl[k + 1][0] - tl[k][1], "dwell", trip["runs"][k]["to_station"]))
        prev_arr = tl[-1][1]
    return win


def day_series(day, model, step_s=None):
    """Time series of one vehicle day for one inverter, on a uniform grid from the first departure to the last
    arrival: t (s after midnight of the date), v (m/s), I_pk, m, cosphi, f_e, P_e, state (0 dwell, 1 accelerate,
    2 cruise, 3 brake, 4 turnaround, 5 coast) and ambient temperature (degC, of the last departure station)."""
    h = step_s or model.p["dt"]
    tls = [model.timeline(trip) for trip in day["trips"]]
    t0, t1 = tls[0][0][0] * 60, tls[-1][-1][1] * 60
    t = np.arange(t0, t1 + h, h)
    v = np.zeros_like(t)
    F = np.zeros_like(t)
    state = np.zeros(t.size, dtype=np.int8)
    amb = np.full(t.size, np.nan)
    for j, (trip, tl) in enumerate(zip(day["trips"], tls)):
        if j > 0:
            state[(t > tls[j - 1][-1][1] * 60) & (t < tl[0][0] * 60)] = 4
        for r, (dep, arr, s, flag) in zip(trip["runs"], tl):
            prof, _ = model.profile(r["L_km"], model.time_used(r)[0])
            sel = (t >= dep * 60) & (t <= arr * 60)
            tau = t[sel] - dep * 60
            v[sel] = np.interp(tau, prof["t"], prof["v"])
            F[sel] = np.interp(tau, prof["t"], prof["F_e"])
            state[sel] = prof["phase"][np.minimum(np.searchsorted(prof["t"], tau), prof["t"].size - 1)]
            amb[t >= dep * 60] = r["temperature_C"]
    el = dt.electrical(v, F, dt.P)
    return dict(t=t, v=v, state=state, ambient_C=amb, **el)


def sample_interval(days, n_days, rng, date_range=None):
    """Indices of the vehicle days of one maintenance interval, drawn with replacement."""
    pool = [i for i, d in enumerate(days) if date_range is None or date_range[0] <= d["date"] <= date_range[1]]
    return rng.choice(pool, n_days, replace=True)


def pct(x, q=(5, 25, 50, 75, 95)):
    return {f"p{k}": round(float(v), 3) for k, v in zip(q, np.percentile(np.asarray(x, dtype=float), q))}


def main(runs_path, tag):
    trips = load_trips(runs_path)
    models = {s: RunModel(s) for s in STRATEGIES}
    base = models["cruise"]

    # runs
    run_rows = []
    for trip in trips:
        tls = {s: m.timeline(trip) for s, m in models.items()}
        for k, r in enumerate(trip["runs"]):
            dep, arr, s0, flag = tls["cruise"][k]
            row = dict(date=r["date"], train=r["train"], direction=r["direction"], seq=r["seq"],
                       from_station=r["from_station"], to_station=r["to_station"], L_km=r["L_km"],
                       T_run_min=r["T_run_min"], T_src=r["T_src"], flag=flag, T_model_min=round(s0["T_model_min"], 3),
                       dep_model_min=round(dep, 3), arr_model_min=round(arr, 3), temperature_C=r["temperature_C"])
            for sname in STRATEGIES:
                s = tls[sname][k][2]
                row.update({f"v_top_kmh_{sname}": round(s["v_top_kmh"], 1),
                            f"t_rated_s_{sname}": round(s["t_rated_s"], 1),
                            f"t_coast_s_{sname}": round(s["t_coast_s"], 1),
                            f"E_trac_kWh_{sname}": round(s["E_trac_kWh"], 3),
                            f"E_regen_kWh_{sname}": round(s["E_regen_kWh"], 3),
                            f"I_rms_A_{sname}": round(s["I_rms_A"], 1)})
            run_rows.append(row)
    with open(HERE / f"hsr_runs_model_{tag}.csv", "w", newline="\n", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=list(run_rows[0]), lineterminator="\n")
        wr.writeheader()
        wr.writerows(run_rows)

    # vehicle days and windows (the timeline is the same for both strategies: both use max(T, T_fast))
    days = vehicle_days(trips)
    day_rows, all_win = [], []
    for d in days:
        win = day_windows(d, base)
        all_win.extend(win)
        runs = [r for trip in d["trips"] for r in trip["runs"]]
        tl = {s: [x for trip in d["trips"] for x in m.timeline(trip)] for s, m in models.items()}
        dw = [w[1] for w in win if w[2] == "dwell"]
        turn = [w[1] for w in win if w[2] == "turnaround"]
        temps = [r["temperature_C"] for r in runs if not math.isnan(r["temperature_C"])]
        b = tl["cruise"]
        row = dict(date=d["date"], unit=d["unit"], trains=" ".join(x["train"] for x in d["trips"]),
                   n_trips=len(d["trips"]), n_runs=len(runs), km=round(sum(r["L_km"] for r in runs), 1),
                   first_dep_min=round(b[0][0], 1), last_arr_min=round(b[-1][1], 1),
                   run_h=round(sum(a - c for c, a, _, _ in b) / 60, 3), dwell_n=len(dw),
                   dwell_ge2_n=sum(x >= 2 for x in dw), dwell_ge5_n=sum(x >= 5 for x in dw),
                   dwell_ge10_n=sum(x >= 10 for x in dw), turnaround_min=round(turn[0], 1) if turn else "",
                   runs_flag_short=sum(x[3] == "short" for x in b),
                   T_amb_min=min(temps) if temps else "", T_amb_max=max(temps) if temps else "")
        for sname in STRATEGIES:
            row.update({f"E_trac_kWh_{sname}": round(sum(x[2]["E_trac_kWh"] for x in tl[sname]), 2),
                        f"t_rated_h_{sname}": round(sum(x[2]["t_rated_s"] for x in tl[sname]) / 3600, 3),
                        f"t_coast_h_{sname}": round(sum(x[2]["t_coast_s"] for x in tl[sname]) / 3600, 3)})
        day_rows.append(row)
    with open(HERE / f"hsr_days_{tag}.csv", "w", newline="\n", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=list(day_rows[0]), lineterminator="\n")
        wr.writeheader()
        wr.writerows(day_rows)

    # maintenance intervals
    rng = np.random.default_rng(20261016)
    iv = [sample_interval(days, INTERVAL_DAYS, rng) for _ in range(2000)]
    starts_day = np.array([r["n_runs"] for r in day_rows])
    early = sample_interval(days, 1, rng, ("2019-10-01", "2019-11-30"))
    intervals = dict(days=INTERVAL_DAYS, n=len(iv), starts=pct([starts_day[i].sum() for i in iv]),
                     early_pool_ok=bool(len(early) == 1))
    for sname in STRATEGIES:
        E_day = np.array([r[f"E_trac_kWh_{sname}"] for r in day_rows])
        rated = np.array([r[f"t_rated_h_{sname}"] for r in day_rows])
        iv_E = np.array([E_day[i].sum() for i in iv])
        intervals[sname] = dict(E_trac_kWh=pct(iv_E), cv_E=round(float(iv_E.std() / iv_E.mean()), 4),
                                t_rated_h=pct([rated[i].sum() for i in iv]))

    # summary
    flags = collections.Counter(r["flag"] for r in run_rows)
    dw = np.array([w[1] for w in all_win if w[2] == "dwell"])
    turn = np.array([w[1] for w in all_win if w[2] == "turnaround"])
    temp = np.array([r["temperature_C"] for r in run_rows])
    by_month = collections.defaultdict(list)
    for r in run_rows:
        by_month[r["date"][:7]].append(r["temperature_C"])
    Pw_max, F_max, v_base = dt.limits()
    runs_s = {}
    for sname in STRATEGIES:
        runs_s[sname] = dict(
            v_top_kmh=pct([r[f"v_top_kmh_{sname}"] for r in run_rows]),
            t_rated_s=pct([r[f"t_rated_s_{sname}"] for r in run_rows]),
            t_coast_s=pct([r[f"t_coast_s_{sname}"] for r in run_rows]),
            E_trac_kWh_per_inverter=pct([r[f"E_trac_kWh_{sname}"] for r in run_rows]),
            I_rms_A=pct([r[f"I_rms_A_{sname}"] for r in run_rows]),
            distinct_models=len(models[sname].cache))
    days_s = {sname: dict(E_trac_kWh_per_inverter=pct([r[f"E_trac_kWh_{sname}"] for r in day_rows]),
                          t_rated_h=pct([r[f"t_rated_h_{sname}"] for r in day_rows]),
                          t_coast_h=pct([r[f"t_coast_h_{sname}"] for r in day_rows])) for sname in STRATEGIES}
    out = dict(
        train=dict(F_max_kN=round(F_max / 1e3, 1), wheel_power_MW=round(Pw_max / 1e6, 2),
                   base_speed_kmh=round(3.6 * v_base, 1), line_speed_kmh=round(3.6 * dt.P["v_line"]),
                   top_speed_kmh=round(3.6 * dt.top_speed(), 1)),
        rules=dict(T_TURN_MIN=T_TURN_MIN, MAX_TRIPS=MAX_TRIPS, DWELL_MIN=DWELL_MIN, ROUND_MIN=ROUND_MIN,
                   INTERVAL_DAYS=INTERVAL_DAYS),
        runs=dict(n=len(run_rows), flags=dict(flags), L_km=pct([r["L_km"] for r in run_rows]),
                  T_run_min=pct([r["T_run_min"] for r in run_rows]), **runs_s),
        dwell_windows=dict(n=int(dw.size), length_min=pct(dw, (1, 5, 25, 50, 75, 95, 99)),
                           share_ge=dict((f"{k}min", round(float(np.mean(dw >= k)), 4)) for k in (2, 3, 5, 10, 15)),
                           per_day=pct([r["dwell_n"] for r in day_rows]),
                           ge5_per_day=pct([r["dwell_ge5_n"] for r in day_rows]),
                           days_without_ge5=round(float(np.mean([r["dwell_ge5_n"] == 0 for r in day_rows])), 4),
                           days_without_ge5_or_turnaround=round(float(np.mean(
                               [r["dwell_ge5_n"] == 0 and r["n_trips"] == 1 for r in day_rows])), 4)),
        turnarounds=dict(n=int(turn.size), length_min=pct(turn) if turn.size else {}),
        vehicle_days=dict(n=len(day_rows), trips=dict(collections.Counter(r["n_trips"] for r in day_rows)),
                          km=pct([r["km"] for r in day_rows]), run_h=pct([r["run_h"] for r in day_rows]),
                          first_dep_h=pct([r["first_dep_min"] / 60 for r in day_rows]),
                          last_arr_h=pct([r["last_arr_min"] / 60 for r in day_rows]),
                          starts=pct(starts_day), days_with_short_runs=int(sum(r["runs_flag_short"] > 0 for r in day_rows)),
                          **days_s),
        intervals=intervals,
        ambient_C=dict(all=pct(temp[~np.isnan(temp)]),
                       by_month={m: pct(np.array(v)[~np.isnan(np.array(v))], (5, 50, 95)) for m, v in sorted(by_month.items())},
                       missing=int(np.isnan(temp).sum())))
    (HERE / f"ho_so_{tag}.json").write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8",
                                            newline="\n")
    print(json.dumps(out, indent=1, ensure_ascii=False))
    figure(days, day_rows, models, run_rows, dw, tag)


def figure(days, day_rows, models, run_rows, dw, tag):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # example: the two-trip day closest to the median running time
    two = [i for i, r in enumerate(day_rows) if r["n_trips"] == 2]
    med = np.median([day_rows[i]["run_h"] for i in two])
    ex = min(two, key=lambda i: abs(day_rows[i]["run_h"] - med))
    fig = plt.figure(figsize=(7.2, 8.4))
    for j, sname in enumerate(STRATEGIES):
        ser = day_series(days[ex], models[sname])
        th = ser["t"] / 3600
        ax = fig.add_subplot(4, 1, j + 1)
        ax.plot(th, 3.6 * ser["v"], lw=0.6, color="k")
        ax.set_ylabel("speed (km/h)")
        ax.set_ylim(-10, 370)
        ax2 = ax.twinx()
        ax2.plot(th, ser["I_pk"], lw=0.4, color="tab:red", alpha=0.7)
        ax2.set_ylabel("$I_{pk}$ (A)", color="tab:red")
        ax2.set_ylim(-15, 500)
        for w in day_windows(days[ex], models[sname]):
            ax.axvspan(w[0] / 60, (w[0] + w[1]) / 60, color="tab:blue" if w[2] == "dwell" else "tab:green",
                       alpha=0.25, lw=0)
        if j == 1:
            ax.set_xlabel("time of day (h)")
        ax.set_title(f"({'ab'[j]}) '{sname}' strategy, vehicle day {days[ex]['date']}: {day_rows[ex]['trains']}; "
                     f"dwell (blue), turnaround (green)", fontsize=8)
    ax = fig.add_subplot(4, 2, 5)
    ax.hist([r["T_run_min"] for r in run_rows], bins=np.arange(0, 200, 5), color="0.4")
    ax.set_xlabel("run time (min)")
    ax.set_ylabel("runs")
    ax.set_title("(c) stop-to-stop run time", fontsize=8)
    ax = fig.add_subplot(4, 2, 6)
    ax.hist(dw, bins=np.arange(0, 31, 1), color="tab:blue")
    ax.set_xlabel("dwell (min)")
    ax.set_title("(d) dwell windows", fontsize=8)
    ax = fig.add_subplot(4, 2, 7)
    for sname, c in zip(STRATEGIES, ("0.3", "tab:orange")):
        ax.hist([r[f"t_rated_h_{sname}"] for r in day_rows], bins=np.arange(0, 3.85, 0.05), color=c, alpha=0.7,
                label=sname)
    ax.legend(fontsize=7, frameon=False)
    ax.set_xlabel("time at rated current per day (h)")
    ax.set_ylabel("vehicle days")
    ax.set_title("(e) load of one inverter", fontsize=8)
    ax = fig.add_subplot(4, 2, 8)
    dates = sorted({r["date"] for r in run_rows})
    tmed = collections.defaultdict(list)
    for r in run_rows:
        if not math.isnan(r["temperature_C"]):
            tmed[r["date"]].append(r["temperature_C"])
    x = np.arange(len(dates))
    ax.fill_between(x, [np.percentile(tmed[d], 5) for d in dates], [np.percentile(tmed[d], 95) for d in dates],
                    color="0.8")
    ax.plot(x, [np.median(tmed[d]) for d in dates], color="k", lw=0.8)
    ticks = [i for i, d in enumerate(dates) if d.endswith("-01")]
    ax.set_xticks(ticks, [dates[i][:7] for i in ticks], fontsize=7)
    ax.set_ylabel("ambient (degC)")
    ax.set_title("(f) station temperature, median and 5-95 %", fontsize=8)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(HERE / f"ho_so_{tag}_nhap.{ext}", dpi=150)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
