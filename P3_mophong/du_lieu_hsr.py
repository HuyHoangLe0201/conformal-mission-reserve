"""P3 case study (G4): stop-to-stop runs of one high-speed corridor from the dataset of Zhang et al. (2022).

Data (CC0): figshare doi:10.6084/m9.figshare.15087882.v4. Two of its files are kept in ../Tailieu_P3/zhang2022hsr/;
both match the MD5 published by figshare (checked 16/10/2026):
  'high-speed trains operation data.csv'        f8f5ee9dab655ca59fcc6ab9756b2cd1, 365416910 bytes: scheduled and
                                                actual arrival and departure per train, station and day (one-minute
                                                resolution), stop time, delays, daily weather and temperature;
  'adjacent railway stations mileage data.csv'  c098b0e138dd6cc3fc80b575191e3695, 205006 bytes: distance in km
                                                between consecutive stops, by direction.
The files are downloaded data: they are only read, never executed, and their paths are passed as arguments.

Steps:
  survey  number of train-days per (origin, terminus, direction), to choose the corridor;
  build   every train-day of the chosen corridor is cut into stop-to-stop runs, with distance, run time and the dwell
          at the next stop, cleaned by the rules below.

Cleaning rules (fixed from the data checks of 16/10/2026, before any thermal result):
  distance  The mileage file has a few wrong entries: Bangbunan is offset by 54 km in the up direction, and
            Tianjinnan-Cangzhouxi is listed as 168 km down and 95 km up for 88 km. The chainage of each station is the
            median cumulative distance over the train-days whose listed distances add up to the line length (the most
            frequent sum); each run takes the chainage difference.
  run time  Actual time if it differs from the scheduled time by at most 30 min, otherwise the scheduled time. Isolated
            actual times shifted by hours occur, for example G101 on 1/1/2020 at Qufudong: 20:47 instead of 09:13,
            with both neighbours on time.
  dwell     Actual dwell if it lies in [0, 60] min and the actual times of both adjacent runs were kept, otherwise
            the scheduled dwell.
  timeline  First departure as recorded (actual if the first run kept its actual time); later times are rebuilt
            from the cleaned run times and dwells, so every train-day is consistent.

Run: python du_lieu_hsr.py survey <operation.csv> <mileage.csv>
     python du_lieu_hsr.py build <operation.csv> <mileage.csv> <origin> <terminus> <tag>
       -> hsr_runs_<tag>.csv and hsr_runs_<tag>_checks.json
"""
import collections
import csv
import hashlib
import json
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).parent
MAX_DEV_MIN = 30.0     # largest accepted |actual - scheduled| run time
MAX_DWELL_MIN = 60.0   # largest accepted actual dwell
SUFFIX = " Railway Station"


def minutes(hms):
    """'HH:MM:SS' or 'H:MM:SS' -> minutes after midnight, None if missing."""
    try:
        h, m, s = hms.strip().split(":")
        return int(h) * 60 + int(m) + int(s) / 60
    except (ValueError, AttributeError):
        return None


def read_rows(path):
    """Rows as dicts; a row with missing fields (a truncated last line) is skipped."""
    with open(path, encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            if None in r.values() or None in r:
                continue
            if "station_order" in r and not r["station_order"].isdigit():
                continue
            yield r


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def mileage_table(path):
    tab = {}
    for r in read_rows(path):
        try:
            km = float(r["mileage"])
        except ValueError:
            continue
        tab[(r["from_station"], r["to_station"], r["train_direction"])] = km
    return tab


def lookup(tab, a, b, direction):
    for key in ((a, b, direction), (a, b, "down"), (a, b, "up"), (b, a, "down"), (b, a, "up")):
        if key in tab:
            return tab[key]
    return None


def train_day_ends(op_path):
    ends = {}
    for r in read_rows(op_path):
        key = (r["date"], r["train_number"], r["train_direction"])
        o = int(r["station_order"])
        lo, hi = ends.get(key, (None, None))
        if lo is None or o < lo[0]:
            lo = (o, r["station_name"])
        if hi is None or o > hi[0]:
            hi = (o, r["station_name"])
        ends[key] = (lo, hi)
    return ends


def survey(op_path):
    ends = train_day_ends(op_path)
    count = collections.Counter()
    trains = collections.defaultdict(set)
    for (date, train, direction), (lo, hi) in ends.items():
        count[(lo[1], hi[1], direction)] += 1
        trains[(lo[1], hi[1], direction)].add(train)
    print(f"{len(ends):,} train-days, {len({k[1] for k in ends}):,} train numbers, "
          f"{len({k[0] for k in ends})} days")
    for (o, t, d), n in count.most_common(25):
        print(f"  {n:6,} train-days  {len(trains[(o, t, d)]):3} trains  {d:4}  {o} -> {t}")


def unwrap(rows, checks):
    """Scheduled and actual times of each stop in minutes, with midnight crossings unwrapped: a time that runs
    backwards by more than 10 minutes belongs to the next day."""
    out = []
    off = {"s": 0.0, "a": 0.0}
    last = {"s": None, "a": None}
    for r in rows:
        vals = []
        for col, which in (("scheduled_arrival_time", "s"), ("scheduled_departure_time", "s"),
                           ("actual_arrival_time", "a"), ("actual_departure_time", "a")):
            v = minutes(r[col])
            if v is None:
                vals.append(None)
                continue
            if last[which] is not None and v + off[which] < last[which] - 10:
                off[which] += 1440
                if which == "s":
                    checks["midnight_crossings_scheduled"] += 1
            v += off[which]
            last[which] = v
            vals.append(v)
        out.append(vals)
    return out


def build(op_path, mil_path, origin, terminus, tag):
    checks = collections.Counter()
    tab = mileage_table(mil_path)
    ends = train_day_ends(op_path)
    keys = {k for k, (lo, hi) in ends.items() if {lo[1], hi[1]} == {origin, terminus}}
    days = collections.defaultdict(list)
    for r in read_rows(op_path):
        k = (r["date"], r["train_number"], r["train_direction"])
        if k in keys:
            days[k].append(r)

    # pass 1: raw runs with listed distances
    raw = {}
    for key, rows in days.items():
        rows.sort(key=lambda r: int(r["station_order"]))
        times = unwrap(rows, checks)
        runs = []
        for k in range(len(rows) - 1):
            a, b = rows[k], rows[k + 1]
            (_, dep_s0, _, dep_a0), (arr_s1, dep_s1, arr_a1, dep_a1) = times[k], times[k + 1]
            last = k + 1 == len(rows) - 1
            runs.append(dict(
                frm=a["station_name"], to=b["station_name"],
                L_listed=lookup(tab, a["station_name"], b["station_name"], key[2]),
                T_s=None if None in (dep_s0, arr_s1) else arr_s1 - dep_s0,
                T_a=None if None in (dep_a0, arr_a1) else arr_a1 - dep_a0,
                dw_s=None if last or None in (arr_s1, dep_s1) else dep_s1 - arr_s1,
                dw_a=None if last or None in (arr_a1, dep_a1) else dep_a1 - arr_a1,
                dep_s=dep_s0, dep_a=dep_a0,
                temp=a["temperature"], weather=a["weather"], holiday=a["major_holiday"]))
        raw[key] = runs

    # chainage from the train-days whose listed distances add up to the line length
    sums = collections.Counter(round(sum(r["L_listed"] for r in runs), 1) for runs in raw.values()
                               if all(r["L_listed"] is not None for r in runs))
    line_km = sums.most_common(1)[0][0]
    cum = collections.defaultdict(list)
    n_chain_days = 0
    for key, runs in raw.items():
        if any(r["L_listed"] is None for r in runs) or abs(sum(r["L_listed"] for r in runs) - line_km) > 0.05:
            continue
        n_chain_days += 1
        c = 0.0
        first = runs[0]["frm"]
        pts = [(first, 0.0)]
        for r in runs:
            c += r["L_listed"]
            pts.append((r["to"], c))
        for s, v in pts:
            cum[s].append(v if first == origin else line_km - v)
    chain = {s: statistics.median(v) for s, v in cum.items()}
    chain_spread = {s: max(v) - min(v) for s, v in cum.items()}

    # pass 2: cleaned runs and rebuilt timeline
    out_rows, wrong_pairs = [], collections.Counter()
    for (date, train, direction), runs in raw.items():
        checks["train_days"] += 1
        Ls = []
        for r in runs:
            if r["frm"] in chain and r["to"] in chain:
                L = abs(chain[r["to"]] - chain[r["frm"]])
                if r["L_listed"] is not None and abs(r["L_listed"] - L) > 1:
                    wrong_pairs[(r["frm"].replace(SUFFIX, ""), r["to"].replace(SUFFIX, ""), direction,
                                 r["L_listed"], L)] += 1
            else:
                L = r["L_listed"]
                checks["runs_without_chainage"] += 1
            Ls.append(L)
        if any(L is None for L in Ls) or any(r["T_s"] is None for r in runs):
            checks["train_days_dropped_incomplete"] += 1
            continue
        keep_a = []
        for r in runs:
            ok = r["T_a"] is not None and r["T_a"] > 0 and abs(r["T_a"] - r["T_s"]) <= MAX_DEV_MIN
            keep_a.append(ok)
            checks["runs"] += 1
            checks["runs_actual_time_kept" if ok else "runs_scheduled_time_used"] += 1
            if r["T_a"] is None:
                checks["runs_actual_time_missing"] += 1
        t = runs[0]["dep_a"] if keep_a[0] and runs[0]["dep_a"] is not None else runs[0]["dep_s"]
        for k, (r, L) in enumerate(zip(runs, Ls)):
            last = k == len(runs) - 1
            T = r["T_a"] if keep_a[k] else r["T_s"]
            if last:
                dw, dw_src = None, ""
            else:
                ok = (keep_a[k] and keep_a[k + 1] and r["dw_a"] is not None and 0 <= r["dw_a"] <= MAX_DWELL_MIN)
                dw, dw_src = (r["dw_a"], "act") if ok else (r["dw_s"], "sched")
                checks["dwells_actual_kept" if ok else "dwells_scheduled_used"] += 1
                if dw is None or dw < 0:
                    dw, dw_src = 0.0, "zero"
                    checks["dwells_set_to_zero"] += 1
            out_rows.append(dict(
                date=date, train=train, direction=direction, seq=k + 1, n_runs=len(runs),
                from_station=r["frm"].replace(SUFFIX, ""), to_station=r["to"].replace(SUFFIX, ""),
                L_km=round(L, 3), L_listed_km=r["L_listed"], T_run_min=T, T_src="act" if keep_a[k] else "sched",
                T_sched_min=r["T_s"], T_act_min=r["T_a"], dwell_min=dw, dwell_src=dw_src,
                dwell_sched_min=r["dw_s"], dwell_act_min=r["dw_a"], dep_min=t, arr_min=t + T,
                temperature_C=r["temp"], weather=r["weather"], holiday=r["holiday"]))
            t = t + T + (dw or 0.0)
        checks["train_days_kept"] += 1

    out = HERE / f"hsr_runs_{tag}.csv"
    with open(out, "w", newline="\n", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=list(out_rows[0]), lineterminator="\n")
        wr.writeheader()
        wr.writerows(out_rows)
    dates = sorted({r["date"] for r in out_rows})
    info = dict(
        corridor=f"{origin} <-> {terminus}", inputs={Path(op_path).name: md5(op_path), Path(mil_path).name: md5(mil_path)},
        dates=[dates[0], dates[-1], len(dates)],
        trains={d: len({r["train"] for r in out_rows if r["direction"] == d}) for d in ("down", "up")},
        line_km=line_km, line_km_frequency=dict((str(k), v) for k, v in sums.most_common(8)),
        chainage_train_days=n_chain_days,
        chainage_km={s.replace(SUFFIX, ""): round(v, 1) for s, v in sorted(chain.items(), key=lambda z: z[1])},
        chainage_spread_km={s.replace(SUFFIX, ""): round(v, 1) for s, v in chain_spread.items() if v > 0},
        listed_distance_differs={f"{a} -> {b} ({d}) listed {Ll:g} km, chainage {Lc:g} km": n
                                 for (a, b, d, Ll, Lc), n in wrong_pairs.most_common()},
        runs_with_listed_distance_differing=sum(wrong_pairs.values()),
        **checks)
    (HERE / f"hsr_runs_{tag}_checks.json").write_text(json.dumps(info, indent=1, ensure_ascii=False),
                                                         encoding="utf-8", newline="\n")
    print(json.dumps(info, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    if sys.argv[1] == "survey":
        survey(sys.argv[2])
    elif sys.argv[1] == "build":
        build(*sys.argv[2:7])
