"""External check: does the rebuilt calendar line up with real-world rates?

Compares the monthly mean of `euribor3m` in the dataset (using our rebuilt
year/month) with the European Central Bank's official monthly average of the
3-month Euribor. If the calendar were shifted by even one month, the 2008-09
rate collapse would make the two series diverge sharply.

    python verify/external_euribor.py  ->  verify/out/external.json
Uses the cached ECB file if present; pass --refresh to download it again.
"""
import csv
import io
import json
import sys
import urllib.request
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ECB_URL = ("https://data-api.ecb.europa.eu/service/data/FM/"
           "M.U2.EUR.RT.MM.EURIBOR3MD_.HSTA?startPeriod=2008-01&endPeriod=2010-12&format=csvdata")
CACHE = ROOT / "verify/external/ecb_euribor3m_monthly.csv"
MONTHS = "jan feb mar apr may jun jul aug sep oct nov dec".split()


def ecb_series(refresh):
    if refresh or not CACHE.exists():
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        text = urllib.request.urlopen(ECB_URL, timeout=30).read().decode()
        rows = [(r["TIME_PERIOD"], r["OBS_VALUE"]) for r in csv.DictReader(io.StringIO(text))]
        with open(CACHE, "w", newline="") as f:
            f.write(f"# source: {ECB_URL}\n# retrieved: {date.today().isoformat()}\n")
            w = csv.writer(f)
            w.writerow(["period", "euribor3m_monthly_avg"])
            w.writerows(rows)
    with open(CACHE) as f:
        lines = [l for l in f if not l.startswith("#")]
    return {r["period"]: float(r["euribor3m_monthly_avg"]) for r in csv.DictReader(lines)}


def dataset_monthly():
    with open(ROOT / "data/raw/bank-additional-full.csv", newline="") as f:
        rows = list(csv.DictReader(f, delimiter=";"))
    year, prev, acc = 2008, None, {}
    for row in rows:
        m = MONTHS.index(row["month"]) + 1
        if prev is not None and m < prev:
            year += 1
        prev = m
        acc.setdefault(f"{year}-{m:02d}", []).append(float(row["euribor3m"]))
    return {k: sum(v) / len(v) for k, v in acc.items()}


def main():
    ecb = ecb_series("--refresh" in sys.argv)
    ds = dataset_monthly()
    table = [dict(period=k, dataset=round(v, 3), ecb=round(ecb[k], 3), diff=round(v - ecb[k], 3))
             for k, v in sorted(ds.items())]
    shifted = {}
    for shift in (-1, 1):  # what the error would be if our calendar were off by a month
        diffs = []
        for k, v in ds.items():
            y, m = map(int, k.split("-"))
            m2 = m + shift
            y2, m2 = (y + (m2 - 1) // 12, (m2 - 1) % 12 + 1)
            key = f"{y2}-{m2:02d}"
            if key in ecb:
                diffs.append(abs(v - ecb[key]))
        shifted[str(shift)] = round(sum(diffs) / len(diffs), 3)
    mad = sum(abs(t["diff"]) for t in table) / len(table)
    out = dict(months=len(table), mean_abs_diff=round(mad, 3),
               max_abs_diff=max(abs(t["diff"]) for t in table),
               mean_abs_diff_if_shifted_by_month=shifted, table=table)
    (ROOT / "verify/out").mkdir(parents=True, exist_ok=True)
    (ROOT / "verify/out/external.json").write_text(json.dumps(out, indent=1))
    print(f"external: {len(table)} months, mean |dataset - ECB| = {mad:.3f} pp "
          f"(vs {shifted} if shifted one month)")


if __name__ == "__main__":
    main()
