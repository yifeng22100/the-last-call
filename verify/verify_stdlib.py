"""Independent re-computation #1: Python standard library only.

No pandas, numpy, scikit-learn or project code. Reads the raw CSV and the
exported out-of-sample predictions and recomputes every checked figure.

    python verify/verify_stdlib.py  ->  verify/out/stdlib.json
"""
import csv
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MONTHS = "jan feb mar apr may jun jul aug sep oct nov dec".split()


def read_raw():
    with open(ROOT / "data/raw/bank-additional-full.csv", newline="") as f:
        rows = list(csv.DictReader(f, delimiter=";"))
    year, prev = 2008, None
    for i, row in enumerate(rows):
        m = MONTHS.index(row["month"]) + 1
        if prev is not None and m < prev:
            year += 1
        prev = m
        row["id"], row["year"], row["ym"] = i, year, year * 100 + m
        row["yes"] = 1 if row["y"] == "yes" else 0
        row["campaign"] = int(row["campaign"])
    return rows


def auc(scores, labels):
    """Mann-Whitney U with average ranks for ties."""
    order = sorted(range(len(scores)), key=lambda i: scores[i])
    ranks = [0.0] * len(scores)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and scores[order[j + 1]] == scores[order[i]]:
            j += 1
        for k in range(i, j + 1):
            ranks[order[k]] = (i + j) / 2 + 1
        i = j + 1
    pos = sum(labels)
    neg = len(labels) - pos
    rank_sum = sum(rk for rk, l in zip(ranks, labels) if l == 1)
    return (rank_sum - pos * (pos + 1) / 2) / (pos * neg)


def interp(x, xs, ys):
    """Linear interpolation; xs increasing, starts at 0."""
    lo, hi = 0, len(xs) - 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if xs[mid] <= x:
            lo = mid
        else:
            hi = mid
    if xs[hi] == xs[lo]:
        return ys[hi]
    if x >= xs[hi]:
        return ys[hi]
    return ys[lo] + (ys[hi] - ys[lo]) * (x - xs[lo]) / (xs[hi] - xs[lo])


def yes_kept(scores, calls, yes, budget):
    order = sorted(range(len(scores)), key=lambda i: -scores[i])  # stable
    cx, cy = [0.0], [0.0]
    for i in order:
        cx.append(cx[-1] + calls[i])
        cy.append(cy[-1] + yes[i])
    return interp(budget * cx[-1], cx, cy) / cy[-1]


def logit(p):
    p = min(max(p, 1e-6), 1 - 1e-6)
    return math.log(p / (1 - p))


def sigmoid(z):
    return 1 / (1 + math.exp(-z))


def guarded_checks(wc, ym, out):
    """Safeguards + money rule, re-implemented from the raw p_logistic column."""
    months = sorted({ym(c) for c in wc})
    by_month = {mth: [c for c in wc if ym(c) == mth] for mth in months}
    first = by_month[months[0]]
    out["regime_break_calls"] = len(first)
    out["regime_break_observed"] = sum(c["event"] for c in first) / len(first)
    out["regime_break_predicted"] = sum(float(c["p_logistic"]) for c in first) / len(first)

    p_g, active = {}, {}
    n_active = 0
    for i, mth in enumerate(months):
        rows_t = by_month[mth]
        if i == 0:
            shift = None
        else:
            prev = by_month[months[i - 1]]
            obs = sum(c["event"] for c in prev) / len(prev)
            zs = [logit(float(c["p_logistic"])) for c in prev]
            lo, hi = -20.0, 20.0
            for _ in range(100):  # bisection on the logit shift
                mid = (lo + hi) / 2
                if sum(sigmoid(z + mid) for z in zs) / len(zs) < obs:
                    lo = mid
                else:
                    hi = mid
            shift = (lo + hi) / 2
            if abs(shift) > math.log(3):
                shift = None  # drift alarm
        for c in rows_t:
            key = (c["client_id"], c["call_k"])
            active[key] = shift is not None
            p = float(c["p_logistic"])
            p_g[key] = p if shift is None else sigmoid(logit(p) + shift)
            n_active += shift is not None
    out["active_share_of_calls"] = n_active / len(wc)

    def policy(threshold):
        run, made = {}, []
        for c in wc:  # rows are in client, call order
            key = (c["client_id"], c["call_k"])
            run[c["client_id"]] = min(run.get(c["client_id"], p_g[key]), p_g[key])
            made.append(run[c["client_id"]] >= threshold or not active[key])
        return made

    total_yes = sum(c["event"] for c in wc)
    for ratio in (10, 20, 30, 50):
        made = policy(1 / ratio)
        n_calls = sum(made)
        n_yes = sum(c["event"] for c, k in zip(wc, made) if k)
        cut = [c["event"] for c, k in zip(wc, made) if not k]
        out[f"money_{ratio}_calls_made"] = n_calls
        out[f"money_{ratio}_yes_kept"] = n_yes
        out[f"money_{ratio}_cut_success_rate"] = sum(cut) / len(cut)
        out[f"money_{ratio}_gain"] = (len(wc) - n_calls) - ratio * (total_yes - n_yes)
    made = policy(0.05)
    out["example_rule_calls_share"] = sum(made) / len(made)
    out["example_rule_yes_kept"] = sum(c["event"] for c, k in zip(wc, made) if k) / total_yes


def main():
    rows = read_raw()
    out = {}
    n = len(rows)
    out["n_clients"] = n
    out["n_yes"] = sum(r["yes"] for r in rows)
    out["total_calls"] = sum(r["campaign"] for r in rows)
    out["n_months"] = len({r["ym"] for r in rows})
    for y in (2008, 2009, 2010):
        sub = [r for r in rows if r["year"] == y]
        out[f"year_{y}_clients"] = len(sub)
        out[f"year_{y}_yes"] = sum(r["yes"] for r in sub)

    # success rate of each call, k = 1..9 and pooled 10+
    for k in range(1, 10):
        out[f"call_{k}_made"] = sum(1 for r in rows if r["campaign"] >= k)
        out[f"call_{k}_yes"] = sum(1 for r in rows if r["campaign"] == k and r["yes"])
    out["call_10+_made"] = sum(max(r["campaign"] - 9, 0) for r in rows)
    out["call_10+_yes"] = sum(1 for r in rows if r["campaign"] >= 10 and r["yes"])
    out["call2_vs_call1"] = (out["call_2_yes"] / out["call_2_made"]) / (out["call_1_yes"] / out["call_1_made"])

    # naive view: conversion grouped by total calls (10 = "10+")
    for c in range(1, 11):
        sub = [r for r in rows if min(r["campaign"], 10) == c]
        out[f"naive_{c}_clients"] = len(sub)
        out[f"naive_{c}_conversion"] = sum(r["yes"] for r in sub) / len(sub)

    # chronological eighths (numpy.array_split convention: first n % 8 get +1)
    size, extra, start = n // 8, n % 8, 0
    for b in range(8):
        end = start + size + (1 if b < extra else 0)
        out[f"eighth_{b + 1}_conversion"] = sum(r["yes"] for r in rows[start:end]) / (end - start)
        start = end

    # evaluation windows
    wins = {"train_initial": (200805, 200812), "validation": (200903, 200912),
            "holdout": (201003, 201011), "out_of_sample": (200903, 201011)}
    for w, (a, b) in wins.items():
        sub = [r for r in rows if a <= r["ym"] <= b]
        out[f"window_{w}_clients"] = len(sub)
        out[f"window_{w}_yes"] = sum(r["yes"] for r in sub)
        out[f"window_{w}_calls"] = sum(r["campaign"] for r in sub)

    # ------------------------------------ metrics from exported predictions
    with open(ROOT / "outputs/oos_call_predictions.csv", newline="") as f:
        calls = list(csv.DictReader(f))
    with open(ROOT / "outputs/oos_client_predictions.csv", newline="") as f:
        clients = list(csv.DictReader(f))
    for c in calls:
        c["event"] = int(c["event"])
    candidates = [k[2:] for k in calls[0] if k.startswith("p_")]
    ym = lambda rec: rows[int(rec["client_id"])]["ym"]  # our own calendar, not month_idx
    win_filter = {"out_of_sample": lambda v: v >= 200903,
                  "validation": lambda v: 200903 <= v <= 200912,
                  "holdout": lambda v: v >= 201003}
    for w, keep in win_filter.items():
        wc = [c for c in calls if keep(ym(c))]
        for cand in candidates:
            out[f"{w}_auc_{cand}"] = auc([float(c[f"p_{cand}"]) for c in wc], [c["event"] for c in wc])
        # running minimum of the selected model's prediction, per client
        run, m = {}, []
        for c in wc:
            p = float(c["p_logistic"])
            run[c["client_id"]] = min(run.get(c["client_id"], p), p)
            m.append(run[c["client_id"]])
        wl = [c for c in clients if keep(ym(c))]
        camp = [int(c["campaign"]) for c in wl]
        yy = [int(c["y"]) for c in wl]
        for bu in (0.25, 0.5, 0.75):
            out[f"{w}_next_call_yes_kept_{bu}"] = yes_kept(m, [1] * len(m), [c["event"] for c in wc], bu)
            out[f"{w}_client_ranking_logistic_yes_kept_{bu}"] = yes_kept(
                [float(c["s_logistic"]) for c in wl], camp, yy, bu)
            # call cap: points for cap N = 1..max, interpolated
            tot_c, tot_y = sum(camp), sum(yy)
            xs, ys = [0.0], [0.0]
            for cap in range(1, max(camp) + 1):
                xs.append(sum(min(c, cap) for c in camp) / tot_c)
                ys.append(sum(1 for c, v in zip(camp, yy) if v and c <= cap) / tot_y)
            out[f"{w}_call_cap_yes_kept_{bu}"] = interp(bu, xs, ys)
        if w == "out_of_sample":
            guarded_checks(wc, ym, out)

    (ROOT / "verify/out").mkdir(parents=True, exist_ok=True)
    (ROOT / "verify/out/stdlib.json").write_text(json.dumps(out, indent=1))
    print(f"stdlib: {len(out)} figures written")


if __name__ == "__main__":
    main()
