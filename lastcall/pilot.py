"""Sample sizes for the pilot, from out-of-sample behaviour of the guarded rule.

Design A - client-level A/B test.
    Randomise clients to the bank's usual calling or the stopping rule.
    Metric: net value per client = V * subscribed - C * calls (in call costs).
    Sample size: normal approximation for a difference in means; checked by a
    Monte Carlo power simulation when the size is practical.

Design B - flagged-call holdback (recommended).
    The rule flags calls whose predicted chance is below break-even (C/V).
    A random half of flagged calls are made anyway. Cutting flagged calls pays
    off if their success rate is below C/V, so test H0: p >= C/V against
    H1: p < C/V (one-sided). Planning value for p: the success rate of the
    calls the rule cut out-of-sample. Sample size: normal approximation;
    checked with the exact binomial power at that size.

    python -m lastcall.pilot  ->  site/data/pilot.json
"""
import json

import numpy as np
import pandas as pd
from scipy.stats import binom, norm

from . import evaluate as ev
from .data import ROOT
from .models import SEED

ALPHA, POWER = 0.05, 0.80
RATIOS = [10, 20, 30]
SIM_LIMIT = 50_000  # simulate design A only when n per arm is practical


def flagged(calls, ratio):
    return ~ev.threshold_policy(calls, calls["p_guarded"].to_numpy(), calls["active"].to_numpy(), 1 / ratio)


def design_a(calls, ratio, rng):
    made = ~flagged(calls, ratio)
    g = calls.assign(made=made, kept=calls["event"] * made).groupby("client_id")
    c = (ratio * g["event"].sum() - g.size()).to_numpy(float)
    t = (ratio * g["kept"].sum() - g["made"].sum()).to_numpy(float)
    delta = t.mean() - c.mean()
    z = norm.ppf(1 - ALPHA / 2) + norm.ppf(POWER)
    n = int(np.ceil(z ** 2 * (c.var(ddof=1) + t.var(ddof=1)) / delta ** 2))
    sim = None
    if n <= SIM_LIMIT:
        hits = 0
        for _ in range(2000):
            a, b = rng.choice(c, n), rng.choice(t, n)
            se = np.sqrt(a.var(ddof=1) / n + b.var(ddof=1) / n)
            hits += abs(b.mean() - a.mean()) / se > norm.ppf(1 - ALPHA / 2)
        sim = round(hits / 2000, 3)
    return dict(effect_per_client=round(float(delta), 4), n_per_arm=n, simulated_power_at_n=sim)


def design_b(calls, ratio):
    f = flagged(calls, ratio)
    p0, p1 = 1 / ratio, float(calls.loc[f, "event"].mean())
    za, zb = norm.ppf(1 - ALPHA), norm.ppf(POWER)
    n_formula = int(np.ceil(((za * np.sqrt(p0 * (1 - p0)) + zb * np.sqrt(p1 * (1 - p1))) / (p0 - p1)) ** 2))

    def exact(n):
        """Exact one-sided binomial test: reject when successes <= k with P(X <= k | p0) <= alpha."""
        k = int(binom.ppf(ALPHA, n, p0))
        while k >= 0 and binom.cdf(k, n, p0) > ALPHA:
            k -= 1
        return float(binom.cdf(k, n, p1)), float(binom.cdf(k, n, p0))

    n = n_formula
    while exact(n)[0] < POWER:  # smallest n at or above the formula with exact power >= target
        n += 1
    power, alpha = exact(n)
    per_month = calls.loc[f].groupby("month_idx").size()  # months in which the rule flagged calls
    made_per_month = per_month.mean() / 2  # half of flagged calls are made
    return dict(break_even=round(p0, 4), planning_success_rate=round(p1, 4),
                flagged_calls_to_make_formula=n_formula, exact_power_at_formula_n=round(exact(n_formula)[0], 3),
                flagged_calls_to_make=n, exact_power=round(power, 3), exact_alpha=round(alpha, 4),
                flagged_calls_per_month_oos=round(float(per_month.mean()), 1),
                months_at_oos_volume=round(n / made_per_month, 1))


def main():
    calls = pd.read_csv(ROOT / "outputs" / "oos_call_predictions.csv")
    calls["active"] = calls["active"].astype(bool)
    rng = np.random.default_rng(SEED)
    out = dict(alpha=ALPHA, power=POWER, source="out-of-sample period 2009-03 to 2010-11, guarded rule",
               design_a="client-level A/B on net value per client (two-sided)",
               design_b="flagged-call holdback: success rate of flagged calls vs break-even (one-sided)",
               by_value_to_cost={})
    for ratio in RATIOS:
        out["by_value_to_cost"][str(ratio)] = dict(a=design_a(calls, ratio, rng), b=design_b(calls, ratio))
        print(ratio, out["by_value_to_cost"][str(ratio)])
    (ROOT / "site" / "data" / "pilot.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
