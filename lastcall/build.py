"""Run every experiment and write the numbers the website and posts quote.

    python -m lastcall.build

Protocol (fixed before any result was inspected for model choice)
    train      all months before the month being predicted (expanding window)
    validate   2009-03 .. 2009-12  -> choose among CANDIDATES by yes-kept@50% calls
    holdout    2010-03 .. 2010-11  -> never used for any choice; reported as-is

Outputs
    site/data/facts.json      every figure quoted anywhere (single source of truth)
    site/data/curves.json     policy curves for the simulator
    site/data/monthly.json    month-by-month series
    outputs/*.csv             predictions and tables for independent verification
"""
import json
import time

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from sklearn.metrics import roc_auc_score
from statsmodels.stats.proportion import proportion_confint

from . import data, evaluate as ev, hazard
from .models import CANDIDATES, CLIENT_FEATURES, LEARNERS, SEED, SIMPLICITY

ROOT = data.ROOT
SITE = ROOT / "site" / "data"
OUT = ROOT / "outputs"
TEST_START = pd.Period("2009-03", "M")
HOLDOUT_START = pd.Period("2010-03", "M")
SELECTION_BUDGET = 0.50
BUDGETS = [0.25, 0.50, 0.75]
KEEP_TARGETS = [0.90, 0.95]
AGE_BINS = [0, 24, 60, 200]
AGE_LABELS = ["under 25", "25-60", "over 60"]
EXAMPLE_THRESHOLD = 0.05  # illustrative: a subscription is worth 20x the cost of a call
VALUE_TO_COST = [10, 20, 30, 50]  # illustrative value-of-a-subscription / cost-of-a-call ratios


def r(x, nd=4):
    return None if x is None or pd.isna(x) else round(float(x), nd)


def wilson(k, n):
    lo, hi = proportion_confint(k, n, alpha=0.05, method="wilson")
    return [r(lo), r(hi)]


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def policy_summary(unit_client, score, calls, yes, boot=True):
    """Yes kept at fixed call budgets, and calls needed to keep 90/95% of yeses."""
    _, c, y = ev._cum_curve(score, calls, yes)
    out = dict(yes_kept={str(b): r(ev.yes_kept_at_budget(c, y, b)) for b in BUDGETS},
               calls_needed={})
    for t in KEEP_TARGETS:
        i = int(np.searchsorted(y, t * y[-1], side="left"))
        out["calls_needed"][str(t)] = r(c[i] / c[-1])
    if boot:
        ci = ev.bootstrap_yes_kept(unit_client, score, calls.astype(float), yes.astype(float), BUDGETS)
        out["yes_kept_ci"] = {str(b): [r(ci[0, j]), r(ci[1, j])] for j, b in enumerate(BUDGETS)}
    return out


def hazard_summary(calls_df, p_col, boot=True):
    m = calls_df.groupby("client_id")[p_col].cummin().to_numpy()
    out = policy_summary(calls_df["client_id"].to_numpy(), m, np.ones(len(m)),
                         calls_df["event"].to_numpy(), boot)
    out["auc_next_call"] = r(roc_auc_score(calls_df["event"], calls_df[p_col]))
    out["mean_predicted"] = r(calls_df[p_col].mean())
    out["mean_observed"] = r(calls_df["event"].mean())
    return out


def main():
    SITE.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(exist_ok=True)
    facts = {}

    # ============================================================ descriptives
    d = data.load()
    log(f"loaded {len(d):,} clients")
    m_test = int(d.loc[d["period"] >= TEST_START, "month_idx"].min())
    m_hold = int(d.loc[d["period"] >= HOLDOUT_START, "month_idx"].min())
    test_months = sorted(d.loc[d["month_idx"] >= m_test, "month_idx"].unique())
    is_test = d["month_idx"] >= m_test

    n, yes = len(d), int(d["y"].sum())
    facts["dataset"] = dict(
        clients=n, subscribed=yes, conversion=r(yes / n), conversion_ci=wilson(yes, n),
        total_calls=int(d["campaign"].sum()),
        first_period=str(d["period"].min()), last_period=str(d["period"].max()),
        months_with_data=int(d["period"].nunique()), sha256=data.RAW_SHA256,
    )
    facts["windows"] = {}
    for name, mask in [("train_initial", d["month_idx"] < m_test),
                       ("validation", is_test & (d["month_idx"] < m_hold)),
                       ("holdout", d["month_idx"] >= m_hold),
                       ("out_of_sample", is_test)]:
        sub = d.loc[mask]
        facts["windows"][name] = dict(first=str(sub["period"].min()), last=str(sub["period"].max()),
                                      clients=int(len(sub)), subscribed=int(sub["y"].sum()),
                                      calls=int(sub["campaign"].sum()), conversion=r(sub["y"].mean()))

    by_year = d.groupby("year").agg(clients=("y", "size"), subscribed=("y", "sum"),
                                    euribor_mean=("euribor3m", "mean"))
    facts["by_year"] = {int(k): dict(clients=int(v.clients), subscribed=int(v.subscribed),
                                     conversion=r(v.subscribed / v.clients),
                                     conversion_ci=wilson(int(v.subscribed), int(v.clients)),
                                     euribor_mean=r(v.euribor_mean, 3))
                        for k, v in by_year.iterrows()}

    # Myth 2: "more calls = lower conversion" (naive) vs success rate of each call
    lt = hazard.life_table(d, max_k=10)
    naive = hazard.naive_by_campaign(d, max_k=10)
    lt.to_csv(OUT / "life_table.csv", index=False)
    naive.to_csv(OUT / "naive_by_campaign.csv", index=False)
    facts["per_call"] = [dict(call=row.call, calls_made=int(row.calls_made), yes=int(row.yes),
                              rate=r(row.hazard), ci=[r(row.ci_low), r(row.ci_high)],
                              cum_share_of_yes=r(row.cum_share_of_yes)) for row in lt.itertuples()]
    facts["naive_by_campaign"] = [dict(campaign=row.campaign, conversion=r(row.conversion),
                                       clients=int(row.clients)) for row in naive.itertuples()]
    facts["call2_vs_call1"] = r(lt.loc[1, "hazard"] / lt.loc[0, "hazard"], 3)

    # Drift
    monthly = d.groupby("period").agg(month_idx=("month_idx", "first"), clients=("y", "size"),
                                      subscribed=("y", "sum"), euribor=("euribor3m", "mean"),
                                      calls=("campaign", "sum"))
    monthly["conversion"] = monthly["subscribed"] / monthly["clients"]
    eighths = np.array_split(np.arange(n), 8)  # chronological eighths of the file
    conv8 = [d["y"].iloc[b].mean() for b in eighths]
    facts["drift"] = dict(
        eighths_conversion=[r(c) for c in conv8],
        eighths_euribor=[r(d["euribor3m"].iloc[b].mean(), 3) for b in eighths],
        ratio_last_to_first_eighth=r(conv8[-1] / conv8[0], 2),
        corr_monthly_euribor_conversion=r(np.corrcoef(monthly["euribor"], monthly["conversion"])[0, 1]),
    )

    # ============================================== Myth 1 & 3: leakage, drift
    gb = LEARNERS["gradient_boosting"]
    typical = ["age", "job", "marital", "education", "default", "housing", "loan", "contact",
               "month", "day_of_week", "duration", "campaign", "pdays", "previous", "poutcome",
               "emp.var.rate", "cons.price.idx", "cons.conf.idx", "euribor3m", "nr.employed"]
    log("random-split AUCs (leakage demo)")
    auc = dict(
        typical_all_inputs_random_split=ev.random_split_auc(gb, d, typical, "y")[0],
        without_duration_random_split=ev.random_split_auc(
            gb, d, [c for c in typical if c != "duration"], "y")[0],
        precall_random_split=ev.random_split_auc(gb, d, CLIENT_FEATURES, "y")[0],
    )
    log("walk-forward client models")
    for name, learner in LEARNERS.items():
        d[f"s_{name}"] = ev.walk_forward(learner, d, CLIENT_FEATURES, "y", test_months)
    dt = d.loc[is_test].copy()
    for name in LEARNERS:
        auc[f"precall_walk_forward_{name}"] = roc_auc_score(dt["y"], dt[f"s_{name}"])
        within = [roc_auc_score(g["y"], g[f"s_{name}"]) for _, g in dt.groupby("month_idx")]
        auc[f"precall_walk_forward_{name}_mean_within_month"] = np.mean(within)
    auc["precall_random_split_logistic"] = ev.random_split_auc(LEARNERS["logistic"], d, CLIENT_FEATURES, "y")[0]
    facts["auc"] = {k: r(v) for k, v in auc.items()}
    facts["auc"]["typical_inputs"] = typical
    facts["auc"]["precall_inputs"] = CLIENT_FEATURES

    log("frozen 2008 models")
    frozen = {}
    for name, learner in LEARNERS.items():
        d[f"f_{name}"] = ev.walk_forward(learner, d, CLIENT_FEATURES, "y", test_months, train_until=m_test)
    fz = d.loc[is_test].groupby("year")
    facts["frozen_model"] = {int(y): dict(actual=r(g["y"].mean()),
                                          **{f"frozen_{k}": r(g[f"f_{k}"].mean()) for k in LEARNERS},
                                          **{f"retrained_{k}": r(g[f"s_{k}"].mean()) for k in LEARNERS})
                             for y, g in fz}
    monthly = monthly.join(d.groupby("period")[[f"f_{k}" for k in LEARNERS] +
                                               [f"s_{k}" for k in LEARNERS]].mean())

    # ============================================ next-call (hazard) models
    pp = hazard.expand(d)
    calls = pp.loc[pp["month_idx"] >= m_test].copy()
    log(f"call rows: {len(pp):,} (out-of-sample {len(calls):,})")
    for name, (learner, feats) in CANDIDATES.items():
        log(f"walk-forward next-call model: {name}")
        calls[f"p_{name}"] = ev.walk_forward(learner, pp, feats, "event", test_months).loc[calls.index]
    val = calls["month_idx"] < m_hold
    hold = ~val

    selection = {name: hazard_summary(calls.loc[val], f"p_{name}", boot=False)
                 for name in CANDIDATES}
    best = max(SIMPLICITY, key=lambda k: (selection[k]["yes_kept"][str(SELECTION_BUDGET)],
                                          -SIMPLICITY.index(k)))
    log(f"selected on 2009 validation: {best}")
    facts["model_selection"] = dict(
        rule=f"highest share of yeses kept at {int(SELECTION_BUDGET * 100)}% of calls on the "
             f"validation window; ties -> simpler model",
        candidates={k: dict(features="pre-call + macro + call number" if "no_macro" not in k
                            else "pre-call + call number (no macro)", **selection[k])
                    for k in CANDIDATES},
        selected=best)
    facts["holdout"] = {name: hazard_summary(calls.loc[hold], f"p_{name}") for name in CANDIDATES}
    facts["out_of_sample"] = {name: hazard_summary(calls, f"p_{name}") for name in CANDIDATES}
    calls["p"] = calls[f"p_{best}"]
    calls["m"] = calls.groupby("client_id")["p"].cummin()

    # baselines on exactly the same clients, per window
    facts["baselines"] = {}
    windows = {"out_of_sample": dt, "validation": dt.loc[dt["month_idx"] < m_hold],
               "holdout": dt.loc[dt["month_idx"] >= m_hold]}
    for wname, w in windows.items():
        b = {}
        for name in LEARNERS:
            b[f"client_ranking_{name}"] = policy_summary(
                w["client_id"].to_numpy(), w[f"s_{name}"].to_numpy(),
                w["campaign"].to_numpy(), w["y"].to_numpy())
        cp = ev.cap_policy_points(w)
        wc, wy = cp["calls"] / w["campaign"].sum(), cp["yes"] / w["y"].sum()
        b["call_cap"] = dict(
            points={str(row.cap): dict(calls_share=r(c), yes_kept=r(y))
                    for row, c, y in zip(cp.itertuples(), wc, wy)},
            # mixing cap N and N+1 at random achieves any point on the segment
            yes_kept={str(bu): r(np.interp(bu, np.r_[0, wc], np.r_[0, wy])) for bu in BUDGETS})
        facts["baselines"][wname] = b
    log("paired bootstrap of policy differences")
    facts["paired_differences"] = {}
    for wname, w in windows.items():
        wcalls = calls.loc[calls["client_id"].isin(w["client_id"])]
        facts["paired_differences"][wname] = {}
        for bu in BUDGETS:
            bs = ev.paired_bootstrap(w, wcalls, w["s_logistic"].to_numpy(), bu)
            res = {}
            for other in ("client_ranking", "call_cap"):
                diff = bs["next_call"] - bs[other]
                res[f"next_call_minus_{other}"] = dict(
                    mean=r(diff.mean()), ci=[r(np.percentile(diff, 2.5)), r(np.percentile(diff, 97.5))],
                    share_of_replicates_positive=r((diff > 0).mean(), 3))
            facts["paired_differences"][wname][str(bu)] = res
    caps = ev.cap_policy_points(dt)
    tot_calls, tot_yes = int(dt["campaign"].sum()), int(dt["y"].sum())

    # calibration of the selected model (needed for the cost/value rule)
    cal = ev.calibration_table(calls["p"].to_numpy(), calls["event"].to_numpy())
    facts["calibration_deciles"] = [dict(predicted=r(a.predicted), observed=r(a.observed), n=int(a.n))
                                    for a in cal.itertuples()]

    # ============================== deployable policy: safeguards + money rule
    p_g, active, guard_log = ev.guarded_recalibration(calls, "p")
    calls["p_guarded"], calls["active"] = p_g, active
    facts["safeguards"] = dict(
        rule="shadow mode in the first live month; monthly recalibration-in-the-large; "
             "drift alarm (back to shadow) when last month's odds were off by more than 3x",
        alarm_log_odds=r(ev.DRIFT_ALARM),
        months={str(monthly.index[monthly["month_idx"] == k][0]): dict(
            status=v["status"], shift=r(v["shift"], 3)) for k, v in guard_log.items()},
        active_share_of_calls=r(active.mean()),
        shadow_share_of_calls=r(1 - active.mean()))
    first = calls.loc[calls["month_idx"] == test_months[0]]
    facts["regime_break"] = dict(
        period=str(TEST_START), calls=int(len(first)),
        observed_success_rate=r(first["event"].mean()),
        mean_predicted_unguarded=r(first["p"].mean()),
        share_predicted_below_1pct=r((first["p"] < 0.01).mean()))
    act = calls.loc[calls["active"]]
    cal_g = ev.calibration_table(act["p_guarded"].to_numpy(), act["event"].to_numpy())
    facts["calibration_guarded_deciles"] = [dict(predicted=r(a.predicted), observed=r(a.observed),
                                                 n=int(a.n)) for a in cal_g.itertuples()]
    tot_c_calls, tot_c_yes = len(calls), int(calls["event"].sum())
    money, grid_rows = {}, []
    grid = np.unique(np.r_[np.round(np.geomspace(0.002, 0.6, 240), 5), 0.1, 0.05, 1 / 30, 0.02])
    for t in grid:
        made = ev.threshold_policy(calls, calls["p_guarded"].to_numpy(), active, t)
        grid_rows.append((float(t), int(made.sum()), int(calls.loc[made, "event"].sum())))
    for ratio in VALUE_TO_COST:
        t = 1 / ratio
        made = ev.threshold_policy(calls, calls["p_guarded"].to_numpy(), active, t)
        n_calls, n_yes = int(made.sum()), int(calls.loc[made, "event"].sum())
        cut = calls.loc[~made]
        # profit change vs. the bank's actual calls, in units of one call's cost
        gain = (tot_c_calls - n_calls) - ratio * (tot_c_yes - n_yes)
        hb = max(grid_rows, key=lambda g: ratio * g[2] - g[1])  # hindsight best
        best_gain = (tot_c_calls - hb[1]) - ratio * (tot_c_yes - hb[2])
        money[str(ratio)] = dict(
            threshold=r(t), calls_made=n_calls, calls_share=r(n_calls / tot_c_calls),
            yes_kept=n_yes, yes_kept_share=r(n_yes / tot_c_yes),
            cut_calls=int(len(cut)), cut_calls_success_rate=r(cut["event"].mean()) if len(cut) else None,
            profit_gain_in_call_costs=r(gain, 1),
            profit_gain_per_1000_clients=r(gain / len(dt) * 1000, 1),
            hindsight_best_threshold=r(hb[0]), hindsight_best_gain=r(best_gain, 1))
    facts["money_rule"] = dict(
        note="Profit change vs. making every call the bank actually made, in units of the cost "
             "of one call; a subscription is worth `ratio` calls. Hindsight best = best "
             "threshold chosen after seeing outcomes (not achievable live; shown for reference).",
        totals=dict(calls=tot_c_calls, yes=tot_c_yes, clients=int(len(dt))),
        by_value_to_cost=money)

    # ======================================== fairness at one example threshold
    calls["made"] = ev.threshold_policy(calls, calls["p_guarded"].to_numpy(), active, EXAMPLE_THRESHOLD)
    pc = calls.groupby("client_id").agg(calls_made=("made", "sum"))
    pc["yes_kept"] = calls.loc[calls["made"]].groupby("client_id")["event"].sum().reindex(pc.index, fill_value=0)
    pc = pc.join(dt.set_index("client_id")[["age", "campaign", "y"]])
    pc["age_group"] = pd.cut(pc["age"], AGE_BINS, labels=AGE_LABELS)
    fair = pc.groupby("age_group", observed=True).agg(
        clients=("y", "size"), subscribed=("y", "sum"), yes_kept=("yes_kept", "sum"),
        calls_bank=("campaign", "sum"), calls_rule=("calls_made", "sum"),
        contacted=("calls_made", lambda x: int((x > 0).sum())))
    facts["example_rule"] = dict(
        threshold=EXAMPLE_THRESHOLD, model=best, policy="guarded (shadow + recalibration + drift alarm)",
        calls_share=r(calls["made"].mean()),
        yes_kept=r(calls.loc[calls["made"], "event"].sum() / calls["event"].sum()),
        by_age={g: dict(clients=int(v.clients), subscribed=int(v.subscribed),
                        yes_kept_share=r(v.yes_kept / v.subscribed),
                        calls_share=r(v.calls_rule / v.calls_bank),
                        contacted_share=r(v.contacted / v.clients))
                for g, v in fair.iterrows()})

    # ================================== interpretable GLM (replicated in R)
    log("interpretable GLM")
    g = pp.copy()
    g["call"] = pd.cut(g["call_k"], [0, 1, 2, 3, 4, 5, 9, 100],
                       labels=["c01", "c02", "c03", "c04", "c05", "c06_09", "c10plus"])
    g["age_group"] = pd.cut(g["age"], AGE_BINS, labels=["a_lt25", "a_25_60", "a_gt60"])
    g = g.rename(columns={"euribor3m": "euribor"})
    glm = smf.logit("event ~ C(call) + C(age_group, Treatment('a_25_60')) + C(contact) "
                    "+ C(poutcome) + euribor", data=g).fit(disp=0)
    ci = np.exp(glm.conf_int())
    facts["glm"] = dict(
        formula="event ~ call number + age group + contact + poutcome + euribor3m "
                "(logistic, one row per call, all 105,754 calls)",
        n_rows=int(glm.nobs),
        odds_ratios={k: dict(or_=r(np.exp(v)), ci=[r(ci.loc[k, 0]), r(ci.loc[k, 1])])
                     for k, v in glm.params.items()},
        coefficients={k: float(v) for k, v in glm.params.items()})
    g[["client_id", "event", "call", "age_group", "contact", "poutcome", "euribor"]].to_csv(
        OUT / "glm_input.csv", index=False)

    # ============================================================= write out
    curves = dict(totals=dict(calls=tot_calls, yes=tot_yes, clients=int(len(dt))))
    s, c, y = ev._cum_curve(calls["m"].to_numpy(), np.ones(len(calls)), calls["event"].to_numpy())
    curves["next_call_model"] = ev.downsample_curve(s, c, y).to_dict("list")
    for name in LEARNERS:
        s, c, y = ev._cum_curve(dt[f"s_{name}"].to_numpy(), dt["campaign"].to_numpy(), dt["y"].to_numpy())
        curves[f"client_ranking_{name}"] = ev.downsample_curve(s, c, y).to_dict("list")
    curves["call_cap"] = caps.to_dict("list")
    curves["model"] = best
    curves["guarded_threshold_grid"] = dict(threshold=[g[0] for g in grid_rows],
                                            calls=[g[1] for g in grid_rows], yes=[g[2] for g in grid_rows])

    calls[["client_id", "month_idx", "call_k", "event", "m", "p_guarded", "active"] +
          [f"p_{k}" for k in CANDIDATES]].to_csv(
        OUT / "oos_call_predictions.csv", index=False)
    dt[["client_id", "month_idx", "campaign", "y"] + [f"s_{k}" for k in LEARNERS]].to_csv(
        OUT / "oos_client_predictions.csv", index=False)
    mo = monthly.reset_index()
    mo["period"] = mo["period"].astype(str)
    (SITE / "monthly.json").write_text(mo.round(5).to_json(orient="records"))
    (SITE / "curves.json").write_text(json.dumps(curves, default=float))
    facts["protocol"] = dict(seed=SEED, test_start=str(TEST_START), holdout_start=str(HOLDOUT_START),
                             selection_budget=SELECTION_BUDGET, budgets=BUDGETS)
    (SITE / "facts.json").write_text(json.dumps(facts, indent=1, default=float))
    log("done")


if __name__ == "__main__":
    main()
