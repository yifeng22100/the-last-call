"""Evaluation: walk-forward predictions, policy curves, bootstrap intervals.

Policy evaluation only ever *removes* calls the bank actually made, so every
outcome we count was observed. Nothing is extrapolated to calls that never
happened (the classic off-policy trap).
"""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from .models import SEED


# ----------------------------------------------------------------- predictions
def walk_forward(make_model, df, features, target, test_months, train_until=None):
    """Expanding-window forecast: for each month t, fit on months < t, predict t.

    If `train_until` is given the model is fitted once on months < train_until
    and frozen (never retrained) - used to show what drift does.
    """
    out = pd.Series(np.nan, index=df.index)
    frozen = None
    if train_until is not None:
        tr = df["month_idx"] < train_until
        frozen = make_model(features).fit(df.loc[tr, features], df.loc[tr, target])
    for t in test_months:
        te = df["month_idx"] == t
        if not te.any():
            continue
        model = frozen
        if model is None:
            tr = df["month_idx"] < t
            model = make_model(features).fit(df.loc[tr, features], df.loc[tr, target])
        out[te] = model.predict_proba(df.loc[te, features])[:, 1]
    return out


def random_split_auc(make_model, df, features, target, folds=5):
    """The usual (shuffled, time-blind) cross-validated AUC."""
    skf = StratifiedKFold(folds, shuffle=True, random_state=SEED)
    p = np.zeros(len(df))
    for tr, te in skf.split(df, df[target]):
        m = make_model(features).fit(df.iloc[tr][features], df.iloc[tr][target])
        p[te] = m.predict_proba(df.iloc[te][features])[:, 1]
    return roc_auc_score(df[target], p), p


# -------------------------------------------------------------- policy curves
def _cum_curve(score, calls, yes, weights=None):
    """Sort units by score (desc) and accumulate calls and yeses."""
    order = np.argsort(-score, kind="stable")
    w = np.ones(len(score)) if weights is None else weights[order]
    c = np.cumsum(calls[order] * w)
    y = np.cumsum(yes[order] * w)
    return score[order], c, y


def hazard_policy_curve(pp_test, p_col="p"):
    """Stop calling a client at the first call whose predicted hazard < t."""
    m = pp_test.groupby("client_id")[p_col].cummin().to_numpy()
    return _cum_curve(m, np.ones(len(m)), pp_test["event"].to_numpy())


def client_policy_curve(clients_test, s_col="s"):
    """Rank clients; calling a client means making all the bank's calls to them."""
    return _cum_curve(clients_test[s_col].to_numpy(),
                      clients_test["campaign"].to_numpy(),
                      clients_test["y"].to_numpy())


def cap_policy_points(clients_test, max_cap=15):
    """Rule of thumb: never call anyone more than N times."""
    camp, y = clients_test["campaign"].to_numpy(), clients_test["y"].to_numpy()
    pts = []
    for n in range(1, max_cap + 1):
        pts.append(dict(cap=n, calls=int(np.minimum(camp, n).sum()),
                        yes=int(((camp <= n) & (y == 1)).sum())))
    return pd.DataFrame(pts)


def yes_kept_at_budget(calls_cum, yes_cum, budget_share):
    """Share of all yeses kept when making `budget_share` of the bank's calls.

    Linear interpolation between ranked units (exact at unit boundaries).
    """
    target = budget_share * calls_cum[-1]
    return float(np.interp(target, np.r_[0, calls_cum], np.r_[0, yes_cum]) / yes_cum[-1])


def bootstrap_yes_kept(unit_client, score, calls, yes, budgets, reps=1000, seed=SEED):
    """Cluster bootstrap over clients (predictions held fixed).

    Captures sampling variability of the evaluation set, not of model fitting.
    """
    rng = np.random.default_rng(seed)
    codes, uniq = pd.factorize(unit_client)
    order = np.argsort(-score, kind="stable")
    codes, calls, yes = codes[order], calls[order], yes[order]
    res = np.empty((reps, len(budgets)))
    for r in range(reps):
        w_client = rng.multinomial(len(uniq), np.full(len(uniq), 1 / len(uniq)))
        w = w_client[codes]
        c, y = np.cumsum(calls * w), np.cumsum(yes * w)
        res[r] = [np.interp(b * c[-1], np.r_[0, c], np.r_[0, y]) / y[-1] for b in budgets]
    return np.percentile(res, [2.5, 97.5], axis=0)


def paired_bootstrap(clients, hazard_calls, client_scores, budget, reps=1000, seed=SEED):
    """Paired cluster bootstrap of yes-kept at one budget for three policies.

    Every replicate re-weights the *same* resampled clients for all policies,
    so differences between policies get proper confidence intervals.

    clients        DataFrame(client_id, campaign, y)
    hazard_calls   DataFrame(client_id, m, event)  - one row per call
    client_scores  array aligned with `clients`
    Returns dict policy -> array(reps) of yes-kept shares.
    """
    rng = np.random.default_rng(seed)
    ids = clients["client_id"].to_numpy()
    pos = pd.Series(np.arange(len(ids)), index=ids)
    hz_code = pos.loc[hazard_calls["client_id"]].to_numpy()
    camp, y = clients["campaign"].to_numpy(), clients["y"].to_numpy()

    hz_order = np.argsort(-hazard_calls["m"].to_numpy(), kind="stable")
    hz_code_o, hz_event_o = hz_code[hz_order], hazard_calls["event"].to_numpy()[hz_order]
    cl_order = np.argsort(-np.asarray(client_scores), kind="stable")
    max_cap = int(camp.max())
    capped_calls = np.minimum(camp[:, None], np.arange(1, max_cap + 1)[None, :])
    capped_yes = (y[:, None] == 1) & (camp[:, None] <= np.arange(1, max_cap + 1)[None, :])

    out = {k: np.empty(reps) for k in ("next_call", "client_ranking", "call_cap")}
    for r in range(reps):
        w = rng.multinomial(len(ids), np.full(len(ids), 1 / len(ids))).astype(float)
        # next-call policy
        wc = w[hz_code_o]
        c, yy = np.cumsum(wc), np.cumsum(hz_event_o * wc)
        out["next_call"][r] = np.interp(budget * c[-1], np.r_[0, c], np.r_[0, yy]) / yy[-1]
        # client ranking
        wo = w[cl_order]
        c, yy = np.cumsum(camp[cl_order] * wo), np.cumsum(y[cl_order] * wo)
        out["client_ranking"][r] = np.interp(budget * c[-1], np.r_[0, c], np.r_[0, yy]) / yy[-1]
        # call cap (random mixing of adjacent caps)
        cc, cy = w @ capped_calls, w @ capped_yes
        out["call_cap"][r] = np.interp(budget, np.r_[0, cc / cc[-1]], np.r_[0, cy / cy[-1]])
    return out


def _logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def _intercept_shift(p, observed_rate):
    """delta such that mean(sigmoid(logit(p) + delta)) == observed_rate (bisection)."""
    z = _logit(np.asarray(p, float))
    lo, hi = -20.0, 20.0
    for _ in range(100):
        mid = (lo + hi) / 2
        if np.mean(1 / (1 + np.exp(-(z + mid)))) < observed_rate:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


DRIFT_ALARM = np.log(3)  # last month's odds off by more than 3x -> do not act


def guarded_recalibration(calls, p_col, alarm=DRIFT_ALARM):
    """Deployment safeguards, applied month by month in time order.

    * Shadow mode: the first live month only scores calls (no call is cut).
    * Recalibration-in-the-large: month t's predictions are shifted on the
      logit scale so that last month's predictions would have matched last
      month's observed success rate (outcomes known before month t starts).
    * Drift alarm: if that shift exceeds `alarm` (odds off by > 3x), the model
      is not trusted in month t - back to shadow mode, all calls are made.
    Returns (p_recalibrated, active, log) aligned with `calls`.
    """
    months = sorted(calls["month_idx"].unique())
    p_out = calls[p_col].to_numpy(float).copy()
    active = np.zeros(len(calls), bool)
    log = {}
    for i, t in enumerate(months):
        idx = (calls["month_idx"] == t).to_numpy()
        if i == 0:
            log[int(t)] = dict(shift=None, status="shadow (first month)")
            continue
        prev = (calls["month_idx"] == months[i - 1]).to_numpy()
        delta = _intercept_shift(calls.loc[prev, p_col], calls.loc[prev, "event"].mean())
        if abs(delta) > alarm:
            log[int(t)] = dict(shift=float(delta), status="shadow (drift alarm)")
            continue
        log[int(t)] = dict(shift=float(delta), status="active")
        p_out[idx] = 1 / (1 + np.exp(-(_logit(p_out[idx]) + delta)))
        active[idx] = True
    return p_out, active, log


def threshold_policy(calls, p, active, threshold):
    """Calls made under 'stop at the first call with p < threshold' (inactive months: all calls)."""
    m = pd.Series(p, index=calls.index).groupby(calls["client_id"]).cummin().to_numpy()
    return (m >= threshold) | ~active


def downsample_curve(score, calls_cum, yes_cum, n=400):
    """Keep ~n evenly spaced points (by call share) for the web page."""
    share = calls_cum / calls_cum[-1]
    idx = np.unique(np.searchsorted(share, np.linspace(0, 1, n + 1)[1:], side="left"))
    idx = np.clip(idx, 0, len(share) - 1)
    return pd.DataFrame(dict(threshold=score[idx], calls=calls_cum[idx], yes=yes_cum[idx]))


def calibration_table(p, y, bins=10):
    q = pd.qcut(p, bins, duplicates="drop")
    g = pd.DataFrame(dict(p=p, y=y)).groupby(q, observed=True).agg(
        predicted=("p", "mean"), observed=("y", "mean"), n=("y", "size"))
    return g.reset_index(drop=True)
