"""Discrete-time hazard ("person-period") representation.

Each client row says: the bank called this client `campaign` times in the
current campaign, and the client either subscribed (y=1) or not. Assuming the
bank stops calling once a client subscribes, a subscriber said yes on call
number `campaign`, and every earlier call was a "not yet".

We expand each client into one row per call k = 1..campaign with
    event = 1  only on the last call of a subscriber, else 0.
The model then estimates h(k, x) = P(yes on call k | still being called at k, x),
the quantity a call-centre manager needs: "is the *next* call worth making?"
(Singer & Willett, 1993).
"""
import numpy as np
import pandas as pd


def expand(clients: pd.DataFrame) -> pd.DataFrame:
    """Return one row per call, with `call_k` (1-based) and `event`."""
    reps = clients["campaign"].to_numpy()
    pp = clients.loc[clients.index.repeat(reps)].copy()
    pp["call_k"] = pp.groupby("client_id").cumcount() + 1
    pp["event"] = ((pp["call_k"] == pp["campaign"]) & (pp["y"] == 1)).astype(int)
    return pp.reset_index(drop=True)


def life_table(clients: pd.DataFrame, max_k: int = 10) -> pd.DataFrame:
    """Empirical hazard by call number, with Wilson 95% intervals.

    Calls at or beyond `max_k` are pooled into a single "max_k+" row.
    """
    from statsmodels.stats.proportion import proportion_confint

    rows = []
    total_yes = int(clients["y"].sum())
    captured = 0
    for k in range(1, max_k + 1):
        if k < max_k:
            at_risk = int((clients["campaign"] >= k).sum())
            yes = int(((clients["campaign"] == k) & (clients["y"] == 1)).sum())
            label = str(k)
        else:  # pooled tail: every call made at k >= max_k
            at_risk = int((clients["campaign"] - (max_k - 1)).clip(lower=0).sum())
            yes = int(((clients["campaign"] >= k) & (clients["y"] == 1)).sum())
            label = f"{max_k}+"
        captured += yes
        lo, hi = proportion_confint(yes, at_risk, alpha=0.05, method="wilson")
        rows.append(dict(call=label, calls_made=at_risk, yes=yes,
                         hazard=yes / at_risk, ci_low=lo, ci_high=hi,
                         cum_share_of_yes=captured / total_yes))
    return pd.DataFrame(rows)


def naive_by_campaign(clients: pd.DataFrame, max_k: int = 10) -> pd.DataFrame:
    """The common (misleading) view: conversion grouped by total calls."""
    c = clients["campaign"].clip(upper=max_k)
    g = clients.groupby(c)["y"].agg(["mean", "size"]).reset_index()
    g.columns = ["campaign", "conversion", "clients"]
    g["campaign"] = g["campaign"].astype(str).where(g["campaign"] < max_k, f"{max_k}+")
    return g


def running_min(pp: pd.DataFrame, col: str = "p") -> np.ndarray:
    """Per client, min of predicted hazard over calls 1..k.

    Under the policy "stop at the first call whose predicted hazard is below
    threshold t", call k is made iff running_min_k >= t. This turns every
    stopping policy into a single sortable number per call.
    """
    return pp.groupby("client_id")[col].cummin().to_numpy()
