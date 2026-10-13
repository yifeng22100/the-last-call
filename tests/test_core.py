"""Invariant tests for the claims a reviewer is most likely to challenge."""
import numpy as np
import pandas as pd
import pytest

from lastcall import data, evaluate as ev, hazard
from lastcall.models import CANDIDATES, CLIENT_FEATURES, HAZARD_FEATURES


@pytest.fixture(scope="module")
def d():
    return data.load()


def test_raw_file_is_official_uci_release(d):
    assert data.sha256(data.RAW_CSV) == data.RAW_SHA256
    assert len(d) == 41188


def test_calendar_rebuild(d):
    assert str(d["period"].min()) == "2008-05" and str(d["period"].max()) == "2010-11"
    assert d["period"].nunique() == 26
    # file order is chronological: the period never goes backwards
    assert (d["period"].diff().dropna().apply(lambda x: x.n) >= 0).all()


def test_no_leaky_inputs():
    for leak in data.EXCLUDED:
        assert leak not in CLIENT_FEATURES
        assert leak not in HAZARD_FEATURES
        for _, feats in CANDIDATES.values():
            assert leak not in feats


def test_expansion_invariants(d):
    pp = hazard.expand(d)
    assert len(pp) == d["campaign"].sum()
    assert pp["event"].sum() == d["y"].sum()
    # at most one success per client, and only on the last call
    per = pp.groupby("client_id").agg(events=("event", "sum"), last=("call_k", "max"))
    assert per["events"].max() == 1
    succ = pp[pp["event"] == 1]
    assert (succ["call_k"] == succ["campaign"]).all()
    assert (per["last"].to_numpy() == d.set_index("client_id").loc[per.index, "campaign"].to_numpy()).all()


def test_life_table_matches_expansion(d):
    pp = hazard.expand(d)
    lt = hazard.life_table(d, max_k=10)
    for k in range(1, 10):
        rows = pp[pp["call_k"] == k]
        assert lt.loc[k - 1, "calls_made"] == len(rows)
        assert lt.loc[k - 1, "yes"] == rows["event"].sum()
    tail = pp[pp["call_k"] >= 10]
    assert lt.loc[9, "calls_made"] == len(tail) and lt.loc[9, "yes"] == tail["event"].sum()
    assert lt["cum_share_of_yes"].iloc[-1] == pytest.approx(1.0)


class _Spy:
    """Fake model that records which months it was trained on."""
    seen = []

    def __init__(self, features):
        pass

    def fit(self, X, y):
        _Spy.seen.append(set(X["month_idx"]))
        return self

    def predict_proba(self, X):
        return np.tile([0.5, 0.5], (len(X), 1))


def test_walk_forward_never_sees_the_future(d):
    _Spy.seen = []
    months = sorted(d["month_idx"].unique())[-5:]
    ev.walk_forward(_Spy, d, ["month_idx"], "y", months)
    for t, trained in zip(months, _Spy.seen):
        assert max(trained) < t


def test_stopping_policy_is_prefix_of_bank_calls():
    # client 1 has predicted hazards 0.3, 0.1, 0.4 -> threshold 0.2 makes call 1 only
    pp = pd.DataFrame(dict(client_id=[1, 1, 1, 2], p=[0.3, 0.1, 0.4, 0.05], event=[0, 0, 1, 0]))
    m = hazard.running_min(pp)
    assert list(m >= 0.2) == [True, False, False, False]


def test_yes_kept_interpolation():
    c, y = np.array([1, 2, 3, 4.0]), np.array([1, 1, 2, 2.0])
    assert ev.yes_kept_at_budget(c, y, 0.25) == 0.5
    assert ev.yes_kept_at_budget(c, y, 1.0) == 1.0
    assert ev.yes_kept_at_budget(c, y, 0.625) == pytest.approx(0.75)
