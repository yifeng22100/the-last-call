"""The website must only show figures that exist in the verified data files."""
import json
import re

from lastcall.data import ROOT

SITE = ROOT / "site"


def _get(obj, path):
    for key in path.split("/"):
        obj = obj[int(key)] if isinstance(obj, list) else obj[key]
    return obj


def test_every_bound_figure_exists():
    facts = json.loads((SITE / "data/facts.json").read_text())
    facts["verification"] = json.loads((SITE / "data/verification.json").read_text())
    html = (SITE / "index.html").read_text()
    paths = re.findall(r'data-fact="([^"]+)"', html)
    assert paths, "no bound figures found"
    for path in paths:
        value = _get(facts, path)
        assert value is not None, path


def test_verification_passed():
    v = json.loads((SITE / "data/verification.json").read_text())
    assert v["passed"] is True
    assert v["implementations"] == 3


def test_no_raw_percentages_in_page_copy():
    """Numbers in running text must come from data-fact bindings, not be typed in."""
    html = (SITE / "index.html").read_text()
    body = re.sub(r"<(script|style)[\s\S]*?</\1>", "", html)
    text = re.sub(r"<span[^>]*data-fact[^>]*></span>", "", body)
    text = re.sub(r"<ol class=\"refs\">[\s\S]*?</ol>", "", text)  # references carry page numbers
    stray = re.findall(r"\d+(?:\.\d+)?%", re.sub(r"<[^>]+>", " ", text))
    # Allowed: method settings, not findings (CI level, example thresholds).
    allowed = {"95%", "5%", "1%", "80%"}
    assert set(stray) <= allowed, f"hand-typed percentages: {stray}"
