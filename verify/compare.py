"""Compare the three independent computations and fail on any mismatch.

    pipeline  site/data/facts.json  (pandas / scikit-learn / statsmodels)
    stdlib    verify/out/stdlib.json (Python standard library only)
    r         verify/out/r.json      (R, including glm refit)

Published figures are rounded (4 dp, ratios 3 dp), so pipeline vs. verifier
allows half a unit in the last published digit. stdlib vs. R must agree to
1e-9. GLM coefficients (statsmodels vs. R glm) must agree to 1e-6 relative.
Writes verify/REPORT.md.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GLM_NAMES = {
    "Intercept": "(Intercept)", "euribor": "euribor",
    "C(call)[T.c02]": "callc02", "C(call)[T.c03]": "callc03", "C(call)[T.c04]": "callc04",
    "C(call)[T.c05]": "callc05", "C(call)[T.c06_09]": "callc06_09", "C(call)[T.c10plus]": "callc10plus",
    "C(age_group, Treatment('a_25_60'))[T.a_lt25]": "age_groupa_lt25",
    "C(age_group, Treatment('a_25_60'))[T.a_gt60]": "age_groupa_gt60",
    "C(contact)[T.telephone]": "contacttelephone",
    "C(poutcome)[T.nonexistent]": "poutcomenonexistent", "C(poutcome)[T.success]": "poutcomesuccess",
}


def pipeline_flat(f):
    """Map facts.json onto the verifiers' flat keys -> (value, decimals)."""
    out = {}
    ds = f["dataset"]
    out.update(n_clients=(ds["clients"], 0), n_yes=(ds["subscribed"], 0),
               total_calls=(ds["total_calls"], 0), n_months=(ds["months_with_data"], 0))
    for y, v in f["by_year"].items():
        out[f"year_{y}_clients"] = (v["clients"], 0)
        out[f"year_{y}_yes"] = (v["subscribed"], 0)
    for row in f["per_call"]:
        out[f"call_{row['call']}_made"] = (row["calls_made"], 0)
        out[f"call_{row['call']}_yes"] = (row["yes"], 0)
    out["call2_vs_call1"] = (f["call2_vs_call1"], 3)
    for row in f["naive_by_campaign"]:
        c = "10" if row["campaign"] == "10+" else row["campaign"]
        out[f"naive_{c}_clients"] = (row["clients"], 0)
        out[f"naive_{c}_conversion"] = (row["conversion"], 4)
    for i, v in enumerate(f["drift"]["eighths_conversion"], 1):
        out[f"eighth_{i}_conversion"] = (v, 4)
    for w, v in f["windows"].items():
        out[f"window_{w}_clients"] = (v["clients"], 0)
        out[f"window_{w}_yes"] = (v["subscribed"], 0)
        out[f"window_{w}_calls"] = (v["calls"], 0)
    sel = f["model_selection"]["selected"]
    per_window = {"validation": f["model_selection"]["candidates"],
                  "holdout": f["holdout"], "out_of_sample": f["out_of_sample"]}
    for w, cands in per_window.items():
        for cand, v in cands.items():
            out[f"{w}_auc_{cand}"] = (v["auc_next_call"], 4)
        for bu, val in cands[sel]["yes_kept"].items():
            out[f"{w}_next_call_yes_kept_{bu}"] = (val, 4)
        base = f["baselines"][w]
        for bu, val in base["client_ranking_logistic"]["yes_kept"].items():
            out[f"{w}_client_ranking_logistic_yes_kept_{bu}"] = (val, 4)
        for bu, val in base["call_cap"]["yes_kept"].items():
            out[f"{w}_call_cap_yes_kept_{bu}"] = (val, 4)
    rb = f["regime_break"]
    out["regime_break_calls"] = (rb["calls"], 0)
    out["regime_break_observed"] = (rb["observed_success_rate"], 4)
    out["regime_break_predicted"] = (rb["mean_predicted_unguarded"], 4)
    out["active_share_of_calls"] = (f["safeguards"]["active_share_of_calls"], 4)
    for ratio, v in f["money_rule"]["by_value_to_cost"].items():
        out[f"money_{ratio}_calls_made"] = (v["calls_made"], 0)
        out[f"money_{ratio}_yes_kept"] = (v["yes_kept"], 0)
        out[f"money_{ratio}_cut_success_rate"] = (v["cut_calls_success_rate"], 4)
        out[f"money_{ratio}_gain"] = (v["profit_gain_in_call_costs"], 1)
    out["example_rule_calls_share"] = (f["example_rule"]["calls_share"], 4)
    out["example_rule_yes_kept"] = (f["example_rule"]["yes_kept"], 4)
    return out


def main():
    facts = json.loads((ROOT / "site/data/facts.json").read_text())
    std = json.loads((ROOT / "verify/out/stdlib.json").read_text())
    rr = json.loads((ROOT / "verify/out/r.json").read_text())
    ext = json.loads((ROOT / "verify/out/external.json").read_text())
    pipe = pipeline_flat(facts)

    failures, rows = [], []
    for key, (pv, dp) in sorted(pipe.items()):
        if key not in std or key not in rr:
            failures.append(f"{key}: missing in {'stdlib' if key not in std else 'R'}")
            continue
        tol = 0 if dp == 0 else 0.5 * 10 ** -dp + 1e-12
        d_ps, d_pr, d_sr = abs(pv - std[key]), abs(pv - rr[key]), abs(std[key] - rr[key])
        ok = d_ps <= tol and d_pr <= tol and d_sr <= 1e-9
        rows.append((key, pv, std[key], rr[key], ok))
        if not ok:
            failures.append(f"{key}: pipeline={pv} stdlib={std[key]} R={rr[key]}")

    glm_rows = []
    for py_name, r_name in GLM_NAMES.items():
        a, b = facts["glm"]["coefficients"][py_name], rr[f"glm_{r_name}"]
        rel = abs(a - b) / max(abs(b), 1e-12)
        glm_rows.append((r_name, a, b, rel))
        if rel > 1e-6:
            failures.append(f"GLM {r_name}: statsmodels={a} R={b}")
    if facts["glm"]["n_rows"] != rr["glm_n"]:
        failures.append("GLM row count differs")

    pilot = json.loads((ROOT / "site/data/pilot.json").read_text())
    pilot_rows = []
    for ratio, v in pilot["by_value_to_cost"].items():
        for name, py, rv, tol in [
            (f"pilot_{ratio}_a_n", v["a"]["n_per_arm"], rr[f"pilot_{ratio}_a_n"], 0),
            (f"pilot_{ratio}_b_n", v["b"]["flagged_calls_to_make"], rr[f"pilot_{ratio}_b_n"], 0),
            (f"pilot_{ratio}_b_power", v["b"]["exact_power"], rr[f"pilot_{ratio}_b_power"], 0.0005 + 1e-12)]:
            ok = abs(py - rv) <= tol
            pilot_rows.append((name, py, rv, ok))
            if not ok:
                failures.append(f"{name}: python={py} R={rv}")

    ext_ok = ext["mean_abs_diff"] < min(ext["mean_abs_diff_if_shifted_by_month"].values()) / 3
    if not ext_ok:
        failures.append("external Euribor check: calendar does not line up with ECB data")

    lines = ["# Verification report", "",
             "Generated by `verify/compare.py`. Every figure below was computed three times, "
             "independently:", "",
             "1. **Pipeline**: pandas / scikit-learn / statsmodels (`lastcall/build.py`), "
             "rounded as published",
             "2. **Python standard library only**: no pandas/numpy/project code "
             "(`verify/verify_stdlib.py`)",
             "3. **R**: base R written separately, incl. `glm()` refit (`verify/verify.R`)", "",
             f"**Result: {'ALL CHECKS PASSED' if not failures else 'FAILURES'}** · "
             f"{len(rows)} figures × 3 implementations · {len(glm_rows)} GLM coefficients "
             f"(statsmodels vs R) · {len(pilot_rows)} pilot sample-size figures (Python vs R) · 1 external calendar check", ""]
    if failures:
        lines += ["## Failures", ""] + [f"- {x}" for x in failures] + [""]
    lines += ["## External check: rebuilt calendar vs. ECB Euribor", "",
              f"Mean |dataset − ECB| over {ext['months']} months = **{ext['mean_abs_diff']} pp** "
              f"(would be {ext['mean_abs_diff_if_shifted_by_month']['-1']} / "
              f"{ext['mean_abs_diff_if_shifted_by_month']['1']} pp if the calendar were one month "
              f"early / late). Source: ECB Data Portal, series FM.M.U2.EUR.RT.MM.EURIBOR3MD_.HSTA.", "",
              "## Pilot sample sizes (lastcall/pilot.py vs R)", "", "| figure | Python | R | ok |", "|---|---|---|---|"]
    lines += [f"| `{n}` | {a} | {b} | {'✅' if ok else '❌'} |" for n, a, b, ok in pilot_rows]
    lines += ["", "## GLM coefficients (log-odds)", "", "| term | statsmodels | R glm | rel. diff |",
              "|---|---|---|---|"]
    lines += [f"| {n} | {a:.8f} | {b:.8f} | {rel:.1e} |" for n, a, b, rel in glm_rows]
    lines += ["", "## Figures", "", "| figure | pipeline (published) | stdlib | R | ok |",
              "|---|---|---|---|---|"]
    fmt = lambda v: f"{v:,}" if isinstance(v, int) else f"{v:.6f}"
    lines += [f"| `{k}` | {fmt(a)} | {fmt(b)} | {fmt(c)} | {'✅' if ok else '❌'} |"
              for k, a, b, c, ok in rows]
    (ROOT / "verify/REPORT.md").write_text("\n".join(lines) + "\n")
    summary = dict(passed=not failures, figures=len(rows), implementations=3,
                   glm_terms=len(glm_rows), pilot_figures=len(pilot_rows),
                   glm_max_rel_diff=max(g[3] for g in glm_rows),
                   external_mean_abs_diff=ext["mean_abs_diff"],
                   external_if_shifted=ext["mean_abs_diff_if_shifted_by_month"])
    (ROOT / "site/data/verification.json").write_text(json.dumps(summary, indent=1))
    print(f"compared {len(rows)} figures x3, {len(glm_rows)} GLM coefficients, external calendar check")
    if failures:
        print("FAILURES:\n  " + "\n  ".join(failures))
        sys.exit(1)
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
