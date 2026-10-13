# The Last Call

**When should a bank stop calling a client?**

By Edwin Tan · [GitHub](https://github.com/yifeng22100)

A next-call model, a stopping rule priced in money, deployment safeguards and a pilot plan, built on the UCI Bank Marketing data (41,188 clients, 105,754 calls, May 2008 to November 2010). It is tested month by month on data the model never saw, and every published figure is computed three independent ways.

**Live site:** https://yifeng22100.github.io/the-last-call/ (interactive simulator and AI analyst)

## What's different here

Most public analyses of this dataset train a classifier on a shuffled split and report accuracy or AUC. This project:

1. **Shows why that misleads.** It quantifies three traps: `duration` leakage, the "more calls convert worse" counting artifact, and a nearly fifteenfold shift in conversion that a shuffled split hides.
2. **Asks the operational question.** A discrete-time hazard model estimates the chance that the *next* call succeeds, so the output is a stopping rule: keep calling while P(yes) ≥ cost of a call ÷ value of a deposit.
3. **Tests it the way it would run.** Expanding-window walk-forward evaluation. Four candidate models were declared up front, the winner was chosen on 2009 only, and 2010 was held out.
4. **Reports what failed.** Gradient boosting broke under drift. The first post-crisis month exposed badly miscalibrated predictions, which led to three safeguards (shadow mode, monthly recalibration, a drift alarm). They were designed after seeing 2009 to 2010, which is disclosed.
5. **Proves every number.** A pandas/scikit-learn pipeline, a Python-standard-library re-implementation and an R re-implementation (including a `glm()` refit) must agree, and the rebuilt calendar is checked against ECB Euribor data. CI re-runs the check on every push.

Full numbers: [`docs/RESULTS.md`](docs/RESULTS.md) (generated, never hand-edited) · verification: [`verify/REPORT.md`](verify/REPORT.md) · method: [`docs/METHODOLOGY.md`](docs/METHODOLOGY.md) · pilot: [`docs/PILOT_PLAN.md`](docs/PILOT_PLAN.md) · model card: [`docs/MODEL_CARD.md`](docs/MODEL_CARD.md)

## Reproduce

Requires Python 3.12 and R ≥ 4.3 with `jsonlite`.

```bash
python3.12 -m venv .venv && .venv/bin/pip install -r requirements.txt
make all      # build (~single-threaded, deterministic), verify three ways, run tests
make serve    # http://localhost:8000
```

`make build` runs every experiment and writes `site/data/*.json` and `outputs/*.csv`. `make verify` recomputes the figures independently and fails on any mismatch. The build is single-threaded on purpose, so results are identical across machines.

## Repository layout

| Path | What it is |
|---|---|
| `lastcall/` | data loading and calendar rebuild, person-period expansion, models, evaluation, build |
| `verify/` | independent re-implementations (stdlib Python, R), ECB check, comparison, report |
| `tests/` | invariants: no leaked inputs, no future data in training, expansion correctness, page figures exist |
| `site/` | static website (no build step): simulator, charts, AI analyst |
| `data/raw/` | the official UCI file (SHA-256 checked on load) |
| `outputs/` | out-of-sample predictions used by the verifiers |

## The AI analyst

The site's analyst runs Claude (`claude-opus-5-5`) directly from the browser with the visitor's own Anthropic API key. The key goes only to `api.anthropic.com`, is kept in session storage (or local storage if the visitor opts in), and is never sent to any other server. The model is instructed to answer only from the verified `facts.json` and the visitor's simulator setting. Without a key, a rule-based briefing is generated locally.

## Data and citation

Moro, S., Rita, P., & Cortez, P. (2014). *Bank marketing* [Data set]. UCI Machine Learning Repository. https://doi.org/10.24432/C5K306. Licensed CC BY 4.0; redistributed unchanged in `data/raw/`.

Moro, S., Cortez, P., & Rita, P. (2014). A data-driven approach to predict the success of bank telemarketing. *Decision Support Systems, 62*, 22–31. https://doi.org/10.1016/j.dss.2014.03.001

Code: MIT License. Historical analysis for education and demonstration; not financial advice.
