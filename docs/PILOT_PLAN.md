# Pilot plan

Goal: decide, with evidence from live calls, whether to cut calls that the model scores below break-even. Sample sizes come from `python -m lastcall.pilot` (`site/data/pilot.json`) and are re-computed independently in R (`verify/REPORT.md`).

## Why not a plain A/B test

Randomising clients to the old process or the rule and comparing net value per client is the obvious design. It needs very large samples unless calls are expensive relative to deposits, because the rule's gain per client is small next to the variation between clients. See the "A/B" column in `pilot.json`: it needs over a thousand clients per arm when a deposit is worth 10 calls, tens of thousands at 20 calls, and millions at 30 calls.

## Recommended design: flagged-call holdback

The rule flags a call when its predicted chance of success is below break-even, `cost of a call ÷ value of a deposit`. Cutting flagged calls pays off exactly when their true success rate is below break-even. So:

1. **Weeks 1 to 2: shadow mode.** Score every scheduled call; change nothing. Exit criteria: model scores arrive on time for at least 99% of calls, and the observed success rate of calls that *would* have been flagged is not above break-even.
2. **From week 3: live with holdback.** For each flagged call, a coin flip (50/50, logged before dialling) decides whether it is made anyway. Unflagged calls are made as usual.
   - **Primary test:** success rate of the flagged-and-made calls against break-even. One-sided, α = 5%, power 80%, planning value = the success rate of the calls the rule cut out-of-sample.
   - **Guardrails**, reviewed weekly: complaints and opt-outs per 1,000 clients; share of subscriptions captured by age group (under 25, 25 to 60, over 60), which must not fall more than 2 percentage points below the 25 to 60 group; model coverage.
   - **Stopping:** analyse once the planned number of flagged calls has been made. No early peeking at the primary test unless a pre-registered sequential boundary is used.
3. **Launch decision.** If the test rejects (flagged calls succeed below break-even), stop making flagged calls. Keep a 10% holdback permanently as monitoring.

## Production safeguards (kept after launch)

- Monthly recalibration-in-the-large from last month's outcomes.
- Drift alarm: if last month's odds were off by more than 3×, revert to shadow mode for the month and alert the analytics owner.
- Monthly retraining on an expanding window.

## Timeline at historical volumes

`pilot.json` converts the required number of flagged calls into months at the 2009 to 2010 average number of flagged calls per month (half of them made). A deposit worth 10 calls needs about half a month, 20 calls a few months, 30 calls about a year. Higher call volumes shorten this proportionally. With a one-month window and low margins, the pilot gives a provisional answer, and the permanent holdback completes the test after launch.

## Roles

| Role | Responsibility |
|---|---|
| Campaign manager | Sets cost and value assumptions; owns the launch decision |
| Analytics owner | Runs scoring, recalibration and the drift alarm; analyses the test |
| Compliance | Reviews guardrails, contact-frequency rules and the age check |
| Call-centre lead | Ensures coin flips are honoured and logged; reports complaints |
