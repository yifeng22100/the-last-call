# Model card: The Last Call next-call model

Format follows Mitchell et al. (2019). Figures are in [`RESULTS.md`](RESULTS.md).

## Model details

- **Type:** discrete-time hazard model. L2-regularised logistic regression (scikit-learn, C = 1.0) on one row per call.
- **Output:** probability that the next call to a client ends in a term-deposit subscription.
- **Inputs:** client attributes (age, job, marital status, education, default, housing loan, personal loan), contact channel, previous-campaign history, economic indicators at the time of contact, and call number. No call duration, total call count, or last-contact timing.
- **Training:** expanding window, retrained monthly. Selected among four pre-declared candidates on 2009 validation data.
- **Decision rule:** keep calling while the recalibrated probability ≥ cost of a call ÷ value of a deposit. Shadow mode in the first live month and whenever the drift alarm fires.

## Intended use

Deciding whether to make the next outbound call to a client in a term-deposit campaign, as decision support for campaign managers. Not for credit, pricing, or eligibility decisions, and not for any decision about a person beyond whether to place a marketing call.

## Evaluation

Walk-forward out-of-sample, March 2009 to November 2010. Holdout: March to November 2010. Metrics: share of subscriptions kept at a given share of calls; calibration by decile; money-rule gain against hindsight-best thresholds. Uncertainty: client-level cluster bootstrap. See `RESULTS.md`.

## Known limitations

- One Portuguese bank, 2008 to 2010, during a financial crisis; the conversion level shifted nearly fifteenfold over the period.
- The model's ranking edge over plain client scoring disappears in the 2010 holdout.
- Predictions are unreliable right after a regime break. The safeguards were designed after observing this in 2009, so their evaluation is not a clean out-of-sample test.
- Observational data: estimates what the next call is likely to yield, not the causal effect of calling.
- Assumes the bank stops calling once a client subscribes, and that client details and channel are constant within a campaign.

## Fairness

Age is the attribute most often treated as sensitive in this dataset (Le Quy et al., 2022). At a 5% threshold, clients over 60 keep a higher share of their subscriptions and calls than clients aged 25 to 60 (see `RESULTS.md`). The pilot plan monitors this by age group. Other protected characteristics are not in the data and cannot be checked.

## References

Le Quy, T., Roy, A., Iosifidis, V., Zhang, W., & Ntoutsi, E. (2022). A survey on datasets for fairness-aware machine learning. *WIREs Data Mining and Knowledge Discovery, 12*(3), Article e1452. https://doi.org/10.1002/widm.1452

Mitchell, M., Wu, S., Zaldivar, A., Barnes, P., Vasserman, L., Hutchinson, B., Spitzer, E., Raji, I. D., & Gebru, T. (2019). Model cards for model reporting. In *Proceedings of the Conference on Fairness, Accountability, and Transparency* (pp. 220–229). ACM. https://doi.org/10.1145/3287560.3287596
