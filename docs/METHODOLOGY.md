# Methodology

All figures referred to here are in [`RESULTS.md`](RESULTS.md), generated from `site/data/facts.json`.

## 1. Data and calendar

- Source: `bank-additional-full.csv` from the UCI Machine Learning Repository (Moro, Rita, & Cortez, 2014). SHA-256 `74adfc57…afb4d8`, byte-identical to the official release; checked on every load.
- The file is documented as ordered by date (May 2008 to November 2010) but has no year column. Rows form 26 contiguous runs of the same month; the year increments when the month name goes backwards. **External check:** the monthly mean of the dataset's `euribor3m` matches the European Central Bank's official monthly average of the 3-month Euribor to within a few hundredths of a percentage point on average, about five times closer than if the calendar were shifted by one month (`verify/external_euribor.py`).

## 2. What each row means, and the one-row-per-call expansion

Each row is a client's last contact in a campaign. `campaign` is the number of contacts in that campaign including the last. `y` says whether the client subscribed.

Assumption (stated, not testable in the data): **the bank stops calling once a client subscribes**, so a subscriber's success happened on call number `campaign`. Each client becomes `campaign` rows (calls k = 1…campaign) with `event = 1` only on a subscriber's last call. This is the standard discrete-time hazard set-up (Singer & Willett, 1993). The success rate of call k is

> h(k) = (subscribers with campaign = k) / (clients with campaign ≥ k).

## 3. Inputs: only what is known before the call

| Used | Excluded, and why |
|---|---|
| age, job, marital, education, default, housing, loan, contact channel | `duration`: only known after the call ends (leakage; Kaufman et al., 2012) |
| previous-campaign history: `previous`, `poutcome`, `pdays` (split into a "previously contacted" flag and a clipped value) | `campaign` as a client-level input: the total number of calls is only known once the campaign is over |
| economic indicators at the time of contact | `month`, `day_of_week`: timing of the *last* contact, not of each call |
| call number k (next-call model only) | |

Channel and client attributes are assumed constant across the calls of one campaign.

## 4. Evaluation protocol (fixed before any model choice)

- **Walk-forward, expanding window** (Tashman, 2000): to predict month t, fit on all months < t. The first predicted month is March 2009, so the first model sees exactly 2008.
- **Validation** March to December 2009 and **holdout** March to November 2010. The holdout was never used for any choice.
- **Candidates**, declared in `lastcall/models.py` before results: logistic regression and gradient boosting, each with and without economic indicators. Hyperparameters were fixed a priori; nothing was tuned.
- **Selection rule:** highest share of subscriptions kept at 50% of calls on validation; ties go to the simpler model.
- **Policies compared** on exactly the same clients: (a) next-call stopping rule (stop at the first call whose predicted chance is below a threshold); (b) client scoring (rank clients and make all of the bank's calls to the top ones); (c) a max-N-calls rule; (d) random.
- **Only observed outcomes are counted.** Every policy removes calls the bank actually made; none extrapolates to calls that never happened.
- **Uncertainty:** cluster bootstrap over clients, 1,000 resamples (Efron & Tibshirani, 1994), paired across policies for differences. Predictions are held fixed, so intervals reflect evaluation-sample variability, not model-fitting variability. Proportions use Wilson intervals (Wilson, 1927).

## 5. What failed, and what changed (disclosed)

1. **Gradient boosting under drift.** With economic indicators it came last on validation. Trees cannot extrapolate beyond the rates seen in training, and 2009 rates were far below 2008's. Without those inputs it performed like the others. This was found by the pre-declared protocol, not by tuning.
2. **Miscalibration after the regime break.** In March 2009 the selected model, trained on 2008, gave every call a predicted chance below 1%, while 17.8% succeeded. A money rule built on such predictions cuts the best calls first. **After seeing this**, three safeguards were added (`lastcall/evaluate.py: guarded_recalibration`):
   - *Shadow mode* in the first live month: scored, no calls cut.
   - *Recalibration-in-the-large* each month (Steyerberg, 2019): shift the logit so last month's predictions match last month's observed rate. Uses only outcomes known before the month starts.
   - *Drift alarm:* if that shift exceeds log(3) (odds off by more than 3×), the model is not trusted that month and the bank's default calling applies.

   Because these were designed after seeing 2009 to 2010, results for the guarded rule are not a clean out-of-sample test. The randomised pilot is.

## 6. Interpretable model

A logistic model on all 105,754 call rows, `event ~ call-number bucket + age group + contact + poutcome + euribor3m`, gives adjusted odds ratios for later calls relative to call 1. It is replicated with R's `glm()`; coefficients agree to about 1e-7 relative.

## 7. Verification

`make verify` recomputes every published figure (132) from the raw file and the exported predictions in two independent implementations (Python standard library only; R) and compares all three, plus the GLM coefficients and the ECB calendar check. A mutation test (deliberately corrupting one count and one share) confirmed the comparison fails as it should. `tests/` checks invariants: no excluded input reaches any model, walk-forward never trains on the month it predicts, the expansion reproduces the life table, and every figure shown on the website exists in the verified data.

## References

Efron, B., & Tibshirani, R. J. (1994). *An introduction to the bootstrap*. Chapman & Hall/CRC. https://doi.org/10.1201/9780429246593

Kaufman, S., Rosset, S., Perlich, C., & Stitelman, O. (2012). Leakage in data mining: Formulation, detection, and avoidance. *ACM Transactions on Knowledge Discovery from Data, 6*(4), Article 15. https://doi.org/10.1145/2382577.2382579

Moro, S., Rita, P., & Cortez, P. (2014). *Bank marketing* [Data set]. UCI Machine Learning Repository. https://doi.org/10.24432/C5K306

Singer, J. D., & Willett, J. B. (1993). It's about time: Using discrete-time survival analysis to study duration and the timing of events. *Journal of Educational Statistics, 18*(2), 155–195. https://doi.org/10.3102/10769986018002155

Steyerberg, E. W. (2019). *Clinical prediction models: A practical approach to development, validation, and updating* (2nd ed.). Springer. https://doi.org/10.1007/978-3-030-16399-0

Tashman, L. J. (2000). Out-of-sample tests of forecasting accuracy: An analysis and review. *International Journal of Forecasting, 16*(4), 437–450. https://doi.org/10.1016/S0169-2070(00)00065-0

Wilson, E. B. (1927). Probable inference, the law of succession, and statistical inference. *Journal of the American Statistical Association, 22*(158), 209–212. https://doi.org/10.1080/01621459.1927.10502953
