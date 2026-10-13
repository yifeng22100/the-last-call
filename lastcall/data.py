"""Load the UCI Bank Marketing data and rebuild its calendar.

The file is documented as "ordered by date (from May 2008 to November 2010)"
but carries no year column. Rows form 26 contiguous runs of the same month;
the year increments every time the month name goes backwards (dec -> mar).
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_CSV = ROOT / "data" / "raw" / "bank-additional-full.csv"
# sha256 of the official UCI release (https://doi.org/10.24432/C5K306)
RAW_SHA256 = "74adfc578bf77a7ff4bb1ba4a9f8709d9e3c6907342959c2c8416847e0afb4d8"

MONTHS = ["jan", "feb", "mar", "apr", "may", "jun",
          "jul", "aug", "sep", "oct", "nov", "dec"]
FIRST_YEAR = 2008

# Client attributes and pre-campaign history: known before the first call.
CLIENT_CATEGORICAL = ["job", "marital", "education", "default", "housing",
                      "loan", "contact", "poutcome"]
CLIENT_NUMERIC = ["age", "previous", "prev_contacted", "pdays_clipped"]
# Monthly/quarterly economic context at the time of contact.
MACRO = ["emp.var.rate", "cons.price.idx", "cons.conf.idx", "euribor3m",
         "nr.employed"]
# Excluded on purpose (see docs/METHODOLOGY.md):
#   duration     - only known after the call ends (target leakage)
#   campaign     - total calls in the campaign, only known once it is over
#   month, day_of_week - timing of the *last* contact, not of each call
EXCLUDED = ["duration", "campaign", "month", "day_of_week"]


def sha256(path: Path) -> str:
    import hashlib
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path: Path = RAW_CSV, check_hash: bool = True) -> pd.DataFrame:
    if check_hash and sha256(path) != RAW_SHA256:
        raise ValueError(f"{path} does not match the official UCI file")
    d = pd.read_csv(path, sep=";")
    d.insert(0, "client_id", range(len(d)))
    d["y"] = (d["y"] == "yes").astype(int)

    month_num = d["month"].map({m: i + 1 for i, m in enumerate(MONTHS)})
    new_year = month_num < month_num.shift(fill_value=month_num.iloc[0])
    d["year"] = FIRST_YEAR + new_year.cumsum()
    d["period"] = pd.PeriodIndex.from_fields(year=d["year"], month=month_num, freq="M")
    # 0-based index of the calendar month in which the client was contacted
    d["month_idx"] = (d["period"] - d["period"].min()).apply(lambda x: x.n)

    # pdays == 999 means "never contacted before"; split it into a flag
    # and a clipped value so linear models are not dominated by 999.
    d["prev_contacted"] = (d["pdays"] != 999).astype(int)
    d["pdays_clipped"] = d["pdays"].where(d["pdays"] != 999, 30).clip(upper=30)
    return d
