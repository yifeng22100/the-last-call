"""Model definitions. Hyperparameters are fixed a priori (no tuning on test data)."""
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from .data import CLIENT_CATEGORICAL, CLIENT_NUMERIC, MACRO

SEED = 20140301  # any fixed value; chosen once, never changed

CLIENT_FEATURES = CLIENT_CATEGORICAL + CLIENT_NUMERIC + MACRO
# Only used to reproduce the "typical notebook" setup in the leakage demo.
CATEGORICAL = CLIENT_CATEGORICAL + ["month", "day_of_week"]
HAZARD_FEATURES = CLIENT_FEATURES + ["call_k"]


def _numeric(features):
    return [f for f in features if f not in CATEGORICAL]


def gradient_boosting(features):
    """Primary model: histogram gradient boosting with native categoricals."""
    cats = [f for f in features if f in CATEGORICAL]
    prep = ColumnTransformer(
        [("cat", OrdinalEncoder(handle_unknown="use_encoded_value",
                                unknown_value=-1, encoded_missing_value=-1), cats),
         ("num", "passthrough", _numeric(features))],
        verbose_feature_names_out=False)
    clf = HistGradientBoostingClassifier(
        learning_rate=0.05, max_iter=300, max_leaf_nodes=15,
        min_samples_leaf=50, l2_regularization=1.0,
        categorical_features=list(range(len(cats))),
        early_stopping=False, random_state=SEED)
    return Pipeline([("prep", prep), ("clf", clf)])


def logistic(features):
    """Transparent baseline: L2 logistic regression on one-hot + scaled inputs."""
    cats = [f for f in features if f in CATEGORICAL]
    prep = ColumnTransformer(
        [("cat", OneHotEncoder(handle_unknown="ignore"), cats),
         ("num", StandardScaler(), _numeric(features))])
    return Pipeline([("prep", prep),
                     ("clf", LogisticRegression(C=1.0, max_iter=2000))])


LEARNERS = {"gradient_boosting": gradient_boosting, "logistic": logistic}

# Candidate next-call models, declared before looking at any test result.
# The winner is picked on the 2009 validation window only (see build.py).
HAZARD_NO_MACRO = [f for f in HAZARD_FEATURES if f not in MACRO]
CANDIDATES = {
    "logistic": (logistic, HAZARD_FEATURES),
    "gradient_boosting": (gradient_boosting, HAZARD_FEATURES),
    "logistic_no_macro": (logistic, HAZARD_NO_MACRO),
    "gradient_boosting_no_macro": (gradient_boosting, HAZARD_NO_MACRO),
}
# Order used to break ties: simpler first.
SIMPLICITY = ["logistic_no_macro", "logistic", "gradient_boosting_no_macro", "gradient_boosting"]
