# ============================================================
#   models/ml_model.py  –  Random Forest + Gradient Boosting
#                          trend-direction classifier
# ============================================================

import os, joblib
import numpy  as np
import pandas as pd
from sklearn.ensemble        import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model    import LogisticRegression
from sklearn.svm             import SVC
from sklearn.metrics         import (accuracy_score, classification_report,
                                     confusion_matrix, roc_auc_score)
from sklearn.model_selection import cross_val_score
from sklearn.pipeline        import Pipeline
from sklearn.preprocessing   import StandardScaler

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import RANDOM_STATE, MODELS_DIR, BUY_THRESHOLD, SELL_THRESHOLD


# ─────────────────────────────────────────────────────────
#  Model factory
# ─────────────────────────────────────────────────────────
def get_models() -> dict:
    return {
        "Random Forest": Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    RandomForestClassifier(
                n_estimators=200, max_depth=10,
                min_samples_split=10, class_weight="balanced",
                random_state=RANDOM_STATE, n_jobs=-1,
            )),
        ]),
        "Gradient Boosting": Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    GradientBoostingClassifier(
                n_estimators=200, learning_rate=0.05,
                max_depth=4, subsample=0.8,
                random_state=RANDOM_STATE,
            )),
        ]),
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    LogisticRegression(
                C=1.0, max_iter=1000, class_weight="balanced",
                random_state=RANDOM_STATE,
            )),
        ]),
        "SVM": Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    SVC(
                kernel="rbf", C=1.0, probability=True,
                class_weight="balanced", random_state=RANDOM_STATE,
            )),
        ]),
    }


# ─────────────────────────────────────────────────────────
#  Training
# ─────────────────────────────────────────────────────────
def train_models(
    X_train: np.ndarray | pd.DataFrame,
    y_train: np.ndarray | pd.Series,
    model_names: list[str] | None = None,
) -> dict:
    """Train all (or selected) models. Returns fitted pipeline dict."""
    models = get_models()
    if model_names:
        models = {k: v for k, v in models.items() if k in model_names}

    trained = {}
    for name, pipe in models.items():
        pipe.fit(X_train, y_train)
        trained[name] = pipe
        print(f"  [✓] Trained: {name}")
    return trained


# ─────────────────────────────────────────────────────────
#  Evaluation
# ─────────────────────────────────────────────────────────
def evaluate_models(
    trained_models: dict,
    X_test: np.ndarray | pd.DataFrame,
    y_test: np.ndarray | pd.Series,
) -> pd.DataFrame:
    """Return a DataFrame of per-model metrics."""
    records = []
    for name, pipe in trained_models.items():
        y_pred = pipe.predict(X_test)
        y_prob = pipe.predict_proba(X_test) if hasattr(pipe, "predict_proba") else None

        acc = accuracy_score(y_test, y_pred)
        try:
            auc = roc_auc_score(y_test, y_prob, multi_class="ovr", average="macro") if y_prob is not None else None
        except Exception:
            auc = None

        records.append({
            "Model":    name,
            "Accuracy": round(acc, 4),
            "AUC-ROC":  round(auc, 4) if auc else "N/A",
            "Report":   classification_report(y_test, y_pred, output_dict=True),
        })
        print(f"  {name}: Accuracy={acc:.4f}")

    return pd.DataFrame(records)


def get_best_model(eval_df: pd.DataFrame, trained_models: dict) -> tuple[str, object]:
    """Return (name, pipeline) of the model with the highest accuracy."""
    best_name = eval_df.loc[eval_df["Accuracy"].idxmax(), "Model"]
    return best_name, trained_models[best_name]


# ─────────────────────────────────────────────────────────
#  Prediction helpers
# ─────────────────────────────────────────────────────────
def predict_signal(
    model,
    X: np.ndarray | pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns (labels, probabilities).
    labels:  0=SELL  1=HOLD  2=BUY
    """
    labels = model.predict(X)
    probs  = model.predict_proba(X) if hasattr(model, "predict_proba") else None
    return labels, probs


def label_to_signal(label: int) -> str:
    return {0: "SELL", 1: "HOLD", 2: "BUY"}.get(int(label), "HOLD")


# ─────────────────────────────────────────────────────────
#  Persistence
# ─────────────────────────────────────────────────────────
def save_model(model, symbol: str, model_name: str):
    fname = f"{symbol}_{model_name.replace(' ', '_')}.pkl"
    path  = os.path.join(MODELS_DIR, fname)
    joblib.dump(model, path)
    print(f"  Model saved → {path}")


def load_model(symbol: str, model_name: str):
    fname = f"{symbol}_{model_name.replace(' ', '_')}.pkl"
    path  = os.path.join(MODELS_DIR, fname)
    if os.path.exists(path):
        return joblib.load(path)
    return None


# ─────────────────────────────────────────────────────────
#  Feature importance
# ─────────────────────────────────────────────────────────
def get_feature_importance(
    model,
    feature_names: list[str],
    top_n: int = 20,
) -> pd.DataFrame:
    """
    Extract feature importances from a fitted pipeline
    that wraps a tree-based estimator.
    """
    clf = model.named_steps.get("clf", model)
    if hasattr(clf, "feature_importances_"):
        fi = pd.DataFrame({
            "Feature":   feature_names,
            "Importance": clf.feature_importances_,
        }).sort_values("Importance", ascending=False).head(top_n)
        return fi
    elif hasattr(clf, "coef_"):
        fi = pd.DataFrame({
            "Feature":   feature_names,
            "Importance": np.abs(clf.coef_).mean(axis=0),
        }).sort_values("Importance", ascending=False).head(top_n)
        return fi
    return pd.DataFrame()
