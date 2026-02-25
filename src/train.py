import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss
from sklearn.pipeline import Pipeline

from data_loader import load_raw, build_chase_dataset
from features import build_preprocessor


def evaluate(clf, X, y):
    """Return evaluation metrics."""
    p = clf.predict_proba(X)[:, 1]
    return {
        'auc': roc_auc_score(y, p),
        'logloss': log_loss(y, p),
        'brier': brier_score_loss(y, p)
    }


def main(matches_path, deliveries_path, out='models/best_model.joblib'):
    # Load data
    matches, deliveries = load_raw(matches_path, deliveries_path)
    df = build_chase_dataset(matches, deliveries)

    X = df.drop(columns=['label', 'match_id'])
    y = df['label']

    # Train/val split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, stratify=y, test_size=0.2, random_state=42
    )

    pre = build_preprocessor()

    # Models to compare
    models = {
        "logreg": LogisticRegression(max_iter=2000),
        "rf": RandomForestClassifier(n_estimators=200, random_state=42),
        "xgb": XGBClassifier(
            n_estimators=200,
            objective="binary:logistic",
            eval_metric="logloss",
            use_label_encoder=False
        )
    }

    best_auc = -1
    best_model = None

    # Training and evaluating
    for name, mdl in models.items():
        pipe = Pipeline([('pre', pre), ('clf', mdl)])
        pipe.fit(X_train, y_train)
        score = evaluate(pipe, X_val, y_val)
        print(name, score)

        if score['auc'] > best_auc:
            best_auc = score['auc']
            best_model = pipe

    # --- Probability Calibration (Fix for new sklearn) ---
    Xt = best_model.named_steps['pre'].transform(X_val)
    base = best_model.named_steps['clf']

    cal = CalibratedClassifierCV(
        estimator=base,    # FIX: use estimator= not base_estimator=
        cv='prefit',
        method='isotonic'
    )
    cal.fit(Xt, y_val)

    # Save model
    joblib.dump({
        'preprocessor': best_model.named_steps['pre'],
        'calibrated_clf': cal
    }, out)

    print(f"Model saved → {out}")


if __name__ == "__main__":
    main('data/matches.csv', 'data/deliveries.csv')
