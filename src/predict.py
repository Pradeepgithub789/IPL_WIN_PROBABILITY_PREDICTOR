import joblib
import pandas as pd

from utils.win_probability import apply_ultra_predictive_adjustments

MODEL_PATH = "models/best_model.joblib"

def predict_one(input_dict):

    m = joblib.load(MODEL_PATH)
    pre = m["preprocessor"]
    clf = m["calibrated_clf"]

    # derive variables
    runs_left = input_dict["target"] - input_dict["current_score"]
    balls_left = max(120 - int(input_dict["overs_completed"] * 6), 0)
    wickets_left = 10 - input_dict["wickets_out"]

    crr = input_dict["current_score"] / max(input_dict["overs_completed"], 0.1)
    rrr = (runs_left * 6) / max(balls_left, 1)

    # Hard checks
    if runs_left <= 0:
        return {"batting_team_win_prob": 100, "bowling_team_win_prob": 0}

    if wickets_left <= 0:
        return {"batting_team_win_prob": 0, "bowling_team_win_prob": 100}

    if balls_left <= 0:
        return {"batting_team_win_prob": 0, "bowling_team_win_prob": 100}

    X = pd.DataFrame([{
        "batting_team": input_dict["batting_team"],
        "bowling_team": input_dict["bowling_team"],
        "venue": input_dict["venue"],
        "runs_left": runs_left,
        "balls_left": balls_left,
        "wickets_left": wickets_left,
        "crr": crr,
        "rrr": rrr,
        "target": input_dict["target"],
        "overs_completed": input_dict["overs_completed"],
    }])

    Xp = pre.transform(X)
    base = clf.predict_proba(Xp)[0][1]

    final_prob = apply_ultra_predictive_adjustments(
        base,
        input_dict["venue"],
        wickets_left,
        runs_left,
        balls_left,
        crr,
        rrr,
        input_dict["overs_completed"]
    )

    return {
        "batting_team_win_prob": final_prob * 100,
        "bowling_team_win_prob": (1-final_prob)*100
    }
