import streamlit as st
import joblib
import pandas as pd

from utils.win_probability import apply_ultra_predictive_adjustments

MODEL_PATH = "models/best_model.joblib"

@st.cache_data
def load_model():
    m = joblib.load(MODEL_PATH)
    return m["preprocessor"], m["calibrated_clf"]

@st.cache_data
def load_dropdown_data():
    deliveries = pd.read_csv("data/deliveries.csv")
    matches = pd.read_csv("data/matches.csv")

    teams = sorted(
        set(deliveries["batting_team"].dropna().unique()) |
        set(deliveries["bowling_team"].dropna().unique()) |
        set(matches["team1"].dropna().unique()) |
        set(matches["team2"].dropna().unique())
    )
    venues = sorted(
        set(deliveries["venue"].dropna().unique()) |
        set(matches["venue"].dropna().unique())
    )

    return teams, venues

pre, clf = load_model()
teams, venues = load_dropdown_data()

st.title("🏏 IPL Win Probability Predictor")

batting_team = st.selectbox("Batting Team", teams)
bowling_team = st.selectbox("Bowling Team", teams)
venue = st.selectbox("Venue", venues)

target = st.number_input("Target", value=160)
current_score = st.number_input("Current Score", value=100)
overs_completed = st.number_input("Overs Completed", value=12.0)
wickets_out = st.number_input("Wickets Out", min_value=0, max_value=10, value=3)

if st.button("Predict"):

    balls_elapsed = int(overs_completed * 6)
    balls_left = max(120 - balls_elapsed, 0)
    runs_left = target - current_score
    wickets_left = 10 - wickets_out

    crr = current_score / max(overs_completed, 0.1)
    rrr = (runs_left * 6) / max(balls_left, 1)

    # Hard rules
    if runs_left <= 0:
        batting_prob = 1.0
    elif wickets_left <= 0:
        batting_prob = 0.0
    elif balls_left <= 0:
        batting_prob = 0.0
    else:
        X = pd.DataFrame([{
            "batting_team": batting_team,
            "bowling_team": bowling_team,
            "venue": venue,
            "runs_left": runs_left,
            "balls_left": balls_left,
            "wickets_left": wickets_left,
            "crr": crr,
            "rrr": rrr,
            "target": target,
            "overs_completed": overs_completed,
        }])

        Xp = pre.transform(X)
        prob = clf.predict_proba(Xp)[0]
        base = prob[1]

        batting_prob = apply_ultra_predictive_adjustments(
            base, venue, wickets_left, runs_left, balls_left, crr, rrr, overs_completed
        )

    bowling_prob = 1 - batting_prob

    # --- Match Situation Output ---
    st.write("### Match Situation")
    st.write(f"**CRR:** {crr:.2f}")
    st.write(f"**RRR:** {rrr:.2f}")

    # --- Final Probabilities ---
    st.success(f"{batting_team} Win Probability: {batting_prob * 100:.2f}%")
    st.info(f"{bowling_team} Win Probability: {bowling_prob * 100:.2f}%")

    
