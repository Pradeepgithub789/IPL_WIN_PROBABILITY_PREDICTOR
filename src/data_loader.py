import pandas as pd

DEL_MATCH_COL = 'match_no'         # deliveries.csv
MATCH_ID_COL = 'match_id'          # matches.csv
FIRST_INNS_COL = 'first_ings_score'
WINNER_COL = 'match_winner'

def load_raw(matches_path, deliveries_path):
    matches = pd.read_csv(matches_path)
    deliveries = pd.read_csv(deliveries_path)
    return matches, deliveries


def build_chase_dataset(matches, deliveries):
    matches = matches.copy()
    deliveries = deliveries.copy()

    # Target = first innings score + 1
    matches['target'] = matches[FIRST_INNS_COL] + 1

    # Only second innings
    d2 = deliveries[deliveries['innings'] == 2].copy()

    # Runs per ball
    d2['total_runs'] = d2['runs_of_bat'].fillna(0) + d2['extras'].fillna(0)

    # Sort balls
    sort_cols = [DEL_MATCH_COL, 'over']
    if 'ball' in d2.columns:
        sort_cols.append('ball')
    d2 = d2.sort_values(sort_cols).reset_index(drop=True)

    # Cumulative runs
    d2['cumulative_runs'] = d2.groupby(DEL_MATCH_COL)['total_runs'].cumsum()

    # Cumulative wickets
    d2['cumulative_wickets'] = (
        d2['player_dismissed']
        .notna()
        .groupby(d2[DEL_MATCH_COL])
        .transform('cumsum')
    )

    # Merge match target + winner ONLY
    d2 = d2.merge(
        matches[[MATCH_ID_COL, 'target', WINNER_COL]],
        left_on=DEL_MATCH_COL,
        right_on=MATCH_ID_COL,
        how='left'
    )

    # Label: 1 if batting team wins
    d2['label'] = (d2['batting_team'] == d2[WINNER_COL]).astype(int)

    # balls elapsed
    if 'ball' in d2.columns:
        d2['balls_elapsed'] = (d2['over'] - 1) * 6 + d2['ball']
    else:
        d2['ball_in_over'] = d2.groupby([DEL_MATCH_COL, 'over']).cumcount() + 1
        d2['balls_elapsed'] = (d2['over'] - 1) * 6 + d2['ball_in_over']

    # balls left
    d2['balls_left'] = 120 - d2['balls_elapsed']
    d2['balls_left'] = d2['balls_left'].clip(lower=0)

    # runs & wickets left
    d2['runs_left'] = d2['target'] - d2['cumulative_runs']
    d2['wickets_left'] = 10 - d2['cumulative_wickets']
    # NEW FEATURE — wicket pressure influences probability
    


    # overs completed
    ball_no = d2['balls_elapsed'] - (d2['over'] - 1) * 6
    d2['overs_completed'] = d2['over'] + (ball_no - 1) / 6.0
    d2['overs_completed'] = d2['overs_completed'].clip(lower=0.001)

    # run rates
    d2['crr'] = d2['cumulative_runs'] / d2['overs_completed']
    d2['rrr'] = d2.apply(lambda r: (r['runs_left'] * 6) / max(r['balls_left'], 1), axis=1)

    # ✔ venue is taken from deliveries.csv directly
    df = d2[[
        DEL_MATCH_COL,
        'batting_team',
        'bowling_team',
        'venue',                # directly from deliveries.csv (safe)
        'runs_left',
        'balls_left',
        'wickets_left',
        'crr',
        'rrr',
        'target',
        'overs_completed',
        'label'
    ]].rename(columns={DEL_MATCH_COL: 'match_id'})

    return df


if __name__ == "__main__":
    matches, deliveries = load_raw('../data/matches.csv', '../data/deliveries.csv')
    df = build_chase_dataset(matches, deliveries)
    print(df.head())
    print(df.columns)
