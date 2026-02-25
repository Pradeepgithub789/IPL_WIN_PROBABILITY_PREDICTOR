# utils/win_probability.py
import math

# ---------------------------------------------------------
# ULTRA-PREDICTIVE MULTIPLIER SYSTEM (FINAL + FLOOR LOGIC)
# ---------------------------------------------------------

def wicket_multiplier_fn(wickets_lost):
    """
    Wicket pressure curve (non-linear).
    Strong effect after wicket 5.
    """
    if wickets_lost <= 4:
        return 1.0
    table = {
        5: 0.88,
        6: 0.74,
        7: 0.55,
        8: 0.34,
        9: 0.18
    }
    return table.get(wickets_lost, 0.10)


def rrr_multiplier_fn(rrr, crr):
    """
    UPDATED RRR curve:
    Earlier version crushed probability too hard.
    Now softened so RRR 14–16 still leaves 10–20% chance.
    """

    if rrr <= crr:
        diff = crr - rrr
        boost = 1.0 + min(0.12, diff * 0.02)
        return boost

    diff = rrr - crr

    if diff < 2:
        return 0.95

    if diff < 4:
        return 0.80

    # softened from 0.60 → 0.65
    if diff < 6:
        return 0.65

    # softened from 0.40 → 0.55
    return 0.55


def venue_modifier_fn(venue_name):
    """
    Venue chase difficulty adjustment.
    Chepauk softened from -0.05 → -0.03.
    """
    table = {
        "Wankhede": 0.05,
        "Eden": 0.02,
        "Chinnaswamy": 0.03,
        "Chepauk": -0.03,
        "MA Chidambaram": -0.03,
        "Dubai": 0.02,
        "Sharjah": 0.03,
        "Ahmedabad": 0.01,
    }

    mod = 0.0
    low = venue_name.lower()

    for key, val in table.items():
        if key.lower() in low:
            mod = val
            break

    return 1.0 + mod


def overs_pressure_multiplier(balls_left, runs_left, overs_completed):
    """
    Death-over pressure logic.
    """
    overs_left = balls_left / 6.0

    # last over
    if overs_left <= 1:
        if runs_left > 10:
            return 0.45
        if runs_left > 5:
            return 0.65
        return 0.85

    # last 3 overs
    if overs_left <= 3:
        if runs_left > 20:
            return 0.60
        if runs_left > 10:
            return 0.75
        return 0.90

    # powerplay
    if overs_completed <= 6:
        return 0.92

    # middle overs
    if overs_completed <= 15:
        return 0.98

    return 0.95  # 15+ overs


def stage_multiplier(overs_completed, wickets_lost):
    """
    Stronger powerplay collapse logic.
    """

    # POWERPLAY (0–6 overs)
    if overs_completed <= 6:
        # heavy collapse
        if wickets_lost >= 4:
            return 0.40
        if wickets_lost == 3:
            return 0.65
        if wickets_lost == 2:
            return 0.85
        return 0.97

    # Middle overs (7–15)
    if overs_completed <= 15:
        return 0.99

    # Death overs (16–20)
    return 0.97





def apply_ultra_predictive_adjustments(base_prob, venue, wickets_left,
                                       runs_left, balls_left, crr, rrr,
                                       overs_completed):
    """
    Combine the model's base probability with realistic cricket logic
    and final floor logic to avoid impossible 0–5% situations.
    """

    wickets_lost = 10 - wickets_left

    # Get multipliers
    w_mult = wicket_multiplier_fn(wickets_lost)
    r_mult = rrr_multiplier_fn(rrr, crr)
    v_mult = venue_modifier_fn(venue)
    o_mult = overs_pressure_multiplier(balls_left, runs_left, overs_completed)
    s_mult = stage_multiplier(overs_completed, wickets_lost)

    # Apply multipliers to model's base prob
    adj = base_prob
    adj *= w_mult
    adj *= r_mult
    adj *= o_mult
    adj *= s_mult

    # Venue effect (soft additive)
    adj = adj + ((1 - adj) * (v_mult - 1))

    # ---------------------------------------------------------
    # FINAL SMART FLOOR LOGIC (PREVENTS stuck 0–5% values)
    # ---------------------------------------------------------

    if wickets_left >= 6:
        adj = max(adj, 0.25)  # strong batting position buffer
    elif wickets_left >= 4:
        adj = max(adj, 0.20)
    elif wickets_left >= 3:
        adj = max(adj, 0.15)
    elif wickets_left >= 2:
        adj = max(adj, 0.10)
    elif wickets_left == 1:
        adj = max(adj, 0.05)

    # clip
    adj = max(0.0, min(1.0, adj))

    return adj
