"""
elo_calc.py — ELO difference and confidence interval calculator.

Uses the standard trinomial model (same as cutechess-cli, Ordo, BayesElo):

  1. Score percentage:  p = (W + D/2) / N
  2. ELO difference:    Δ = -400 * log10(1/p - 1)
  3. Score variance:     σ² = w*(1-p)² + d*(0.5-p)² + l*(0-p)²
                         where w,d,l are win/draw/loss frequencies
  4. Std error of mean:  SE(p) = √(σ²/N)
  5. Logistic derivative: dElo/dp = 400 / (ln(10) * p * (1-p))
  6. SE(ELO) = |dElo/dp| * SE(p)
  7. 95% CI  = ±1.96 * SE(ELO)

Also includes the pentanomial model for paired games (Fishtest-style),
which gives tighter confidence intervals when games are played in
color-swapped pairs from the same opening.

References:
  - cutechess-cli: Sprt.cpp
  - Ordo rating program by Miguel Ballicora
  - Fishtest pentanomial: https://github.com/glinscott/fishtest
"""

import math


def elo_difference(wins, draws, losses):
    """Calculate ELO difference from W/D/L using the logistic model.

    Returns:
        elo_diff (float): Estimated ELO difference (positive = player is stronger)
        ci_95 (float): 95% confidence interval half-width (±)
        se (float): Standard error of the ELO estimate
    """
    n = wins + draws + losses
    if n == 0:
        return 0.0, float('inf'), float('inf')

    # Score percentage
    p = (wins + draws * 0.5) / n

    # Edge cases
    if p <= 0.0:
        return -float('inf'), float('inf'), float('inf')
    if p >= 1.0:
        return float('inf'), float('inf'), float('inf')

    # ELO difference (logistic model)
    elo = -400.0 * math.log10(1.0 / p - 1.0)

    # Trinomial variance of a single game result
    w_freq = wins / n
    d_freq = draws / n
    l_freq = losses / n

    var = (w_freq * (1.0 - p) ** 2 +
           d_freq * (0.5 - p) ** 2 +
           l_freq * (0.0 - p) ** 2)

    # Standard error of the mean score
    se_p = math.sqrt(var / n)

    # Convert SE(p) to SE(ELO) via the derivative of the logistic function
    # ELO = -400 * log10(1/p - 1)
    # dELO/dp = 400 / (ln(10) * p * (1 - p))
    d_elo_dp = 400.0 / (math.log(10) * p * (1.0 - p))
    se_elo = abs(d_elo_dp) * se_p

    # 95% confidence interval
    ci_95 = 1.96 * se_elo

    return elo, ci_95, se_elo


def elo_difference_pentanomial(pair_results):
    """Calculate ELO with pentanomial model for paired games.

    When games are played in pairs (same opening, swapped colors),
    the pentanomial model gives tighter, more accurate CIs.

    Args:
        pair_results: list of (score_game1, score_game2) tuples
                      where each score is 1.0 (win), 0.5 (draw), 0.0 (loss)
                      from the perspective of the tested engine.

    Returns:
        elo_diff, ci_95, se
    """
    if not pair_results:
        return 0.0, float('inf'), float('inf')

    # Pentanomial bins: pair score can be 0, 0.5, 1.0, 1.5, 2.0
    bins = [0] * 5  # [0, 0.5, 1.0, 1.5, 2.0]
    for s1, s2 in pair_results:
        pair_score = s1 + s2
        idx = int(pair_score * 2)  # 0→0, 0.5→1, 1.0→2, 1.5→3, 2.0→4
        if 0 <= idx <= 4:
            bins[idx] += 1

    n_pairs = sum(bins)
    if n_pairs == 0:
        return 0.0, float('inf'), float('inf')

    # Mean pair score (out of 2)
    pair_values = [0.0, 0.5, 1.0, 1.5, 2.0]
    mu = sum(bins[i] * pair_values[i] for i in range(5)) / n_pairs

    # Score percentage (normalize to 0..1)
    p = mu / 2.0

    if p <= 0.0 or p >= 1.0:
        elo = 400.0 if p >= 1.0 else -400.0
        return elo, float('inf'), float('inf')

    # ELO
    elo = -400.0 * math.log10(1.0 / p - 1.0)

    # Pentanomial variance (variance of pair scores)
    var = sum(bins[i] * (pair_values[i] - mu) ** 2 for i in range(5)) / n_pairs

    # SE of mean pair score, then convert to per-game
    se_pair = math.sqrt(var / n_pairs)
    se_p = se_pair / 2.0  # pair score is out of 2, so divide by 2

    # Convert to ELO
    d_elo_dp = 400.0 / (math.log(10) * p * (1.0 - p))
    se_elo = abs(d_elo_dp) * se_p
    ci_95 = 1.96 * se_elo

    return elo, ci_95, se_elo


def elo_wdl_summary(wins, draws, losses, label=""):
    """Return a formatted summary string."""
    n = wins + draws + losses
    elo, ci, _ = elo_difference(wins, draws, losses)
    pct = (wins + draws * 0.5) / n * 100 if n > 0 else 0
    prefix = f"{label}: " if label else ""
    return (f"{prefix}+{wins} ={draws} -{losses} "
            f"({pct:.1f}%) | ELO: {elo:+.0f} ±{ci:.0f}")


def estimated_elo(wins, draws, losses, anchor_elo):
    """Estimate absolute ELO given results against an anchor of known ELO.

    Returns:
        estimated_elo, ci_95
    """
    elo_diff, ci, _ = elo_difference(wins, draws, losses)
    return anchor_elo + elo_diff, ci


# ── Self-test ─────────────────────────────────────────────────────
if __name__ == "__main__":
    # Test with known values
    print("=== Trinomial Model Tests ===")

    # 50% score → 0 ELO diff
    elo, ci, _ = elo_difference(50, 0, 50)
    print(f"50W 0D 50L: ELO={elo:+.1f} ±{ci:.1f}")

    # 55% score
    elo, ci, _ = elo_difference(55, 0, 45)
    print(f"55W 0D 45L: ELO={elo:+.1f} ±{ci:.1f}")

    # Typical engine match
    elo, ci, _ = elo_difference(59, 109, 32)
    print(f"59W 109D 32L: ELO={elo:+.1f} ±{ci:.1f}")

    # 600-game match
    elo, ci, _ = elo_difference(169, 296, 135)
    print(f"169W 296D 135L: ELO={elo:+.1f} ±{ci:.1f}")

    print("\n=== Estimated ELO vs SF-2800 ===")
    est, ci = estimated_elo(59, 85, 56, 2800)
    print(f"59W 85D 56L vs 2800: Est={est:.0f} ±{ci:.0f}")

    print("\n=== Summary Format ===")
    print(elo_wdl_summary(169, 296, 135, "v309 vs v306"))
