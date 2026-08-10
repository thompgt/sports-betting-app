from typing import List
import math

# Smallest distance a probability is allowed to sit from 0 or 1 before a logit
# transform would blow up to +/- infinity.
_PROB_EPS = 1e-12

# Upper bound on the power-method exponent. A market needing more than this is
# too degenerate to devig meaningfully.
_MAX_POWER_EXPONENT = 1e6

# How far the devigged probabilities may sum from 1.0 before the solve is
# treated as failed rather than merely imprecise.
_DEVIG_RESIDUAL_TOLERANCE = 1e-6

def american_to_decimal(odds: float) -> float:
    """
    Converts American odds to decimal odds.
    -110 -> 1.909...
    +150 -> 2.5

    American odds have no representation strictly between -100 and +100: both
    +100 and -100 already mean an even-money bet. A feed emitting -99 or +42 is
    malformed, and accepting it would yield decimal odds below 1.0 and an
    implied probability above 1, which then poisons the devig for every other
    outcome in the market. Mirrors the guard in crates/edge-core/src/odds.rs.
    """
    if odds >= 100:
        return (odds / 100) + 1
    if odds <= -100:
        return (100 / abs(odds)) + 1
    raise ValueError(
        f"American odds must be >= +100 or <= -100, got {odds}."
    )

def decimal_to_implied_prob(decimal_odds: float) -> float:
    """Calculates implied probability from decimal odds."""
    if decimal_odds <= 1.0:
        raise ValueError("Decimal odds must be greater than 1.0.")
    return 1 / decimal_odds

def strip_vig_power_method(probabilities: List[float], max_iterations: int = 100, tolerance: float = 1e-10) -> List[float]:
    """
    Removes the bookmaker's overround (vig) using the Power Method.
    Solves for 'k' in: sum(p_i^k) = 1.0

    Reference: Shin/Power method for devigging multi-way markets.

    The solve is a *safeguarded* Newton: f(k) = sum(p_i^k) - 1 is strictly
    decreasing in k for probabilities in (0, 1), so the root can be bracketed
    and every Newton step that would leave the bracket - or fail to shrink it -
    is replaced by a bisection step. A bare Newton iteration can step to a wildly
    wrong exponent and, because the old implementation returned p**k whatever
    happened, that exponent became a "fair" price the service then traded
    against. This raises instead: no answer beats a confidently wrong one.
    """
    if not probabilities:
        raise ValueError("Cannot devig an empty market.")
    if any(not math.isfinite(p) or p <= 0.0 for p in probabilities):
        raise ValueError("Implied probabilities must be finite and strictly positive.")
    if any(p >= 1.0 for p in probabilities):
        # A single outcome already claiming certainty leaves no root to find.
        raise ValueError("Implied probabilities must be below 1.0 to devig.")

    overround = sum(probabilities)
    if overround <= 0.0:
        raise ValueError("Implied probabilities must sum to a positive number.")

    if math.isclose(overround, 1.0, rel_tol=1e-12):
        return list(probabilities)

    if overround < 1.0:
        # Underround: the book is quoting a sum below 1, so there is no margin to
        # strip. Scale proportionally rather than solving for an exponent < 1.
        return [p / overround for p in probabilities]

    def f(k: float) -> float:
        return sum(p ** k for p in probabilities) - 1.0

    # Bracket the root. f is strictly decreasing, and f(1) = overround - 1 > 0
    # here, so the root lies at some k > 1. Walk the upper bound out until f
    # turns negative.
    lo, hi = 1.0, 2.0
    f_lo = overround - 1.0
    f_hi = f(hi)
    while f_hi > 0.0:
        lo, f_lo = hi, f_hi
        hi *= 2.0
        if hi > _MAX_POWER_EXPONENT:
            raise ValueError(
                f"Could not bracket a devig exponent for overround {overround:.6f}; "
                "the market is too degenerate to devig."
            )
        f_hi = f(hi)

    k = lo
    for _ in range(max_iterations):
        f_k = f(k)
        if abs(f_k) < tolerance:
            break

        # Keep the bracket tight around the root at every step.
        if f_k > 0.0:
            lo = k
        else:
            hi = k

        derivative = sum(p ** k * math.log(p) for p in probabilities)
        if derivative != 0.0 and math.isfinite(derivative):
            candidate = k - f_k / derivative
        else:
            candidate = math.inf  # force the bisection fallback below

        # Reject a Newton step that leaves the bracket or stalls; bisect instead.
        if not (math.isfinite(candidate) and lo < candidate < hi):
            candidate = 0.5 * (lo + hi)

        if abs(candidate - k) < tolerance:
            k = candidate
            break
        k = candidate
    else:
        raise ValueError(
            f"Devig solver failed to converge in {max_iterations} iterations "
            f"(overround {overround:.6f}); refusing to return an unconverged fair price."
        )

    fair = [p ** k for p in probabilities]
    total = sum(fair)
    if not math.isfinite(total) or abs(total - 1.0) > _DEVIG_RESIDUAL_TOLERANCE:
        raise ValueError(
            f"Devig solution does not sum to 1 (got {total!r}); refusing to return it."
        )
    # Converged to within tolerance; divide out the last few ULPs so callers can
    # rely on the result being a genuine probability distribution.
    return [p / total for p in fair]

def pool_log_odds(fair_prob_vectors: List[List[float]]) -> List[float]:
    """
    Pools several sources' *already-devigged* probability vectors into one
    consensus, in log-odds (logit) space, then renormalises onto the simplex.

    Each element of `fair_prob_vectors` is one source's fair probabilities for
    the same outcomes, in the same order. Pooling must happen *after* devigging:
    averaging vigged probabilities mixes the books' margins into the consensus,
    and the power method then strips a margin that is the average of several
    different margins rather than any book's actual one.

    Log-odds is the right scale to average probabilities on - it is unbounded,
    symmetric about 0.5, and treats a move from 0.50 to 0.55 as smaller than a
    move from 0.90 to 0.95, which is how prices actually behave. Mirrors
    `consensus()` in crates/edge-core/src/consensus.rs.
    """
    if not fair_prob_vectors:
        raise ValueError("Cannot pool an empty set of sources.")

    n_outcomes = len(fair_prob_vectors[0])
    if n_outcomes == 0:
        raise ValueError("Cannot pool sources with no outcomes.")
    if any(len(v) != n_outcomes for v in fair_prob_vectors):
        raise ValueError("All sources must quote the same number of outcomes.")

    n_sources = len(fair_prob_vectors)
    pooled_logits: List[float] = []
    for i in range(n_outcomes):
        logits = []
        for vector in fair_prob_vectors:
            # Clamp strictly inside (0, 1) so a degenerate 0.0/1.0 quote does
            # not send the pooled logit to +/-inf and poison every outcome.
            p = min(max(vector[i], _PROB_EPS), 1.0 - _PROB_EPS)
            logits.append(math.log(p / (1.0 - p)))
        pooled_logits.append(sum(logits) / n_sources)

    raw = [1.0 / (1.0 + math.exp(-l)) for l in pooled_logits]
    total = sum(raw)
    if total <= 0.0:
        # Unreachable for clamped inputs, but a uniform prior beats a ZeroDivisionError.
        return [1.0 / n_outcomes] * n_outcomes

    # Per-outcome logit pooling does not preserve the simplex, so renormalise.
    return [p / total for p in raw]

def calculate_ev(bet_decimal_odds: float, fair_prob: float) -> float:
    """
    Calculates Expected Value (EV).
    EV = (Probability of Winning * Amount Won per $1) - (Probability of Losing * $1)
    """
    if not (0 <= fair_prob <= 1):
        raise ValueError("Fair probability must be between 0 and 1.")
    
    # Amount won per $1 (excluding stake) = decimal_odds - 1
    return (fair_prob * (bet_decimal_odds - 1)) - (1 - fair_prob)

def kelly_criterion(bet_decimal_odds: float, fair_prob: float, bankroll_fraction: float = 0.1) -> float:
    """
    Calculates the optimal fractional stake using the Kelly Criterion.
    f* = (bp - q) / b
    where:
    - b is the net decimal odds received (decimal_odds - 1)
    - p is the fair probability of winning
    - q is the probability of losing (1 - p)
    """
    if bet_decimal_odds <= 1.0:
        return 0.0
        
    b = bet_decimal_odds - 1
    p = fair_prob
    q = 1 - p
    
    kelly_fraction = (b * p - q) / b
    
    # Multiply by bankroll_fraction (e.g., "Quarter Kelly" = 0.25)
    # Return 0 if Kelly is negative (no value)
    return max(0.0, kelly_fraction * bankroll_fraction)
