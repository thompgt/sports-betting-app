from typing import List
import math

# Smallest distance a probability is allowed to sit from 0 or 1 before a logit
# transform would blow up to +/- infinity.
_PROB_EPS = 1e-12

def american_to_decimal(odds: int) -> float:
    """
    Converts American odds to decimal odds.
    -110 -> 1.909...
    +150 -> 2.5
    """
    if odds == 0:
        raise ValueError("American odds cannot be zero.")
    
    if odds > 0:
        return (odds / 100) + 1
    else:
        return (100 / abs(odds)) + 1

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
    """
    overround = sum(probabilities)
    if overround < 1.0:
        # If probabilities are less than 1.0, they are already "fair" or undervalued
        return [p / overround for p in probabilities]
    
    if math.isclose(overround, 1.0, rel_tol=1e-12):
        return probabilities

    # Initial guess for k
    n = len(probabilities)
    # k = 1.0 is the identity. If overround > 1.0, k will be > 1.0
    k = 1.0 
    
    # Newton-Raphson to solve f(k) = sum(p_i^k) - 1 = 0
    for _ in range(max_iterations):
        # Clip k to prevent OverflowError in p**k
        # p is always <= 1.0 (implied prob), so p**k won't overflow for k > 0
        # However, if p is extremely small and k is small, it's fine.
        # If p is extremely small and k is large, p**k -> 0.
        k = max(1e-5, min(k, 1000.0))
        
        f_k = sum(p**k for p in probabilities) - 1.0
        f_prime_k = sum(p**k * math.log(p) for p in probabilities if p > 0)
        
        if abs(f_prime_k) < 1e-12:
            break
            
        step = f_k / f_prime_k
        new_k = k - step
        
        if abs(new_k - k) < tolerance:
            k = new_k
            break
        k = new_k

    return [p**k for p in probabilities]

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
