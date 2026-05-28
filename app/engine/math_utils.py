from typing import List
import math

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
    if overround <= 1.0:
        # No vig or already fair; return as is or normalize if exactly 1.0
        if math.isclose(overround, 1.0, rel_tol=1e-9):
            return probabilities
        raise ValueError(f"Total implied probability must be > 1.0 to devig (got {overround}).")

    # Initial guess for k
    k = 1.0
    if overround > 1.0:
        # Use log(n)/log(sum(p_i)) as a better starting point for k
        n = len(probabilities)
        k = math.log(n) / math.log(overround) if overround > 0 else 1.0

    # Newton-Raphson to solve f(k) = sum(p_i^k) - 1 = 0
    for _ in range(max_iterations):
        # Clip k to prevent OverflowError in p**k
        # Since p < 1, very large k -> 0, very small k -> infinity
        # Usually k is between 1 and 20 for sports markets.
        k = max(0.1, min(k, 100.0))
        
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
