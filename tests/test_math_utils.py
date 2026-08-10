import pytest
import math
from app.engine.math_utils import (
    american_to_decimal,
    decimal_to_implied_prob,
    strip_vig_power_method,
    calculate_ev,
    kelly_criterion
)

def test_american_to_decimal():
    assert math.isclose(american_to_decimal(-110), 1.909090909, rel_tol=1e-7)
    assert american_to_decimal(150) == 2.5
    assert american_to_decimal(-200) == 1.5
    assert american_to_decimal(200) == 3.0
    # +100 and -100 are both even money and are the closest valid odds to zero.
    assert american_to_decimal(100) == 2.0
    assert american_to_decimal(-100) == 2.0

def test_american_to_decimal_rejects_the_impossible_gap():
    # No book quotes -99 or +42. Accepting them yields decimal odds below 1.0
    # and an implied probability above 1, which poisons the devig.
    for bad in (0, 1, 42, 99, -1, -42, -99):
        with pytest.raises(ValueError):
            american_to_decimal(bad)

def test_decimal_to_implied_prob():
    assert decimal_to_implied_prob(2.0) == 0.5
    assert math.isclose(decimal_to_implied_prob(1.909090909), 0.52380952, rel_tol=1e-7)
    with pytest.raises(ValueError):
        decimal_to_implied_prob(0.5)

def test_strip_vig_power_method_two_way():
    # Standard -110/-110 line
    prob_1 = decimal_to_implied_prob(american_to_decimal(-110))
    prob_2 = decimal_to_implied_prob(american_to_decimal(-110))
    
    fair_probs = strip_vig_power_method([prob_1, prob_2])
    
    assert len(fair_probs) == 2
    assert math.isclose(fair_probs[0], 0.5)
    assert math.isclose(fair_probs[1], 0.5)
    assert math.isclose(sum(fair_probs), 1.0)

def test_strip_vig_power_method_multi_way():
    # Example 3-way market (Soccer Home/Draw/Away)
    # Odds: 2.0, 3.5, 4.0
    probs = [1/2.0, 1/3.5, 1/4.0] # Sum: 0.5 + 0.2857 + 0.25 = 1.0357
    fair_probs = strip_vig_power_method(probs)
    
    assert len(fair_probs) == 3
    assert math.isclose(sum(fair_probs), 1.0, abs_tol=1e-9)
    # Larger odds should have proportionally less "vig impact" in power method
    assert fair_probs[0] < probs[0]
    assert fair_probs[1] < probs[1]
    assert fair_probs[2] < probs[2]

def test_calculate_ev():
    # If fair prob is 50% and we get 2.1 odds
    # EV = (0.5 * 1.1) - (0.5 * 1.0) = 0.55 - 0.5 = 0.05 (5%)
    assert math.isclose(calculate_ev(2.1, 0.5), 0.05)
    
    # If fair prob is 50% and we get 1.9 odds
    # EV = (0.5 * 0.9) - (0.5 * 1.0) = 0.45 - 0.5 = -0.05 (-5%)
    assert math.isclose(calculate_ev(1.9, 0.5), -0.05)

def test_kelly_criterion():
    # Odds 2.0 (+100), Fair Prob 55%
    # Kelly = (1.0 * 0.55 - 0.45) / 1.0 = 0.1
    # Fractional (0.1) = 0.01 (1% of bankroll)
    assert math.isclose(kelly_criterion(2.0, 0.55, 0.1), 0.01)
    
    # Negative EV should return 0
    assert kelly_criterion(1.9, 0.5, 0.1) == 0.0
