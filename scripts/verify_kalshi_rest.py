#!/usr/bin/env python3
"""
Verification script for Kalshi REST endpoints.

Tests:
1. Reaches Kalshi REST base URL.
2. Queries open events and markets.
3. Retrieves orderbook for active markets.
4. Validates parsing of modern orderbook_fp and legacy formats.
5. Verifies BBO (Best Bid & Offer) and binary YES/NO reflection.
"""

import sys
import os
import json

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.ingestion.kalshi_client import KalshiRestClient, KalshiClientError, DEFAULT_KALSHI_BASE_URL

def verify_kalshi_endpoints(base_url: str = DEFAULT_KALSHI_BASE_URL):
    print("=" * 65)
    print("KALSHI REST API ENDPOINT VERIFICATION")
    print("=" * 65)
    print(f"Target Base URL: {base_url}")
    
    client = KalshiRestClient(base_url=base_url)
    
    # 1. Test /events endpoint
    print("\n[Step 1] Querying /events (status=open)...")
    try:
        events_resp = client.get_events(status="open", limit=5)
        events = events_resp.get("events", [])
        print(f"  --> HTTP 200 OK. Found {len(events)} open events.")
        for ev in events[:3]:
            print(f"      * Event: {ev.get('event_ticker')} | {ev.get('title')}")
    except Exception as e:
        print(f"  [ERROR] /events failed: {e}")
        return False

    # 2. Test /markets endpoint
    print("\n[Step 2] Querying /markets (status=open)...")
    try:
        markets_resp = client.get_markets(status="open", limit=10)
        markets = markets_resp.get("markets", [])
        print(f"  --> HTTP 200 OK. Found {len(markets)} open markets.")
        if not markets:
            print("  [WARN] No open markets returned in initial batch.")
            return False
            
        print("  Sample markets:")
        for m in markets[:3]:
            print(f"      * Ticker: {m.get('ticker')} | Title: {m.get('title')} | Status: {m.get('status')}")
    except Exception as e:
        print(f"  [ERROR] /markets failed: {e}")
        return False

    # 3. Test /markets/{ticker}/orderbook endpoint on active candidates
    print("\n[Step 3] Querying orderbook for active market...")
    target_ticker = None
    # Prefer a market from the events we queried (often election or macro markets with liquidity)
    if events:
        event_ticker = events[0].get("event_ticker")
        try:
            ev_markets = client.get_markets(event_ticker=event_ticker)
            if ev_markets.get("markets"):
                target_ticker = ev_markets["markets"][0]["ticker"]
        except Exception:
            pass

    if not target_ticker and markets:
        target_ticker = markets[0]["ticker"]

    print(f"  Selected Ticker: {target_ticker}")
    try:
        raw_ob = client.get_raw_orderbook(target_ticker)
        print(f"  --> Raw orderbook response keys: {list(raw_ob.keys())}")
        
        parsed_ob = client.parse_orderbook_response(target_ticker, raw_ob)
        print(f"  --> Parsed orderbook successfully:")
        print(f"      * Total Bid Levels: {len(parsed_ob.bids)}")
        print(f"      * Total Ask Levels: {len(parsed_ob.asks)}")
        print(f"      * Best Bid (YES): {parsed_ob.best_bid}c")
        print(f"      * Best Ask (YES): {parsed_ob.best_ask}c")
        print(f"      * Spread: {parsed_ob.spread}c")
        print(f"      * Is Crossed: {parsed_ob.is_crossed}")
        print(f"      * Is One-Sided: {parsed_ob.is_one_sided}")

        if parsed_ob.bids:
            top_bid = parsed_ob.bids[0]
            print(f"      * Top Bid: {top_bid.quantity} @ {top_bid.price_cents}c (${top_bid.price_dollars:.2f})")
        if parsed_ob.asks:
            top_ask = parsed_ob.asks[0]
            print(f"      * Top Ask: {top_ask.quantity} @ {top_ask.price_cents}c (${top_ask.price_dollars:.2f})")
    except Exception as e:
        print(f"  [ERROR] /markets/{target_ticker}/orderbook failed: {e}")
        return False

    # 4. Synthetic validation of orderbook formats
    print("\n[Step 4] Validating dual schema compatibility (orderbook_fp & legacy)...")
    synthetic_fp = {
        "orderbook_fp": {
            "yes_dollars": [["0.4500", "100.00"], ["0.4400", "250.00"]],
            "no_dollars": [["0.5100", "80.00"], ["0.5000", "120.00"]]
        }
    }
    parsed_fp = client.parse_orderbook_response("TEST-FP", synthetic_fp)
    assert parsed_fp.best_bid == 45, f"Expected 45, got {parsed_fp.best_bid}"
    assert parsed_fp.best_ask == 49, f"Expected 49 (100 - 51), got {parsed_fp.best_ask}"
    assert parsed_fp.spread == 4, f"Expected 4, got {parsed_fp.spread}"
    print("  --> orderbook_fp parsing: PASS (reflected YES ask = 100 - NO bid)")

    synthetic_legacy = {
        "orderbook": {
            "yes": [[45, 100], [44, 250]],
            "no": [[51, 80], [50, 120]]
        }
    }
    parsed_legacy = client.parse_orderbook_response("TEST-LEGACY", synthetic_legacy)
    assert parsed_legacy.best_bid == 45
    assert parsed_legacy.best_ask == 49
    assert parsed_legacy.spread == 4
    print("  --> legacy orderbook parsing: PASS")

    print("\n" + "=" * 65)
    print("ALL KALSHI REST ENDPOINTS VERIFIED SUCCESSFULLY!")
    print("=" * 65)
    return True

if __name__ == "__main__":
    success = verify_kalshi_endpoints()
    sys.exit(0 if success else 1)
