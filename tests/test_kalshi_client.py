import json
import pytest
from unittest.mock import patch, MagicMock
from app.ingestion.kalshi_client import (
    KalshiRestClient,
    ParsedOrderbook,
    OrderbookLevel,
    KalshiHttpError,
    KalshiClientError,
    DEFAULT_KALSHI_BASE_URL
)

def test_client_init_and_url_handling():
    client = KalshiRestClient(base_url="https://api.elections.kalshi.com/trade-api/v2/")
    assert client.base_url == "https://api.elections.kalshi.com/trade-api/v2"
    assert client.min_interval > 0

def test_parse_modern_orderbook_fp():
    payload = {
        "orderbook_fp": {
            "yes_dollars": [["0.4500", "100.00"], ["0.4400", "250.00"], ["0.4300", "500.00"]],
            "no_dollars": [["0.5100", "80.00"], ["0.5000", "120.00"], ["0.4900", "300.00"]]
        }
    }
    ob = KalshiRestClient.parse_orderbook_response("TEST-TICKER", payload, ts=1700000000.0)
    assert ob.ticker == "TEST-TICKER"
    assert ob.timestamp == 1700000000.0
    assert len(ob.bids) == 3
    assert len(ob.asks) == 3

    # Bids: YES bids descending
    assert ob.best_bid == 45
    assert ob.bids[0].quantity == 100
    assert ob.bids[1].price_cents == 44
    assert ob.bids[2].price_cents == 43

    # Asks: reflected NO bids: NO 51c -> YES ask 49c, ascending
    assert ob.best_ask == 49
    assert ob.asks[0].quantity == 80
    assert ob.asks[1].price_cents == 50  # 100 - 50
    assert ob.asks[2].price_cents == 51  # 100 - 49

    assert ob.spread == 4
    assert not ob.is_crossed
    assert not ob.is_one_sided
    assert not ob.is_empty
    assert ob.mid_price == 47.0

def test_parse_legacy_orderbook():
    payload = {
        "orderbook": {
            "yes": [[45, 100], [44, 250]],
            "no": [[51, 80], [50, 120]]
        }
    }
    ob = KalshiRestClient.parse_orderbook_response("TEST-LEGACY", payload)
    assert ob.best_bid == 45
    assert ob.best_ask == 49
    assert ob.spread == 4
    assert len(ob.bids) == 2
    assert len(ob.asks) == 2

def test_parse_empty_orderbook():
    payload = {"orderbook_fp": {"yes_dollars": [], "no_dollars": []}}
    ob = KalshiRestClient.parse_orderbook_response("TEST-EMPTY", payload)
    assert ob.best_bid is None
    assert ob.best_ask is None
    assert ob.spread is None
    assert ob.is_empty
    assert not ob.is_one_sided

def test_parse_one_sided_orderbook():
    payload = {
        "orderbook_fp": {
            "yes_dollars": [["0.4500", "50.00"]],
            "no_dollars": []
        }
    }
    ob = KalshiRestClient.parse_orderbook_response("TEST-ONE-SIDED", payload)
    assert ob.best_bid == 45
    assert ob.best_ask is None
    assert ob.is_one_sided
    assert not ob.is_empty
    assert ob.mid_price is None

def test_malformed_and_out_of_range_levels_ignored():
    payload = {
        "orderbook_fp": {
            "yes_dollars": [
                ["0.0000", "100.00"],    # 0c is invalid limit price
                ["1.0000", "100.00"],    # 100c is invalid limit price
                ["invalid", "50.00"],    # bad float
                ["0.4500", "0.00"],      # zero size
                ["0.4500", "100.00"]     # valid
            ],
            "no_dollars": [
                ["0.5100", "50.00"],     # valid -> 49c ask
                ["0.0000", "20.00"]      # would reflect to 100c -> discarded
            ]
        }
    }
    ob = KalshiRestClient.parse_orderbook_response("TEST-MALFORMED", payload)
    assert len(ob.bids) == 1
    assert ob.best_bid == 45
    assert len(ob.asks) == 1
    assert ob.best_ask == 49

def test_crossed_orderbook():
    # YES bid 52c, NO bid 50c -> YES ask 50c
    payload = {
        "orderbook": {
            "yes": [[52, 100]],
            "no": [[50, 100]]
        }
    }
    ob = KalshiRestClient.parse_orderbook_response("TEST-CROSSED", payload)
    assert ob.best_bid == 52
    assert ob.best_ask == 50
    assert ob.is_crossed

@patch("urllib.request.urlopen")
def test_get_markets_mocked(mock_urlopen):
    mock_resp = MagicMock()
    mock_resp.status = 200
    mock_resp.read.return_value = json.dumps({
        "markets": [
            {"ticker": "T1", "event_ticker": "E1", "title": "Market 1", "status": "open"}
        ],
        "cursor": ""
    }).encode("utf-8")
    mock_urlopen.return_value.__enter__.return_value = mock_resp

    client = KalshiRestClient()
    res = client.get_markets(status="open", limit=1)
    assert "markets" in res
    assert len(res["markets"]) == 1
    assert res["markets"][0]["ticker"] == "T1"

@patch("urllib.request.urlopen")
def test_http_error_handling(mock_urlopen):
    import urllib.error
    mock_urlopen.side_effect = urllib.error.HTTPError(
        url="http://test",
        code=404,
        msg="Not Found",
        hdrs={},
        fp=None
    )

    client = KalshiRestClient()
    with pytest.raises(KalshiHttpError) as exc_info:
        client.get_market("NONEXISTENT")
    assert exc_info.value.status_code == 404
