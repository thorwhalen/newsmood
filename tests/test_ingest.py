"""Tests for newsmood.ingest using small synthetic stores."""

from datetime import datetime, timezone

import pytest

from newsmood.ingest import (
    CANONICAL_COLUMNS,
    PARSERS,
    ingest_searches,
    iter_canonical_rows,
    make_doc_id,
    make_text_to_embed,
    parse_key,
    parse_newsdata,
    parse_yahoo_finance,
    parse_yahoo_finance_headlines,
)


# ---------------------------------------------------------------------------
# Key parsing
# ---------------------------------------------------------------------------


class TestParseKey:
    def test_full_key(self):
        info = parse_key("2025-03-13/2025-03-13--18-00-04__Profit_Warning.json")
        assert info["date"] == "2025-03-13"
        assert info["query"] == "Profit_Warning"
        assert info["file_ts"] == datetime(2025, 3, 13, 18, 0, 4, tzinfo=timezone.utc)

    def test_empty_query(self):
        info = parse_key("2025-02-19/2025-02-19--16-25-05__.json")
        assert info["query"] == ""

    def test_no_filename(self):
        info = parse_key("2025-03-13/")
        assert info["date"] == "2025-03-13"
        assert info["file_ts"] == datetime(2025, 3, 13, tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# Text + doc_id
# ---------------------------------------------------------------------------


class TestMakeTextToEmbed:
    def test_title_and_desc(self):
        assert make_text_to_embed("T", "D") == "T. D"

    def test_strip_paywall(self):
        assert make_text_to_embed("T", "ONLY AVAILABLE IN PAID PLANS") == "T"

    def test_avoid_double_period(self):
        assert make_text_to_embed("End.", "More") == "End.More"

    def test_only_title(self):
        assert make_text_to_embed("T", None) == "T"

    def test_only_desc(self):
        assert make_text_to_embed(None, "D") == "D"

    def test_empty(self):
        assert make_text_to_embed(None, None) == ""


class TestMakeDocId:
    def test_case_and_whitespace_normalized(self):
        assert make_doc_id("Hello   World") == make_doc_id("hello world")

    def test_different_text_different_id(self):
        assert make_doc_id("a") != make_doc_id("b")

    def test_length(self):
        assert len(make_doc_id("x")) == 16


# ---------------------------------------------------------------------------
# Per-source parsers
# ---------------------------------------------------------------------------


def _info(query="Profit_Warning"):
    return {
        "date": "2025-03-13",
        "file_ts": datetime(2025, 3, 13, 18, 0, 4, tzinfo=timezone.utc),
        "query": query,
    }


class TestParseNewsdata:
    def test_basic(self):
        value = [
            {
                "title": "Earnings miss",
                "description": "Acme reported a Q3 miss.",
                "pubDate": "2025-03-13 09:00:00",
                "link": "https://x",
                "source_id": "wsj",
                "category": ["business"],
                "country": ["united states of america"],
                "language": "english",
            }
        ]
        rows = list(parse_newsdata(value, _info()))
        assert len(rows) == 1
        row = rows[0]
        assert row["source"] == "newsdata"
        assert row["title"] == "Earnings miss"
        assert row["text_to_embed"].startswith("Earnings miss. ")
        assert row["ts"].year == 2025
        assert row["source_id"] == "wsj"

    def test_skips_paywall_only(self):
        value = [
            {"title": None, "description": "ONLY AVAILABLE IN PAID PLANS", "pubDate": "2025-03-13"}
        ]
        assert list(parse_newsdata(value, _info())) == []

    def test_uses_file_ts_fallback(self):
        value = [{"title": "T", "description": "D"}]  # no pubDate
        rows = list(parse_newsdata(value, _info()))
        assert rows[0]["ts"] == _info()["file_ts"]


class TestParseYahooFinance:
    def test_basic(self):
        value = {
            "news": [
                {
                    "title": "Stocks slip",
                    "publisher": "Bloomberg",
                    "link": "https://...",
                    "providerPublishTime": 1741856400,  # 2025-03-13T09:00:00Z
                    "type": "STORY",
                }
            ]
        }
        rows = list(parse_yahoo_finance(value, _info()))
        assert len(rows) == 1
        r = rows[0]
        assert r["source"] == "yahoo_finance"
        assert r["text_to_embed"] == "Stocks slip"
        assert r["source_id"] == "Bloomberg"
        assert r["ts"] == datetime(2025, 3, 13, 9, tzinfo=timezone.utc)

    def test_no_news_field(self):
        assert list(parse_yahoo_finance({"other": []}, _info())) == []


class TestParseYahooFinanceHeadlines:
    def test_basic(self):
        value = ["Apple debuts iPhone 16e", "Stocks decline amid tariff threats"]
        rows = list(parse_yahoo_finance_headlines(value, _info(query="")))
        assert len(rows) == 2
        for r in rows:
            assert r["source"] == "yahoo_finance_headlines"
            assert r["ts"] == _info()["file_ts"]

    def test_skips_non_strings(self):
        value = ["ok", None, 42, "also ok"]
        rows = list(parse_yahoo_finance_headlines(value, _info()))
        assert [r["text_to_embed"] for r in rows] == ["ok", "also ok"]


# ---------------------------------------------------------------------------
# High-level ingest
# ---------------------------------------------------------------------------


class FakeStore(dict):
    """Just a dict — same interface as dol.JsonFiles for our purposes."""


def _make_store():
    s = FakeStore()
    s["newsdata/2025-03-13/2025-03-13--09-00-00__Earnings_Miss.json"] = [
        {"title": "Earnings miss", "description": "Q3", "pubDate": "2025-03-13 09:00:00"},
        {"title": "Profit warning", "description": "Guidance cut", "pubDate": "2025-03-13 15:30:00"},
    ]
    s["yahoo_finance/2025-03-13/2025-03-13--18-00-04__Profit_Warning.json"] = {
        "news": [
            {
                "title": "Markets react",
                "publisher": "Reuters",
                "providerPublishTime": 1741892404,
            }
        ]
    }
    s["yahoo_finance_headlines/2025-03-13/2025-03-13--12-00-00__.json"] = [
        "Tech stocks rally"
    ]
    s["unrelated/2025-03-13/foo.json"] = {"ignored": True}
    return s


def test_iter_canonical_rows_all_sources():
    store = _make_store()
    rows = list(iter_canonical_rows(store))
    sources = sorted({r["source"] for r in rows})
    assert sources == ["newsdata", "yahoo_finance", "yahoo_finance_headlines"]
    # 'unrelated/...' is silently skipped
    assert len(rows) == 4


def test_iter_canonical_rows_source_filter():
    store = _make_store()
    rows = list(iter_canonical_rows(store, source="newsdata"))
    assert len(rows) == 2
    assert all(r["source"] == "newsdata" for r in rows)


def test_iter_canonical_rows_date_filter():
    store = _make_store()
    # since 2025-03-14 -> no rows
    rows = list(iter_canonical_rows(store, since="2025-03-14"))
    assert rows == []


def test_iter_canonical_rows_unknown_source_raises():
    store = _make_store()
    with pytest.raises(ValueError, match="Unknown source"):
        list(iter_canonical_rows(store, source="bogus"))


def test_ingest_searches_columns_and_dtypes():
    df = ingest_searches(_make_store())
    assert df.columns.tolist() == list(CANONICAL_COLUMNS)
    assert len(df) == 4
    # ts column is tz-aware UTC datetime
    assert str(df["ts"].dt.tz) == "UTC"
    # rows sorted ascending
    assert df["ts"].is_monotonic_increasing


def test_ingest_searches_dedupe():
    """Same text in two sources gets one row (same doc_id)."""
    s = FakeStore()
    s["newsdata/2025-03-13/2025-03-13--09-00-00__X.json"] = [
        {"title": "Same headline", "description": None, "pubDate": "2025-03-13 09:00:00"}
    ]
    s["yahoo_finance_headlines/2025-03-13/2025-03-13--09-30-00__.json"] = [
        "Same headline"
    ]
    df = ingest_searches(s, dedupe=True)
    assert len(df) == 1
    # earliest ts kept
    assert df["ts"].iloc[0].hour == 9 and df["ts"].iloc[0].minute == 0
