"""Per-ticker article matching: lightweight regex-based NER.

Given a list of articles and a ticker universe with name variants, attach a
``tickers`` column listing matched tickers per article. The default universe
is a curated subset of the largest S&P 100 names — extend as needed.

Matching strategy
-----------------
- Tickers are uppercased and matched as standalone word tokens, optionally
  preceded by ``$``: ``AAPL``, ``$AAPL`` match; ``BAPPLE`` does not.
- Name variants are matched **case-insensitively** as whole tokens, allowing
  trailing legal suffixes (``Inc``, ``Corp``, ``Co``, ``Ltd``, ``plc``) to be
  optional. ``Apple Inc`` matches ``apple inc``, ``Apple Inc.``, ``apple``.
- A single article may match multiple tickers.

Examples
--------
>>> from newsmood.tickers import match_tickers, DEFAULT_UNIVERSE
>>> match_tickers('Apple Inc. announced new chips today.', universe=DEFAULT_UNIVERSE)
['AAPL']
>>> sorted(match_tickers('NVDA and AMD both rallied; $TSLA dipped.',
...                      universe=DEFAULT_UNIVERSE))
['AMD', 'NVDA', 'TSLA']
"""

import re
from collections.abc import Iterable, Mapping
from typing import Optional, Sequence

import pandas as pd


# A modest universe: S&P 100-ish picks across sectors with most common name
# variants. The list is intentionally finite — over-broad matchers (e.g.
# generic words like "GO" for Alphabet) are excluded.
DEFAULT_UNIVERSE: dict[str, list[str]] = {
    # Mega-cap tech
    "AAPL": ["Apple"],
    "MSFT": ["Microsoft"],
    "GOOGL": ["Alphabet", "Google"],
    "AMZN": ["Amazon"],
    "META": ["Meta Platforms", "Meta", "Facebook"],
    "NVDA": ["NVIDIA", "Nvidia"],
    "TSLA": ["Tesla"],
    # Other tech / semis
    "AMD": ["Advanced Micro Devices", "AMD"],
    "AVGO": ["Broadcom"],
    "INTC": ["Intel"],
    "QCOM": ["Qualcomm"],
    "ORCL": ["Oracle"],
    "ADBE": ["Adobe"],
    "CRM": ["Salesforce"],
    "NFLX": ["Netflix"],
    # Financials
    "BRK-B": ["Berkshire Hathaway", "Berkshire"],
    "JPM": ["JPMorgan", "JP Morgan", "JPMorgan Chase"],
    "BAC": ["Bank of America"],
    "WFC": ["Wells Fargo"],
    "GS": ["Goldman Sachs"],
    "MS": ["Morgan Stanley"],
    "C": ["Citigroup", "Citi"],
    "V": ["Visa"],
    "MA": ["Mastercard"],
    # Healthcare
    "JNJ": ["Johnson & Johnson"],
    "UNH": ["UnitedHealth", "United Health"],
    "PFE": ["Pfizer"],
    "MRK": ["Merck"],
    "ABBV": ["AbbVie"],
    "LLY": ["Eli Lilly", "Lilly"],
    # Consumer
    "WMT": ["Walmart", "Wal-Mart"],
    "HD": ["Home Depot"],
    "COST": ["Costco"],
    "MCD": ["McDonald's", "McDonalds"],
    "NKE": ["Nike"],
    "SBUX": ["Starbucks"],
    "KO": ["Coca-Cola", "Coca Cola"],
    "PEP": ["PepsiCo", "Pepsi"],
    "PG": ["Procter & Gamble", "P&G"],
    "DIS": ["Disney", "Walt Disney"],
    "TGT": ["Target Corp"],
    # Energy / industrials
    "XOM": ["ExxonMobil", "Exxon"],
    "CVX": ["Chevron"],
    "BA": ["Boeing"],
    "CAT": ["Caterpillar"],
    "GE": ["GE Aerospace", "General Electric"],
    # Telecom / utilities
    "T": ["AT&T"],
    "VZ": ["Verizon"],
    # Communication
    "TMUS": ["T-Mobile"],
    # ETFs / indices
    "SPY": ["S&P 500 ETF", "SPDR S&P 500"],
    "QQQ": ["Invesco QQQ", "Nasdaq 100 ETF"],
    "IWM": ["Russell 2000 ETF", "iShares Russell 2000"],
    # Crypto (large-cap proxies)
    "COIN": ["Coinbase"],
    "MSTR": ["MicroStrategy", "Strategy"],
}

# Suffixes that may follow a company name and should be optionally stripped.
_NAME_TAIL_PATTERN = (
    r"(?:[,.\s]+(?:Inc|Corporation|Corp|Co|Company|Ltd|Limited|plc|PLC|N\.V\.|S\.A\.|"
    r"AG|SE|SA|Holdings|Group|LLC))?\.?"
)


def _build_ticker_regex(ticker: str) -> re.Pattern:
    """Strict whole-token regex for a ticker, allowing optional ``$`` prefix."""
    return re.compile(rf"(?<![A-Za-z0-9])\$?{re.escape(ticker)}(?![A-Za-z0-9])")


def _build_name_regex(name: str) -> re.Pattern:
    """Case-insensitive name regex with optional legal suffix."""
    return re.compile(
        rf"(?<![A-Za-z0-9])"
        rf"{re.escape(name)}"
        rf"{_NAME_TAIL_PATTERN}"
        rf"(?![A-Za-z0-9])",
        re.IGNORECASE,
    )


class TickerMatcher:
    """Pre-compiled matcher over a universe. ``__call__`` returns matched tickers.

    >>> matcher = TickerMatcher({'AAPL': ['Apple'], 'MSFT': ['Microsoft']})
    >>> matcher('Apple beat estimates while Microsoft slipped.')
    ['AAPL', 'MSFT']
    >>> matcher('$AAPL is up 2%')
    ['AAPL']
    >>> matcher('No mentions here.')
    []
    """

    def __init__(self, universe: Mapping[str, Iterable[str]]):
        self._universe = {t: list(v) for t, v in universe.items()}
        self._patterns: list[tuple[str, list[re.Pattern]]] = [
            (
                ticker,
                [_build_ticker_regex(ticker)]
                + [_build_name_regex(name) for name in names],
            )
            for ticker, names in self._universe.items()
        ]

    def __call__(self, text: str) -> list[str]:
        if not text:
            return []
        matched: list[str] = []
        for ticker, patterns in self._patterns:
            if any(p.search(text) for p in patterns):
                matched.append(ticker)
        return matched

    @property
    def universe(self) -> dict[str, list[str]]:
        return dict(self._universe)


def match_tickers(
    text: str, *, universe: Mapping[str, Iterable[str]]
) -> list[str]:
    """One-shot helper; prefer :class:`TickerMatcher` for many calls."""
    return TickerMatcher(universe)(text)


def attach_tickers(
    df: pd.DataFrame,
    *,
    universe: Mapping[str, Iterable[str]] = DEFAULT_UNIVERSE,
    text_col: str = "text_to_embed",
    out_col: str = "tickers",
) -> pd.DataFrame:
    """Return a copy of ``df`` with an added ``tickers`` column (list[str]).

    Articles with no matches get an empty list.
    """
    matcher = TickerMatcher(universe)
    out = df.copy()
    out[out_col] = out[text_col].fillna("").map(matcher)
    return out


def explode_to_ticker_rows(
    df: pd.DataFrame, *, ticker_col: str = "tickers"
) -> pd.DataFrame:
    """Explode a ``tickers`` list column to one row per (article, ticker).

    Articles with zero matches are dropped. The new column is named ``ticker``
    (singular) for clarity at the panel level.
    """
    out = df.explode(ticker_col).rename(columns={ticker_col: "ticker"})
    out = out[out["ticker"].notna()].copy()
    return out.reset_index(drop=True)


def per_ticker_session_features(
    article_panel: pd.DataFrame,
    *,
    seed_vecs: Optional[Mapping[str, list[float]]] = None,
    session_col: str = "session",
    ticker_col: str = "ticker",
    vec_col: str = "vector",
    query_col: str = "query",
    top_queries: Optional[Sequence[str]] = None,
    include_mean_vector_components: bool = False,
) -> pd.DataFrame:
    """Session-by-ticker features.

    ``article_panel`` is an exploded DataFrame (one row per article-ticker
    pair) with ``session``, ``ticker``, and ``vector`` columns. Aggregation
    is identical to :func:`newsmood.features.session_features` but grouped by
    ``(ticker, session)``.
    """
    if article_panel.empty:
        return pd.DataFrame()
    from newsmood.features import _as_matrix, cosine_to_seeds
    import numpy as np

    valid = article_panel.dropna(subset=[session_col, ticker_col, vec_col])
    if valid.empty:
        return pd.DataFrame()

    seed_names: list[str] = []
    seed_matrix = None
    if seed_vecs:
        seed_names = list(seed_vecs.keys())
        seed_matrix = np.vstack(
            [np.asarray(v, dtype=float) for v in seed_vecs.values()]
        )

    queries_in_data: list[str] = []
    if query_col in valid.columns:
        if top_queries is not None:
            queries_in_data = sorted(top_queries)
        else:
            queries_in_data = sorted(
                q for q in valid[query_col].dropna().unique() if q
            )

    rows: list[dict] = []
    for (ticker, session), group in valid.groupby([ticker_col, session_col]):
        mat = _as_matrix(group[vec_col])
        if mat is None:
            continue
        mean_vec = mat.mean(axis=0)
        row: dict = {
            "ticker": ticker,
            "session": session,
            "news_count": int(len(group)),
        }
        if seed_matrix is not None:
            cos = cosine_to_seeds(mean_vec[None, :], seed_matrix).ravel()
            for name, val in zip(seed_names, cos):
                row[f"cos_{name}"] = float(val)
        if include_mean_vector_components:
            for i, v in enumerate(mean_vec):
                row[f"mean_vec_{i}"] = float(v)
        if queries_in_data:
            counts = group[query_col].value_counts()
            for q in queries_in_data:
                row[f"q_{q}"] = int(counts.get(q, 0))
        rows.append(row)
    return pd.DataFrame(rows).set_index(["ticker", "session"]).sort_index()


__all__ = [
    "DEFAULT_UNIVERSE",
    "TickerMatcher",
    "match_tickers",
    "attach_tickers",
    "explode_to_ticker_rows",
    "per_ticker_session_features",
]
