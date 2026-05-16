"""Ingest news from the local searches store into a canonical DataFrame.

Three on-disk sources are handled:

1. ``newsdata``: file value is ``List[dict]`` (articles with ``title``,
   ``description``, ``pubDate``, ``source_id``, ``link``, …). Title and
   description are usable; the ``content`` field is paywalled
   (``"ONLY AVAILABLE IN PAID PLANS"``).

2. ``yahoo_finance``: file value is a top-level ``dict`` with a ``news`` key
   holding ``List[dict]``. Each item has ``title``, ``publisher``, ``link``,
   ``providerPublishTime`` (epoch), ``type``, …

3. ``yahoo_finance_headlines``: file value is ``List[str]`` (bare headlines).
   There is no per-article timestamp; we fall back to the file timestamp
   embedded in the filename: ``YYYY-MM-DD--HH-MM-SS``.

The output schema is unified so every row has the same fields regardless of
source. Junk (paywall placeholders, empty text) is filtered out by default.

Examples
--------
>>> from newsmood.ingest import ingest_searches, DEFAULT_SEARCHES_ROOT
>>> import dol
>>> store = dol.JsonFiles(DEFAULT_SEARCHES_ROOT)  # doctest: +SKIP
>>> df = ingest_searches(store, source='yahoo_finance_headlines')  # doctest: +SKIP
>>> df.columns.tolist()  # doctest: +SKIP
['doc_id', 'ts', 'source', 'query', 'title', 'description',
 'text_to_embed', 'link', 'source_id', 'category', 'country',
 'language', 'key']
"""

import hashlib
import os
import re
from collections.abc import Iterable, Iterator
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

import pandas as pd


# -- Configuration -----------------------------------------------------------

DEFAULT_SEARCHES_ROOT = os.environ.get(
    "NEWSMOOD_SEARCHES_ROOT",
    str(Path("~/Dropbox/_odata/finance/mood/news/searches").expanduser()),
)

PAYWALL_PLACEHOLDER = "ONLY AVAILABLE IN PAID PLANS"

SUPPORTED_SOURCES = ("newsdata", "yahoo_finance", "yahoo_finance_headlines")

# Canonical output columns (in order)
CANONICAL_COLUMNS = (
    "doc_id",
    "ts",
    "source",
    "query",
    "title",
    "description",
    "text_to_embed",
    "link",
    "source_id",
    "category",
    "country",
    "language",
    "key",
)


# -- Filename parsing --------------------------------------------------------

# e.g. "2025-03-13/2025-03-13--18-00-04__Profit_Warning.json"
_KEY_RE = re.compile(
    r"(?P<date>\d{4}-\d{2}-\d{2})"
    r"(?:/(?P<dt>\d{4}-\d{2}-\d{2}--\d{2}-\d{2}-\d{2}))?"
    r"(?:__(?P<query>[^.]*))?"
)


def parse_key(key: str) -> dict[str, Optional[Any]]:
    """Extract ``date``, ``file_ts``, and ``query`` from a store key.

    >>> parse_key('2025-03-13/2025-03-13--18-00-04__Profit_Warning.json')['query']
    'Profit_Warning'
    >>> parse_key('2025-03-13/2025-03-13--18-00-04__.json')['query']
    ''
    >>> ts = parse_key('2025-03-13/2025-03-13--18-00-04__Profit_Warning.json')['file_ts']
    >>> ts.isoformat()
    '2025-03-13T18:00:04+00:00'
    """
    out: dict[str, Optional[Any]] = {"date": None, "file_ts": None, "query": None}
    m = _KEY_RE.search(key)
    if not m:
        return out
    out["date"] = m.group("date")
    out["query"] = m.group("query")
    dt = m.group("dt")
    if dt:
        try:
            out["file_ts"] = datetime.strptime(dt, "%Y-%m-%d--%H-%M-%S").replace(
                tzinfo=timezone.utc
            )
        except ValueError:
            out["file_ts"] = None
    if out["file_ts"] is None and out["date"]:
        try:
            out["file_ts"] = datetime.fromisoformat(out["date"]).replace(
                tzinfo=timezone.utc
            )
        except ValueError:
            pass
    return out


# -- Timestamp parsing -------------------------------------------------------


def _parse_timestamp(val: Any) -> Optional[datetime]:
    """Best-effort coercion of a timestamp-like value to aware UTC datetime."""
    if val is None or val == "":
        return None
    if isinstance(val, datetime):
        return val if val.tzinfo else val.replace(tzinfo=timezone.utc)
    if isinstance(val, (int, float)):
        v = float(val)
        if v > 1e12:  # ms -> s
            v /= 1000.0
        try:
            return datetime.fromtimestamp(v, tz=timezone.utc)
        except (OSError, ValueError, OverflowError):
            return None
    if isinstance(val, str):
        s = val.strip()
        if not s:
            return None
        # ISO-8601
        try:
            dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
            return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
        except ValueError:
            pass
        # Common news formats
        for fmt in (
            "%a, %d %b %Y %H:%M:%S %z",
            "%a, %d %b %Y %H:%M:%S %Z",
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%d",
        ):
            try:
                dt = datetime.strptime(s, fmt)
                return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
            except ValueError:
                continue
        # numeric string epoch
        try:
            return _parse_timestamp(float(s))
        except ValueError:
            return None
    return None


# -- Canonical text + doc_id -------------------------------------------------


def _norm_text(s: Optional[str]) -> str:
    if not s or s == PAYWALL_PLACEHOLDER:
        return ""
    return s.strip()


def make_text_to_embed(title: Optional[str], description: Optional[str]) -> str:
    """Build the canonical text used for embedding.

    Concatenates title + ". " + description, normalizes whitespace, strips
    paywall placeholders.

    >>> make_text_to_embed('Earnings miss', 'Company X reported weak Q3.')
    'Earnings miss. Company X reported weak Q3.'
    >>> make_text_to_embed('Just title', None)
    'Just title'
    >>> make_text_to_embed(None, 'Just desc')
    'Just desc'
    >>> make_text_to_embed('t', 'ONLY AVAILABLE IN PAID PLANS')
    't'
    """
    t = _norm_text(title)
    d = _norm_text(description)
    if t and d:
        # Avoid duplicate period
        sep = "" if t.endswith((".", "!", "?")) else ". "
        return f"{t}{sep}{d}"
    return t or d


def make_doc_id(text: str, *, hash_len: int = 16) -> str:
    """SHA-1-based, lowercase-normalized content hash.

    >>> make_doc_id('Hello, World!') == make_doc_id('hello, world!')
    True
    >>> len(make_doc_id('x'))
    16
    """
    canon = " ".join(text.lower().split())
    return hashlib.sha1(canon.encode("utf-8")).hexdigest()[:hash_len]


# -- Per-source parsers ------------------------------------------------------


def parse_newsdata(value: Any, key_info: dict) -> Iterator[dict]:
    """Yield canonical rows from one ``newsdata/*.json`` value."""
    if not isinstance(value, list):
        return
    for item in value:
        if not isinstance(item, dict):
            continue
        title = item.get("title")
        description = item.get("description")
        text = make_text_to_embed(title, description)
        if not text:
            continue
        ts = _parse_timestamp(item.get("pubDate")) or key_info.get("file_ts")
        yield {
            "doc_id": make_doc_id(text),
            "ts": ts,
            "source": "newsdata",
            "query": key_info.get("query"),
            "title": _norm_text(title) or None,
            "description": _norm_text(description) or None,
            "text_to_embed": text,
            "link": item.get("link"),
            "source_id": item.get("source_id"),
            "category": item.get("category"),
            "country": item.get("country"),
            "language": item.get("language"),
        }


def parse_yahoo_finance(value: Any, key_info: dict) -> Iterator[dict]:
    """Yield canonical rows from one ``yahoo_finance/*.json`` value.

    The shape is ``{'news': [{'title', 'publisher', 'link',
    'providerPublishTime', ...}, ...], ...}``.
    """
    if not isinstance(value, dict):
        return
    items = value.get("news")
    if not isinstance(items, list):
        return
    for item in items:
        if not isinstance(item, dict):
            continue
        title = item.get("title")
        text = make_text_to_embed(title, None)
        if not text:
            continue
        ts = _parse_timestamp(item.get("providerPublishTime")) or key_info.get("file_ts")
        yield {
            "doc_id": make_doc_id(text),
            "ts": ts,
            "source": "yahoo_finance",
            "query": key_info.get("query"),
            "title": _norm_text(title) or None,
            "description": None,
            "text_to_embed": text,
            "link": item.get("link"),
            "source_id": item.get("publisher"),
            "category": item.get("type"),  # e.g. STORY / VIDEO
            "country": None,
            "language": None,
        }


def parse_yahoo_finance_headlines(value: Any, key_info: dict) -> Iterator[dict]:
    """Yield canonical rows from one ``yahoo_finance_headlines/*.json`` value.

    The file is a bare ``List[str]`` of headlines; no per-item timestamp, so
    the file_ts (from the filename) is used for every row.
    """
    if not isinstance(value, list):
        return
    file_ts = key_info.get("file_ts")
    for item in value:
        if not isinstance(item, str):
            continue
        text = make_text_to_embed(item, None)
        if not text:
            continue
        yield {
            "doc_id": make_doc_id(text),
            "ts": file_ts,
            "source": "yahoo_finance_headlines",
            "query": key_info.get("query"),
            "title": text,
            "description": None,
            "text_to_embed": text,
            "link": None,
            "source_id": None,
            "category": None,
            "country": None,
            "language": None,
        }


# Registry of parsers — extension point.
PARSERS: dict[str, Callable[[Any, dict], Iterator[dict]]] = {
    "newsdata": parse_newsdata,
    "yahoo_finance": parse_yahoo_finance,
    "yahoo_finance_headlines": parse_yahoo_finance_headlines,
}


def register_parser(
    source: str, parser: Callable[[Any, dict], Iterator[dict]]
) -> None:
    """Register an additional source parser at runtime."""
    PARSERS[source] = parser


# -- High-level ingest -------------------------------------------------------


def _key_source(key: str) -> Optional[str]:
    """Return the source name from a store key, or None."""
    head = key.split("/", 1)[0]
    return head if head in PARSERS else None


def iter_canonical_rows(
    store,
    *,
    source: Optional[str] = None,
    since: Optional[str] = None,
    until: Optional[str] = None,
) -> Iterator[dict]:
    """Iterate canonical rows from a ``dol.JsonFiles``-style store.

    Parameters
    ----------
    store
        A ``MutableMapping`` whose keys look like
        ``"<source>/<YYYY-MM-DD>/<filename>"`` and whose values are parsed JSON.
    source
        If given, restrict to one source (must be a key of :data:`PARSERS`).
    since, until
        Optional ``"YYYY-MM-DD"`` date filters (inclusive ``since``, exclusive
        ``until``), evaluated against the file date — fast key-level pruning.
    """
    if source is not None and source not in PARSERS:
        raise ValueError(
            f"Unknown source {source!r}. Known: {list(PARSERS)}"
        )

    for key in store:
        src = _key_source(key)
        if src is None:
            continue
        if source is not None and src != source:
            continue
        info = parse_key(key)
        if since and (info.get("date") or "") < since:
            continue
        if until and (info.get("date") or "9999-99-99") >= until:
            continue
        try:
            value = store[key]
        except (KeyError, ValueError, OSError):
            continue
        for row in PARSERS[src](value, info):
            row["key"] = key
            yield row


def ingest_searches(
    store,
    *,
    source: Optional[str] = None,
    since: Optional[str] = None,
    until: Optional[str] = None,
    dedupe: bool = True,
) -> pd.DataFrame:
    """Read a searches store and return a canonical ``DataFrame``.

    Parameters
    ----------
    store
        Typically ``dol.JsonFiles(DEFAULT_SEARCHES_ROOT)``.
    source
        Optional source filter; one of :data:`SUPPORTED_SOURCES`.
    since, until
        Optional date bounds (``"YYYY-MM-DD"``), inclusive/exclusive.
    dedupe
        If True (default), drop duplicate rows sharing the same ``doc_id``
        keeping the earliest ``ts``.

    Returns
    -------
    DataFrame
        Columns: :data:`CANONICAL_COLUMNS`. ``ts`` is a tz-aware datetime
        (UTC). Rows with no usable text or no ``ts`` are dropped.
    """
    rows = list(iter_canonical_rows(store, source=source, since=since, until=until))
    if not rows:
        return pd.DataFrame(columns=list(CANONICAL_COLUMNS))
    df = pd.DataFrame(rows)
    # Ensure column order + presence
    for col in CANONICAL_COLUMNS:
        if col not in df.columns:
            df[col] = None
    df = df[list(CANONICAL_COLUMNS)]
    # Strict: drop rows without a valid timestamp
    df = df.dropna(subset=["ts", "text_to_embed"]).copy()
    # Normalize ts to tz-aware UTC
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    df = df.sort_values("ts").reset_index(drop=True)
    if dedupe:
        df = df.drop_duplicates(subset="doc_id", keep="first").reset_index(drop=True)
    return df


__all__ = [
    "DEFAULT_SEARCHES_ROOT",
    "PAYWALL_PLACEHOLDER",
    "SUPPORTED_SOURCES",
    "CANONICAL_COLUMNS",
    "PARSERS",
    "parse_key",
    "parse_newsdata",
    "parse_yahoo_finance",
    "parse_yahoo_finance_headlines",
    "register_parser",
    "make_text_to_embed",
    "make_doc_id",
    "iter_canonical_rows",
    "ingest_searches",
]
