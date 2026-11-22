# newsmood.ingest

Ingest news from the local searches store into a canonical DataFrame.

Three on-disk sources are handled:

1. `newsdata`: file value is `List[dict]` (articles with `title`,
   `description`, `pubDate`, `source_id`, `link`, …). Title and
   description are usable; the `content` field is paywalled
   (`"ONLY AVAILABLE IN PAID PLANS"`).
2. `yahoo_finance`: file value is a top-level `dict` with a `news` key
   holding `List[dict]`. Each item has `title`, `publisher`, `link`,
   `providerPublishTime` (epoch), `type`, …
3. `yahoo_finance_headlines`: file value is `List[str]` (bare headlines).
   There is no per-article timestamp; we fall back to the file timestamp
   embedded in the filename: `YYYY-MM-DD--HH-MM-SS`.

The output schema is unified so every row has the same fields regardless of
source. Junk (paywall placeholders, empty text) is filtered out by default.

### Examples

```pycon
>>> from newsmood.ingest import ingest_searches, DEFAULT_SEARCHES_ROOT
>>> import dol
>>> store = dol.JsonFiles(DEFAULT_SEARCHES_ROOT)
>>> df = ingest_searches(store, source='yahoo_finance_headlines')
>>> df.columns.tolist()
['doc_id', 'ts', 'source', 'query', 'title', 'description',
 'text_to_embed', 'link', 'source_id', 'category', 'country',
 'language', 'key']
```

### Functions

| [`parse_key`](#newsmood.ingest.parse_key)(key)                                   | Extract `date`, `file_ts`, and `query` from a store key.              |
|---------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------|
| [`parse_newsdata`](#newsmood.ingest.parse_newsdata)(value, key_info)                  | Yield canonical rows from one `newsdata/*.json` value.                |
| [`parse_yahoo_finance`](#newsmood.ingest.parse_yahoo_finance)(value, key_info)             | Yield canonical rows from one `yahoo_finance/*.json` value.           |
| [`parse_yahoo_finance_headlines`](#newsmood.ingest.parse_yahoo_finance_headlines)(value, key_info)   | Yield canonical rows from one `yahoo_finance_headlines/*.json` value. |
| [`register_parser`](#newsmood.ingest.register_parser)(source, parser)                  | Register an additional source parser at runtime.                      |
| [`make_text_to_embed`](#newsmood.ingest.make_text_to_embed)(title, description)           | Build the canonical text used for embedding.                          |
| [`make_doc_id`](#newsmood.ingest.make_doc_id)(text, \*[, hash_len])                | SHA-1-based, lowercase-normalized content hash.                       |
| [`iter_canonical_rows`](#newsmood.ingest.iter_canonical_rows)(store, \*[, source, ...])    | Iterate canonical rows from a `dol.JsonFiles`-style store.            |
| [`ingest_searches`](#newsmood.ingest.ingest_searches)(store, \*[, source, since, ...]) | Read a searches store and return a canonical `DataFrame`.             |

### newsmood.ingest.ingest_searches(store, , source=None, since=None, until=None, dedupe=True)

Read a searches store and return a canonical `DataFrame`.

* **Parameters:**
  * **store** – Typically `dol.JsonFiles(DEFAULT_SEARCHES_ROOT)`.
  * **source** ([`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]) – Optional source filter; one of `SUPPORTED_SOURCES`.
  * **since** ([`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]) – Optional date bounds (`"YYYY-MM-DD"`), inclusive/exclusive.
  * **until** ([`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]) – Optional date bounds (`"YYYY-MM-DD"`), inclusive/exclusive.
  * **dedupe** ([`bool`](https://docs.python.org/3/builtins/functions.html#bool)) – If True (default), drop duplicate rows sharing the same `doc_id`
    keeping the earliest `ts`.
* **Returns:**
  Columns: `CANONICAL_COLUMNS`. `ts` is a tz-aware datetime
  (UTC). Rows with no usable text or no `ts` are dropped.
* **Return type:**
  `DataFrame`

### newsmood.ingest.iter_canonical_rows(store, , source=None, since=None, until=None)

Iterate canonical rows from a `dol.JsonFiles`-style store.

* **Parameters:**
  * **store** – A `MutableMapping` whose keys look like
    `"<source>/<YYYY-MM-DD>/<filename>"` and whose values are parsed JSON.
  * **source** ([`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]) – If given, restrict to one source (must be a key of `PARSERS`).
  * **since** ([`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]) – Optional `"YYYY-MM-DD"` date filters (inclusive `since`, exclusive
    `until`), evaluated against the file date — fast key-level pruning.
  * **until** ([`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]) – Optional `"YYYY-MM-DD"` date filters (inclusive `since`, exclusive
    `until`), evaluated against the file date — fast key-level pruning.
* **Return type:**
  [`Iterator`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Iterator)[[`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)]

### newsmood.ingest.make_doc_id(text, , hash_len=16)

SHA-1-based, lowercase-normalized content hash.

```pycon
>>> make_doc_id('Hello, World!') == make_doc_id('hello, world!')
True
>>> len(make_doc_id('x'))
16
```

* **Return type:**
  [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)

### newsmood.ingest.make_text_to_embed(title, description)

Build the canonical text used for embedding.

Concatenates title + “. “ + description, normalizes whitespace, strips
paywall placeholders.

```pycon
>>> make_text_to_embed('Earnings miss', 'Company X reported weak Q3.')
'Earnings miss. Company X reported weak Q3.'
>>> make_text_to_embed('Just title', None)
'Just title'
>>> make_text_to_embed(None, 'Just desc')
'Just desc'
>>> make_text_to_embed('t', 'ONLY AVAILABLE IN PAID PLANS')
't'
```

* **Return type:**
  [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)

### newsmood.ingest.parse_key(key)

Extract `date`, `file_ts`, and `query` from a store key.

```pycon
>>> parse_key('2025-03-13/2025-03-13--18-00-04__Profit_Warning.json')['query']
'Profit_Warning'
>>> parse_key('2025-03-13/2025-03-13--18-00-04__.json')['query']
''
>>> ts = parse_key('2025-03-13/2025-03-13--18-00-04__Profit_Warning.json')['file_ts']
>>> ts.isoformat()
'2025-03-13T18:00:04+00:00'
```

* **Return type:**
  [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`Any`](https://docs.python.org/3/library/typing.html#typing.Any)]]

### newsmood.ingest.parse_newsdata(value, key_info)

Yield canonical rows from one `newsdata/*.json` value.

* **Return type:**
  [`Iterator`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Iterator)[[`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)]

### newsmood.ingest.parse_yahoo_finance(value, key_info)

Yield canonical rows from one `yahoo_finance/*.json` value.

The shape is `{'news': [{'title', 'publisher', 'link',
'providerPublishTime', ...}, ...], ...}`.

* **Return type:**
  [`Iterator`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Iterator)[[`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)]

### newsmood.ingest.parse_yahoo_finance_headlines(value, key_info)

Yield canonical rows from one `yahoo_finance_headlines/*.json` value.

The file is a bare `List[str]` of headlines; no per-item timestamp, so
the file_ts (from the filename) is used for every row.

* **Return type:**
  [`Iterator`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Iterator)[[`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)]

### newsmood.ingest.register_parser(source, parser)

Register an additional source parser at runtime.

* **Return type:**
  [`None`](https://docs.python.org/3/builtins/constants.html#None)
