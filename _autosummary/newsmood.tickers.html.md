# newsmood.tickers

Per-ticker article matching: lightweight regex-based NER.

Given a list of articles and a ticker universe with name variants, attach a
`tickers` column listing matched tickers per article. The default universe
is a curated subset of the largest S&P 100 names — extend as needed.

## Matching strategy

- Tickers are uppercased and matched as standalone word tokens, optionally
  preceded by `$`: `AAPL`, `$AAPL` match; `BAPPLE` does not.
- Name variants are matched **case-insensitively** as whole tokens, allowing
  trailing legal suffixes (`Inc`, `Corp`, `Co`, `Ltd`, `plc`) to be
  optional. `Apple Inc` matches `apple inc`, `Apple Inc.`, `apple`.
- A single article may match multiple tickers.

### Examples

```pycon
>>> from newsmood.tickers import match_tickers, DEFAULT_UNIVERSE
>>> match_tickers('Apple Inc. announced new chips today.', universe=DEFAULT_UNIVERSE)
['AAPL']
>>> sorted(match_tickers('NVDA and AMD both rallied; $TSLA dipped.',
...                      universe=DEFAULT_UNIVERSE))
['AMD', 'NVDA', 'TSLA']
```

### Functions

| [`match_tickers`](#newsmood.tickers.match_tickers)(text, \*, universe)                 | One-shot helper; prefer [`TickerMatcher`](#newsmood.tickers.TickerMatcher) for many calls.   |
|----------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------|
| [`attach_tickers`](#newsmood.tickers.attach_tickers)(df, \*[, universe, text_col, ...]) | Return a copy of `df` with an added `tickers` column (list[str]).                                        |
| [`explode_to_ticker_rows`](#newsmood.tickers.explode_to_ticker_rows)(df, \*[, ticker_col])      | Explode a `tickers` list column to one row per (article, ticker).                                        |
| [`per_ticker_session_features`](#newsmood.tickers.per_ticker_session_features)(article_panel, \*)    | Session-by-ticker features.                                                                              |

### Classes

| [`TickerMatcher`](#newsmood.tickers.TickerMatcher)(universe)   | Pre-compiled matcher over a universe.   |
|----------------------------------------------------------------------------|-----------------------------------------|

### *class* newsmood.tickers.TickerMatcher(universe)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

Pre-compiled matcher over a universe. `__call__` returns matched tickers.

```pycon
>>> matcher = TickerMatcher({'AAPL': ['Apple'], 'MSFT': ['Microsoft']})
>>> matcher('Apple beat estimates while Microsoft slipped.')
['AAPL', 'MSFT']
>>> matcher('$AAPL is up 2%')
['AAPL']
>>> matcher('No mentions here.')
[]
```

### newsmood.tickers.attach_tickers(df, , universe={'AAPL': ['Apple'], 'ABBV': ['AbbVie'], 'ADBE': ['Adobe'], 'AMD': ['Advanced Micro Devices', 'AMD'], 'AMZN': ['Amazon'], 'AVGO': ['Broadcom'], 'BA': ['Boeing'], 'BAC': ['Bank of America'], 'BRK-B': ['Berkshire Hathaway', 'Berkshire'], 'C': ['Citigroup', 'Citi'], 'CAT': ['Caterpillar'], 'COIN': ['Coinbase'], 'COST': ['Costco'], 'CRM': ['Salesforce'], 'CVX': ['Chevron'], 'DIS': ['Disney', 'Walt Disney'], 'GE': ['GE Aerospace', 'General Electric'], 'GOOGL': ['Alphabet', 'Google'], 'GS': ['Goldman Sachs'], 'HD': ['Home Depot'], 'INTC': ['Intel'], 'IWM': ['Russell 2000 ETF', 'iShares Russell 2000'], 'JNJ': ['Johnson & Johnson'], 'JPM': ['JPMorgan', 'JP Morgan', 'JPMorgan Chase'], 'KO': ['Coca-Cola', 'Coca Cola'], 'LLY': ['Eli Lilly', 'Lilly'], 'MA': ['Mastercard'], 'MCD': ["McDonald's", 'McDonalds'], 'META': ['Meta Platforms', 'Meta', 'Facebook'], 'MRK': ['Merck'], 'MS': ['Morgan Stanley'], 'MSFT': ['Microsoft'], 'MSTR': ['MicroStrategy', 'Strategy'], 'NFLX': ['Netflix'], 'NKE': ['Nike'], 'NVDA': ['NVIDIA', 'Nvidia'], 'ORCL': ['Oracle'], 'PEP': ['PepsiCo', 'Pepsi'], 'PFE': ['Pfizer'], 'PG': ['Procter & Gamble', 'P&G'], 'QCOM': ['Qualcomm'], 'QQQ': ['Invesco QQQ', 'Nasdaq 100 ETF'], 'SBUX': ['Starbucks'], 'SPY': ['S&P 500 ETF', 'SPDR S&P 500'], 'T': ['AT&T'], 'TGT': ['Target Corp'], 'TMUS': ['T-Mobile'], 'TSLA': ['Tesla'], 'UNH': ['UnitedHealth', 'United Health'], 'V': ['Visa'], 'VZ': ['Verizon'], 'WFC': ['Wells Fargo'], 'WMT': ['Walmart', 'Wal-Mart'], 'XOM': ['ExxonMobil', 'Exxon']}, text_col='text_to_embed', out_col='tickers')

Return a copy of `df` with an added `tickers` column (list[str]).

Articles with no matches get an empty list.

* **Return type:**
  `DataFrame`

### newsmood.tickers.explode_to_ticker_rows(df, , ticker_col='tickers')

Explode a `tickers` list column to one row per (article, ticker).

Articles with zero matches are dropped. The new column is named `ticker`
(singular) for clarity at the panel level.

* **Return type:**
  `DataFrame`

### newsmood.tickers.match_tickers(text, , universe)

One-shot helper; prefer [`TickerMatcher`](#newsmood.tickers.TickerMatcher) for many calls.

* **Return type:**
  [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]

### newsmood.tickers.per_ticker_session_features(article_panel, , seed_vecs=None, session_col='session', ticker_col='ticker', vec_col='vector', query_col='query', top_queries=None, include_mean_vector_components=False)

Session-by-ticker features.

`article_panel` is an exploded DataFrame (one row per article-ticker
pair) with `session`, `ticker`, and `vector` columns. Aggregation
is identical to [`newsmood.features.session_features()`](newsmood.features.html.md#newsmood.features.session_features) but grouped by
`(ticker, session)`.

* **Return type:**
  `DataFrame`
