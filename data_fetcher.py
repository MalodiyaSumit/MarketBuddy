"""
Data Fetcher Module
Fetches stock data from yfinance and news from NewsAPI
"""

import os
import yfinance as yf
import pandas as pd
import requests
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()

NEWSAPI_KEY = os.getenv("NEWSAPI_KEY", "")

# Indian stock symbols mapping
INDIAN_INDICES = {
    "NIFTY": "^NSEI",
    "SENSEX": "^BSESN",
    "BANKNIFTY": "^NSEBANK",
}

# Popular Indian stocks for monitoring
WATCHLIST = [
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS",
    "HINDUNILVR.NS", "SBIN.NS", "BHARTIARTL.NS", "KOTAKBANK.NS", "ITC.NS",
    "LT.NS", "AXISBANK.NS", "BAJFINANCE.NS", "MARUTI.NS", "TITAN.NS",
    "SUNPHARMA.NS", "ULTRACEMCO.NS", "NESTLEIND.NS", "WIPRO.NS", "HCLTECH.NS"
]


def get_stock_data(symbol: str, period: str = "3mo", interval: str = "1d") -> pd.DataFrame:
    """
    Fetch stock data from yfinance

    Args:
        symbol: Stock symbol (e.g., 'RELIANCE.NS' for NSE, 'AAPL' for US)
        period: Data period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
        interval: Data interval (1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo)

    Returns:
        DataFrame with OHLCV data
    """
    try:
        # Handle Indian index names
        if symbol.upper() in INDIAN_INDICES:
            symbol = INDIAN_INDICES[symbol.upper()]

        # Add .NS suffix for Indian stocks if not present
        if not any(suffix in symbol.upper() for suffix in ['.NS', '.BO', '^', '.']):
            symbol = f"{symbol}.NS"

        ticker = yf.Ticker(symbol)
        df = ticker.history(period=period, interval=interval)

        if df.empty:
            return pd.DataFrame()

        # Clean column names
        df.columns = [col.lower().replace(' ', '_') for col in df.columns]
        df = df.reset_index()

        return df

    except Exception as e:
        print(f"Error fetching data for {symbol}: {e}")
        return pd.DataFrame()


def get_stock_info(symbol: str) -> dict:
    """
    Get detailed stock information

    Args:
        symbol: Stock symbol

    Returns:
        Dictionary with stock info
    """
    try:
        if symbol.upper() in INDIAN_INDICES:
            symbol = INDIAN_INDICES[symbol.upper()]
        elif not any(suffix in symbol.upper() for suffix in ['.NS', '.BO', '^', '.']):
            symbol = f"{symbol}.NS"

        ticker = yf.Ticker(symbol)
        info = ticker.info

        return {
            "name": info.get("longName", info.get("shortName", symbol)),
            "symbol": symbol,
            "current_price": info.get("currentPrice", info.get("regularMarketPrice", 0)),
            "previous_close": info.get("previousClose", 0),
            "open": info.get("open", info.get("regularMarketOpen", 0)),
            "day_high": info.get("dayHigh", info.get("regularMarketDayHigh", 0)),
            "day_low": info.get("dayLow", info.get("regularMarketDayLow", 0)),
            "volume": info.get("volume", info.get("regularMarketVolume", 0)),
            "market_cap": info.get("marketCap", 0),
            "pe_ratio": info.get("trailingPE", 0),
            "52w_high": info.get("fiftyTwoWeekHigh", 0),
            "52w_low": info.get("fiftyTwoWeekLow", 0),
            "sector": info.get("sector", "N/A"),
            "industry": info.get("industry", "N/A"),
        }

    except Exception as e:
        print(f"Error fetching info for {symbol}: {e}")
        return {}


def get_index_data(index_name: str = "NIFTY") -> dict:
    """
    Get index data (NIFTY/SENSEX)

    Args:
        index_name: Name of index (NIFTY, SENSEX, BANKNIFTY)

    Returns:
        Dictionary with index data
    """
    symbol = INDIAN_INDICES.get(index_name.upper(), "^NSEI")

    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info
        hist = ticker.history(period="2d")

        if len(hist) >= 2:
            prev_close = hist['Close'].iloc[-2]
            current = hist['Close'].iloc[-1]
            change = current - prev_close
            change_pct = (change / prev_close) * 100
        else:
            current = info.get("regularMarketPrice", 0)
            prev_close = info.get("previousClose", 0)
            change = current - prev_close
            change_pct = (change / prev_close) * 100 if prev_close else 0

        return {
            "name": index_name.upper(),
            "value": current,
            "change": change,
            "change_pct": change_pct,
            "open": info.get("regularMarketOpen", 0),
            "high": info.get("regularMarketDayHigh", 0),
            "low": info.get("regularMarketDayLow", 0),
            "prev_close": prev_close,
        }

    except Exception as e:
        print(f"Error fetching index data for {index_name}: {e}")
        return {}


def get_market_news(query: str = "Indian stock market", limit: int = 5) -> list:
    """
    Fetch market news from NewsAPI

    Args:
        query: Search query
        limit: Number of articles to return

    Returns:
        List of news articles
    """
    if not NEWSAPI_KEY:
        return []

    try:
        url = "https://newsapi.org/v2/everything"
        params = {
            "q": query,
            "apiKey": NEWSAPI_KEY,
            "language": "en",
            "sortBy": "publishedAt",
            "pageSize": limit,
            "from": (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d"),
        }

        response = requests.get(url, params=params, timeout=10)
        data = response.json()

        if data.get("status") == "ok":
            articles = []
            for article in data.get("articles", [])[:limit]:
                articles.append({
                    "title": article.get("title", ""),
                    "description": article.get("description", ""),
                    "url": article.get("url", ""),
                    "source": article.get("source", {}).get("name", ""),
                    "published": article.get("publishedAt", ""),
                })
            return articles

        return []

    except Exception as e:
        print(f"Error fetching news: {e}")
        return []


def get_bulk_stock_data(symbols: list = None, period: str = "3mo") -> dict:
    """
    Fetch data for multiple stocks

    Args:
        symbols: List of stock symbols
        period: Data period

    Returns:
        Dictionary with symbol as key and DataFrame as value
    """
    if symbols is None:
        symbols = WATCHLIST

    data = {}
    for symbol in symbols:
        df = get_stock_data(symbol, period=period)
        if not df.empty:
            data[symbol] = df

    return data


if __name__ == "__main__":
    # Test the data fetcher
    print("Testing Data Fetcher...")

    # Test stock data
    print("\n1. Testing stock data fetch:")
    df = get_stock_data("RELIANCE")
    print(f"RELIANCE data shape: {df.shape}")
    print(df.tail())

    # Test stock info
    print("\n2. Testing stock info:")
    info = get_stock_info("TCS")
    print(f"TCS Info: {info}")

    # Test index data
    print("\n3. Testing index data:")
    nifty = get_index_data("NIFTY")
    print(f"NIFTY: {nifty}")

    sensex = get_index_data("SENSEX")
    print(f"SENSEX: {sensex}")

    # Test news
    print("\n4. Testing news fetch:")
    news = get_market_news(limit=3)
    for article in news:
        print(f"- {article['title'][:50]}...")
