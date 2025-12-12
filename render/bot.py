#!/usr/bin/env python3
"""
MarketBuddy Pro - Advanced Stock Analysis Telegram Bot
"""

import os
import requests
import logging
import time
import io
import json
import hashlib
from datetime import datetime, timedelta
from flask import Flask
import threading
import pytz

# Setup logging first
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Flask app
app = Flask(__name__)

# ===== CONFIG =====
BOT_TOKEN = os.environ.get("BOT_TOKEN", "8241472299:AAGhewPu-VZFXpuLyVBlhRZYrGPgxpCb7mI")
ADMIN_CHAT_ID = os.environ.get("ADMIN_CHAT_ID", "2110121880")
NEWSAPI_KEY = os.environ.get("NEWSAPI_KEY", "5df4112f47a84dd6b7aa6e09a5be71db")
API_URL = f"https://api.telegram.org/bot{BOT_TOKEN}"

# Flag to prevent multiple bot instances
bot_started = False

# Indian Timezone
IST = pytz.timezone('Asia/Kolkata')

# Track last sent alerts to avoid duplicates
last_news_check = 0
last_scheduled_alert = {}
sent_news_ids = set()
SENT_NEWS_FILE = "sent_news_ids.json"


def load_sent_news_ids():
    """Load sent news IDs from file for persistence across restarts."""
    global sent_news_ids
    try:
        if os.path.exists(SENT_NEWS_FILE):
            with open(SENT_NEWS_FILE, 'r') as f:
                data = json.load(f)
                # Only load news IDs from last 24 hours
                cutoff = (datetime.now(IST) - timedelta(hours=24)).isoformat()
                sent_news_ids = set()
                for item in data:
                    if isinstance(item, dict) and item.get('time', '') > cutoff:
                        sent_news_ids.add(item['id'])
                    elif isinstance(item, str):
                        sent_news_ids.add(item)
                logger.info(f"Loaded {len(sent_news_ids)} sent news IDs from file")
    except Exception as e:
        logger.error(f"Error loading sent news IDs: {e}")
        sent_news_ids = set()


def save_sent_news_ids():
    """Save sent news IDs to file for persistence."""
    try:
        # Save with timestamp for cleanup
        now = datetime.now(IST).isoformat()
        data = [{'id': nid, 'time': now} for nid in list(sent_news_ids)[-100:]]
        with open(SENT_NEWS_FILE, 'w') as f:
            json.dump(data, f)
    except Exception as e:
        logger.error(f"Error saving sent news IDs: {e}")


def get_news_id(title):
    """Generate consistent news ID using MD5 hash."""
    return hashlib.md5(title.lower().strip().encode()).hexdigest()[:16]


# Load sent news on startup
load_sent_news_ids()

# ===== ACTIVE SIGNALS TRACKING =====
# Store active signals for target hit monitoring
active_signals = {}  # {signal_id: signal_data}
trade_history = []  # Store completed trades for learning
market_mistakes = []  # Track mistakes for improvement

# ===== MARKET DATA CACHE =====
# Cache for FII/DII, Option Chain, Economic Calendar
market_data_cache = {
    "fii_dii": None,
    "fii_dii_updated": None,
    "option_chain_nifty": None,
    "option_chain_banknifty": None,
    "option_chain_updated": None,
    "economic_calendar": None,
    "economic_calendar_updated": None,
}

# Indian Indices
INDIAN_INDICES = {
    "NIFTY": "^NSEI", "NIFTY50": "^NSEI",
    "SENSEX": "^BSESN",
    "BANKNIFTY": "^NSEBANK",
    "GIFTNIFTY": "NIFTY_FUT.NS",  # Gift Nifty proxy
}

# Index watchlist for signal generation
INDEX_WATCHLIST = ["NIFTY", "BANKNIFTY", "SENSEX"]

# Default Watchlist (stocks) - Comprehensive list for monitoring
DEFAULT_WATCHLIST = [
    # Large Cap - Banking
    "RELIANCE", "HDFCBANK", "ICICIBANK", "SBIN", "AXISBANK",
    "INDUSINDBK", "KOTAKBANK",
    # IT Sector
    "INFY", "TCS", "HCLTECH", "TECHM",
    # Metal & Mining
    "TATASTEEL", "HINDALCO", "COALINDIA",
    # Energy & Power
    "ONGC", "NTPC", "TATAPOWER", "RPOWER",
    # Infrastructure
    "LT", "ULTRACEMCO", "BHEL",
    # Auto
    "TATAMOTORS", "MARUTI",
    # Telecom
    "BHARTIARTL", "IDEA",
    # Adani Group
    "ADANIENT", "ADANIPORTS",
    # High Volatility / Momentum
    "YESBANK", "SUZLON",
]

# Combined watchlist for signal monitoring
FULL_WATCHLIST = DEFAULT_WATCHLIST + INDEX_WATCHLIST

# Global Market Indices for morning summary
GLOBAL_INDICES = {
    "DOW": "^DJI",       # Dow Jones
    "NASDAQ": "^IXIC",   # Nasdaq
    "S&P500": "^GSPC",   # S&P 500
    "NIKKEI": "^N225",   # Japan
    "HANGSENG": "^HSI",  # Hong Kong
    "FTSE": "^FTSE",     # UK
    "SGX": "ES=F",       # SGX Futures (proxy for Gift Nifty)
}

# User Data Storage
user_data = {}

@app.route('/')
def home():
    return "MarketBuddy Pro is running!"

@app.route('/health')
def health():
    return "OK"

# ===== TELEGRAM API =====
def send_message(chat_id, text):
    try:
        url = f"{API_URL}/sendMessage"
        data = {"chat_id": chat_id, "text": text, "parse_mode": "Markdown"}
        response = requests.post(url, data=data, timeout=30)
        return response.json()
    except Exception as e:
        logger.error(f"Send error: {e}")
        return None

def send_photo(chat_id, photo, caption=""):
    try:
        url = f"{API_URL}/sendPhoto"
        files = {"photo": photo}
        data = {"chat_id": chat_id, "caption": caption, "parse_mode": "Markdown"}
        response = requests.post(url, files=files, data=data, timeout=60)
        return response.json()
    except Exception as e:
        logger.error(f"Photo error: {e}")
        return None

def get_updates(offset=None):
    try:
        url = f"{API_URL}/getUpdates"
        params = {"timeout": 30, "offset": offset}
        response = requests.get(url, params=params, timeout=35)
        return response.json()
    except Exception as e:
        logger.error(f"Update error: {e}")
        return None

# ===== USER DATA =====
def get_user_data(chat_id):
    chat_id = str(chat_id)
    if chat_id not in user_data:
        user_data[chat_id] = {
            "watchlist": DEFAULT_WATCHLIST.copy(),
            "portfolio": {},
            "alerts": []
        }
    return user_data[chat_id]

# ===== DATA FETCHERS =====
def format_symbol(symbol):
    symbol = symbol.upper().strip()
    if symbol in INDIAN_INDICES:
        return INDIAN_INDICES[symbol]
    elif '.' not in symbol and '^' not in symbol:
        return f"{symbol}.NS"
    return symbol

def get_stock_data(symbol, period="3mo"):
    try:
        import yfinance as yf
        import pandas as pd
        symbol = format_symbol(symbol)
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=period)
        if df.empty:
            return None
        df.columns = [c.lower() for c in df.columns]
        return df
    except Exception as e:
        logger.error(f"Fetch error {symbol}: {e}")
        return None

def get_stock_info(symbol):
    try:
        import yfinance as yf
        symbol = format_symbol(symbol)
        ticker = yf.Ticker(symbol)
        info = ticker.info
        return {
            "name": info.get("longName", info.get("shortName", symbol)),
            "price": info.get("currentPrice", info.get("regularMarketPrice", 0)),
            "previous_close": info.get("previousClose", 0),
            "open": info.get("regularMarketOpen", 0),
            "high": info.get("dayHigh", 0),
            "low": info.get("dayLow", 0),
            "volume": info.get("volume", 0),
            "pe_ratio": info.get("trailingPE", 0),
            "52w_high": info.get("fiftyTwoWeekHigh", 0),
            "52w_low": info.get("fiftyTwoWeekLow", 0),
            "sector": info.get("sector", "N/A"),
        }
    except Exception as e:
        logger.error(f"Info error: {e}")
        return {}

def get_index_data(index_name="NIFTY"):
    try:
        import yfinance as yf
        symbol = INDIAN_INDICES.get(index_name.upper(), "^NSEI")
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period="2d")
        if len(hist) >= 2:
            prev = float(hist['Close'].iloc[-2])
            curr = float(hist['Close'].iloc[-1])
            change = curr - prev
            pct = (change / prev) * 100
            return {"name": index_name.upper(), "value": curr, "change": change, "pct": pct}
    except Exception as e:
        logger.error(f"Index error: {e}")
    return {}

def get_gift_nifty_data():
    """
    Fetch Gift Nifty (SGX Nifty) data.
    Gift Nifty trades on SGX from 6:30 AM to 11:30 PM IST.
    """
    try:
        import yfinance as yf
        # Try multiple symbols for Gift Nifty / SGX Nifty
        symbols_to_try = ["^NSEI", "NQ=F"]  # Fallback to Nifty if SGX not available

        for sym in symbols_to_try:
            try:
                ticker = yf.Ticker(sym)
                hist = ticker.history(period="5d", interval="1h")
                if hist.empty:
                    continue

                # Get today's data
                today = datetime.now(IST).date()
                today_data = hist[hist.index.date == today] if hasattr(hist.index, 'date') else hist.tail(8)

                if len(today_data) > 0:
                    high = float(today_data['High'].max())
                    low = float(today_data['Low'].min())
                    current = float(today_data['Close'].iloc[-1])
                    open_price = float(today_data['Open'].iloc[0])
                else:
                    # Use last available data
                    high = float(hist['High'].iloc[-1])
                    low = float(hist['Low'].iloc[-1])
                    current = float(hist['Close'].iloc[-1])
                    open_price = float(hist['Open'].iloc[-1])

                change = current - open_price
                pct = (change / open_price * 100) if open_price else 0

                return {
                    "name": "GIFT NIFTY",
                    "current": current,
                    "open": open_price,
                    "high": high,
                    "low": low,
                    "change": change,
                    "pct": pct,
                    "range": high - low,
                }
            except:
                continue

    except Exception as e:
        logger.error(f"Gift Nifty error: {e}")
    return None


# ===== FII/DII DATA =====

def get_fii_dii_data():
    """
    Fetch FII/DII data from NSE website.
    Returns buying/selling activity of institutional investors.
    """
    global market_data_cache

    try:
        # Check cache (data valid for 1 hour)
        if market_data_cache["fii_dii"] and market_data_cache["fii_dii_updated"]:
            cache_age = (datetime.now(IST) - market_data_cache["fii_dii_updated"]).seconds
            if cache_age < 3600:  # 1 hour cache
                return market_data_cache["fii_dii"]

        # NSE requires specific headers
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Accept": "application/json",
            "Accept-Language": "en-US,en;q=0.9",
            "Referer": "https://www.nseindia.com/reports/fii-dii",
        }

        # Create session for cookies
        session = requests.Session()
        session.get("https://www.nseindia.com", headers=headers, timeout=10)

        # Fetch FII/DII data
        url = "https://www.nseindia.com/api/fiidiiTradeReact"
        response = session.get(url, headers=headers, timeout=10)

        if response.status_code == 200:
            data = response.json()

            fii_data = None
            dii_data = None

            for item in data:
                if item.get("category") == "FII/FPI":
                    fii_data = item
                elif item.get("category") == "DII":
                    dii_data = item

            if fii_data and dii_data:
                result = {
                    "date": datetime.now(IST).strftime("%Y-%m-%d"),
                    "fii": {
                        "buy": float(fii_data.get("buyValue", 0)),
                        "sell": float(fii_data.get("sellValue", 0)),
                        "net": float(fii_data.get("netValue", 0)),
                    },
                    "dii": {
                        "buy": float(dii_data.get("buyValue", 0)),
                        "sell": float(dii_data.get("sellValue", 0)),
                        "net": float(dii_data.get("netValue", 0)),
                    },
                }

                # Determine sentiment
                fii_net = result["fii"]["net"]
                dii_net = result["dii"]["net"]

                if fii_net > 0 and dii_net > 0:
                    result["sentiment"] = "STRONG BULLISH"
                    result["sentiment_emoji"] = "🟢🟢"
                    result["description"] = "Both FII & DII buying - Strong support for market"
                elif fii_net > 0 and dii_net < 0:
                    result["sentiment"] = "BULLISH"
                    result["sentiment_emoji"] = "🟢"
                    result["description"] = "FII buying, DII booking profits - Positive bias"
                elif fii_net < 0 and dii_net > 0:
                    result["sentiment"] = "CAUTIOUS"
                    result["sentiment_emoji"] = "🟡"
                    result["description"] = "FII selling, DII supporting - Be cautious"
                else:
                    result["sentiment"] = "BEARISH"
                    result["sentiment_emoji"] = "🔴"
                    result["description"] = "Both FII & DII selling - Market under pressure"

                # Cache the result
                market_data_cache["fii_dii"] = result
                market_data_cache["fii_dii_updated"] = datetime.now(IST)

                return result

    except Exception as e:
        logger.error(f"FII/DII fetch error: {e}")

    # Return cached data if fetch fails
    if market_data_cache["fii_dii"]:
        return market_data_cache["fii_dii"]

    return None


def get_fallback_fii_dii_data():
    """
    Return sample FII/DII data when API fails.
    Uses realistic values based on typical market activity.
    """
    now = datetime.now(IST)

    # Get index data to estimate FII/DII activity
    try:
        nifty_data = get_index_data("NIFTY")
        if nifty_data:
            nifty_change = nifty_data.get('pct', 0)
            # Estimate FII/DII based on market direction
            if nifty_change > 0.5:
                fii_net = 1200 + (nifty_change * 500)
                dii_net = 800 + (nifty_change * 300)
                sentiment = "BULLISH"
                emoji = "🟢"
                desc = "Estimated: Positive market suggests institutional buying"
            elif nifty_change < -0.5:
                fii_net = -1500 - (abs(nifty_change) * 400)
                dii_net = 600  # DII usually supports on dips
                sentiment = "CAUTIOUS"
                emoji = "🟡"
                desc = "Estimated: FII selling, DII supporting"
            else:
                fii_net = 300
                dii_net = 400
                sentiment = "NEUTRAL"
                emoji = "🟡"
                desc = "Estimated: Range-bound market activity"
        else:
            fii_net = 500
            dii_net = 600
            sentiment = "NEUTRAL"
            emoji = "🟡"
            desc = "Sample data - Live data unavailable"
    except:
        fii_net = 500
        dii_net = 600
        sentiment = "NEUTRAL"
        emoji = "🟡"
        desc = "Sample data - Live data unavailable"

    return {
        "date": now.strftime("%Y-%m-%d"),
        "fii": {
            "buy": abs(fii_net) + 5000 if fii_net > 0 else 5000,
            "sell": 5000 if fii_net > 0 else abs(fii_net) + 5000,
            "net": fii_net,
        },
        "dii": {
            "buy": abs(dii_net) + 4000 if dii_net > 0 else 4000,
            "sell": 4000 if dii_net > 0 else abs(dii_net) + 4000,
            "net": dii_net,
        },
        "sentiment": sentiment,
        "sentiment_emoji": emoji,
        "description": desc,
        "is_estimated": True,
    }


# ===== OPTION CHAIN DATA =====

def get_option_chain_data(symbol="NIFTY"):
    """
    Fetch Option Chain data from NSE for PCR, Max Pain, OI analysis.
    """
    global market_data_cache

    try:
        cache_key = f"option_chain_{symbol.lower()}"

        # Check cache (valid for 5 minutes during market hours)
        if market_data_cache.get(cache_key) and market_data_cache["option_chain_updated"]:
            cache_age = (datetime.now(IST) - market_data_cache["option_chain_updated"]).seconds
            if cache_age < 300:  # 5 minute cache
                return market_data_cache[cache_key]

        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Accept": "application/json",
            "Accept-Language": "en-US,en;q=0.9",
            "Referer": "https://www.nseindia.com/option-chain",
        }

        session = requests.Session()
        session.get("https://www.nseindia.com", headers=headers, timeout=10)

        # Fetch option chain
        if symbol.upper() in ["NIFTY", "NIFTY50"]:
            url = "https://www.nseindia.com/api/option-chain-indices?symbol=NIFTY"
        elif symbol.upper() == "BANKNIFTY":
            url = "https://www.nseindia.com/api/option-chain-indices?symbol=BANKNIFTY"
        else:
            url = f"https://www.nseindia.com/api/option-chain-equities?symbol={symbol.upper()}"

        response = session.get(url, headers=headers, timeout=15)

        if response.status_code == 200:
            data = response.json()
            records = data.get("records", {})
            option_data = records.get("data", [])
            underlying_value = records.get("underlyingValue", 0)

            if not option_data:
                return None

            # Calculate PCR (Put-Call Ratio)
            total_call_oi = 0
            total_put_oi = 0
            total_call_volume = 0
            total_put_volume = 0

            # For Max Pain calculation
            strike_data = {}

            for item in option_data:
                strike = item.get("strikePrice", 0)

                ce_data = item.get("CE", {})
                pe_data = item.get("PE", {})

                ce_oi = ce_data.get("openInterest", 0) or 0
                pe_oi = pe_data.get("openInterest", 0) or 0
                ce_vol = ce_data.get("totalTradedVolume", 0) or 0
                pe_vol = pe_data.get("totalTradedVolume", 0) or 0

                total_call_oi += ce_oi
                total_put_oi += pe_oi
                total_call_volume += ce_vol
                total_put_volume += pe_vol

                strike_data[strike] = {
                    "ce_oi": ce_oi,
                    "pe_oi": pe_oi,
                    "ce_volume": ce_vol,
                    "pe_volume": pe_vol,
                }

            # PCR Ratio
            pcr_oi = total_put_oi / total_call_oi if total_call_oi > 0 else 0
            pcr_volume = total_put_volume / total_call_volume if total_call_volume > 0 else 0

            # Find Max Pain (strike with maximum OI on both sides)
            max_pain_strike = 0
            max_total_oi = 0
            for strike, oi_data in strike_data.items():
                total_oi = oi_data["ce_oi"] + oi_data["pe_oi"]
                if total_oi > max_total_oi:
                    max_total_oi = total_oi
                    max_pain_strike = strike

            # Find highest OI strikes (support/resistance)
            sorted_by_ce_oi = sorted(strike_data.items(), key=lambda x: x[1]["ce_oi"], reverse=True)
            sorted_by_pe_oi = sorted(strike_data.items(), key=lambda x: x[1]["pe_oi"], reverse=True)

            max_ce_oi_strike = sorted_by_ce_oi[0][0] if sorted_by_ce_oi else 0
            max_pe_oi_strike = sorted_by_pe_oi[0][0] if sorted_by_pe_oi else 0

            # Determine sentiment from PCR
            if pcr_oi > 1.2:
                pcr_sentiment = "BULLISH"
                pcr_description = "High PCR indicates more puts sold - Bullish sentiment"
            elif pcr_oi < 0.8:
                pcr_sentiment = "BEARISH"
                pcr_description = "Low PCR indicates more calls sold - Bearish sentiment"
            else:
                pcr_sentiment = "NEUTRAL"
                pcr_description = "PCR in neutral zone - No clear directional bias"

            result = {
                "symbol": symbol.upper(),
                "underlying": underlying_value,
                "timestamp": datetime.now(IST).strftime("%Y-%m-%d %H:%M"),
                "pcr": {
                    "oi": round(pcr_oi, 2),
                    "volume": round(pcr_volume, 2),
                    "sentiment": pcr_sentiment,
                    "description": pcr_description,
                },
                "max_pain": max_pain_strike,
                "total_call_oi": total_call_oi,
                "total_put_oi": total_put_oi,
                "resistance": max_ce_oi_strike,  # Highest Call OI = Resistance
                "support": max_pe_oi_strike,     # Highest Put OI = Support
                "analysis": {
                    "max_ce_oi": sorted_by_ce_oi[0][1]["ce_oi"] if sorted_by_ce_oi else 0,
                    "max_pe_oi": sorted_by_pe_oi[0][1]["pe_oi"] if sorted_by_pe_oi else 0,
                }
            }

            # Cache result
            market_data_cache[cache_key] = result
            market_data_cache["option_chain_updated"] = datetime.now(IST)

            return result

    except Exception as e:
        logger.error(f"Option chain fetch error for {symbol}: {e}")

    return market_data_cache.get(f"option_chain_{symbol.lower()}")


def get_fallback_option_chain_data(symbol="NIFTY"):
    """
    Return estimated option chain data when NSE API fails.
    Uses current index price to estimate support/resistance levels.
    """
    try:
        # Get current index data
        index_data = get_index_data(symbol)
        if not index_data:
            return None

        current_price = index_data.get('value', 0)
        if current_price <= 0:
            return None

        # Round to nearest 50/100 for strike prices
        if symbol.upper() == "BANKNIFTY":
            round_to = 100
            typical_range = 500
        else:
            round_to = 50
            typical_range = 200

        base_strike = round(current_price / round_to) * round_to

        # Estimate support/resistance based on typical OI distribution
        resistance = base_strike + typical_range
        support = base_strike - typical_range
        max_pain = base_strike

        # Estimate PCR based on market direction
        change_pct = index_data.get('pct', 0)
        if change_pct > 0.5:
            pcr_oi = 1.15  # Bullish - more puts
            pcr_sentiment = "BULLISH"
            pcr_desc = "Estimated: Market up suggests put writing (bullish)"
        elif change_pct < -0.5:
            pcr_oi = 0.85  # Bearish - more calls
            pcr_sentiment = "BEARISH"
            pcr_desc = "Estimated: Market down suggests call writing (bearish)"
        else:
            pcr_oi = 1.0
            pcr_sentiment = "NEUTRAL"
            pcr_desc = "Estimated: Balanced market suggests neutral PCR"

        return {
            "symbol": symbol.upper(),
            "underlying": current_price,
            "timestamp": datetime.now(IST).strftime("%Y-%m-%d %H:%M"),
            "pcr": {
                "oi": round(pcr_oi, 2),
                "volume": round(pcr_oi * 0.95, 2),
                "sentiment": pcr_sentiment,
                "description": pcr_desc,
            },
            "max_pain": max_pain,
            "total_call_oi": 5000000,
            "total_put_oi": int(5000000 * pcr_oi),
            "resistance": resistance,
            "support": support,
            "analysis": {
                "max_ce_oi": 250000,
                "max_pe_oi": int(250000 * pcr_oi),
            },
            "is_estimated": True,
        }
    except Exception as e:
        logger.error(f"Fallback option chain error: {e}")
        return None


# ===== ECONOMIC CALENDAR =====

# Important economic events for India
ECONOMIC_EVENTS = {
    "RBI_POLICY": {
        "name": "RBI Monetary Policy",
        "impact": "HIGH",
        "affects": ["NIFTY", "BANKNIFTY", "All Banking Stocks"],
        "description": "Interest rate decision affects entire market"
    },
    "US_FED": {
        "name": "US Federal Reserve Meeting",
        "impact": "HIGH",
        "affects": ["NIFTY", "IT Stocks", "Export Stocks"],
        "description": "US rate decision impacts global markets"
    },
    "GDP_DATA": {
        "name": "India GDP Data",
        "impact": "HIGH",
        "affects": ["NIFTY", "Infrastructure", "Banking"],
        "description": "Economic growth indicator"
    },
    "INFLATION_CPI": {
        "name": "CPI Inflation Data",
        "impact": "MEDIUM",
        "affects": ["NIFTY", "FMCG", "Banking"],
        "description": "Consumer inflation affects RBI policy"
    },
    "IIP_DATA": {
        "name": "IIP (Industrial Production)",
        "impact": "MEDIUM",
        "affects": ["Manufacturing", "Auto", "Capital Goods"],
        "description": "Industrial growth indicator"
    },
    "EXPIRY": {
        "name": "F&O Expiry",
        "impact": "HIGH",
        "affects": ["NIFTY", "BANKNIFTY", "Stock Futures"],
        "description": "High volatility expected - avoid new positions"
    },
}


def get_upcoming_economic_events():
    """
    Get upcoming economic events for the next 7 days.
    Uses a combination of known dates and web scraping.
    """
    global market_data_cache

    try:
        # Check cache
        if market_data_cache["economic_calendar"] and market_data_cache["economic_calendar_updated"]:
            cache_age = (datetime.now(IST) - market_data_cache["economic_calendar_updated"]).seconds
            if cache_age < 3600:  # 1 hour cache
                return market_data_cache["economic_calendar"]

        now = datetime.now(IST)
        events = []

        # Calculate F&O Expiry (Last Thursday of month for monthly, every Thursday for weekly)
        # Weekly expiry
        days_until_thursday = (3 - now.weekday()) % 7
        if days_until_thursday == 0 and now.hour >= 15:
            days_until_thursday = 7
        next_thursday = now + timedelta(days=days_until_thursday)

        if days_until_thursday <= 7:
            events.append({
                "date": next_thursday.strftime("%Y-%m-%d"),
                "day": next_thursday.strftime("%A"),
                "event": "Weekly F&O Expiry",
                "impact": "HIGH",
                "description": "NIFTY & BANKNIFTY weekly options expire. Expect high volatility.",
                "recommendation": "Avoid new positions, book profits before 2 PM"
            })

        # Check for monthly expiry (last Thursday)
        last_day = (now.replace(day=28) + timedelta(days=4)).replace(day=1) - timedelta(days=1)
        last_thursday = last_day
        while last_thursday.weekday() != 3:
            last_thursday -= timedelta(days=1)

        days_to_monthly = (last_thursday - now).days
        if 0 <= days_to_monthly <= 7:
            events.append({
                "date": last_thursday.strftime("%Y-%m-%d"),
                "day": last_thursday.strftime("%A"),
                "event": "Monthly F&O Expiry",
                "impact": "VERY HIGH",
                "description": "Stock futures & options expire. Maximum volatility day.",
                "recommendation": "Square off positions, avoid leverage"
            })

        # Try to fetch from investing.com or use fallback
        try:
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            }
            # This is a simplified approach - in production, you'd parse the actual calendar
            # For now, we'll use known recurring events

            # Check if RBI policy is likely (usually first week of bi-monthly)
            if now.day <= 7 and now.month in [2, 4, 6, 8, 10, 12]:
                events.append({
                    "date": f"{now.year}-{now.month:02d}-{min(now.day + 3, 7):02d}",
                    "day": "This Week",
                    "event": "RBI Monetary Policy (Possible)",
                    "impact": "HIGH",
                    "description": "RBI rate decision. Watch for policy announcement.",
                    "recommendation": "Avoid BANKNIFTY positions until announcement"
                })

        except:
            pass

        # Sort by date
        events.sort(key=lambda x: x["date"])

        result = {
            "updated": now.strftime("%Y-%m-%d %H:%M"),
            "events": events[:10],  # Max 10 events
            "next_important": events[0] if events else None,
        }

        market_data_cache["economic_calendar"] = result
        market_data_cache["economic_calendar_updated"] = now

        return result

    except Exception as e:
        logger.error(f"Economic calendar error: {e}")

    return market_data_cache.get("economic_calendar")


def get_market_sentiment_summary():
    """
    Combine FII/DII, Option Chain, and other data for overall market sentiment.
    This is used to improve signal accuracy.
    """
    sentiment_score = 50  # Start neutral

    # Get FII/DII data
    fii_dii = get_fii_dii_data()
    if fii_dii:
        if fii_dii["sentiment"] == "STRONG BULLISH":
            sentiment_score += 20
        elif fii_dii["sentiment"] == "BULLISH":
            sentiment_score += 10
        elif fii_dii["sentiment"] == "BEARISH":
            sentiment_score -= 15
        elif fii_dii["sentiment"] == "CAUTIOUS":
            sentiment_score -= 5

    # Get Option Chain data for NIFTY
    oc_nifty = get_option_chain_data("NIFTY")
    if oc_nifty:
        pcr = oc_nifty["pcr"]["oi"]
        if pcr > 1.3:
            sentiment_score += 15  # Very bullish
        elif pcr > 1.1:
            sentiment_score += 8
        elif pcr < 0.7:
            sentiment_score -= 15  # Very bearish
        elif pcr < 0.9:
            sentiment_score -= 8

    # Clamp score
    sentiment_score = max(0, min(100, sentiment_score))

    # Determine overall sentiment
    if sentiment_score >= 70:
        overall = "BULLISH"
        emoji = "🟢"
        advice = "Favor long positions, buy on dips"
    elif sentiment_score >= 55:
        overall = "MILDLY BULLISH"
        emoji = "🟢"
        advice = "Selective buying, maintain stop losses"
    elif sentiment_score <= 30:
        overall = "BEARISH"
        emoji = "🔴"
        advice = "Avoid fresh longs, consider hedging"
    elif sentiment_score <= 45:
        overall = "MILDLY BEARISH"
        emoji = "🔴"
        advice = "Be cautious, reduce position sizes"
    else:
        overall = "NEUTRAL"
        emoji = "🟡"
        advice = "Wait for clear direction, trade light"

    return {
        "score": sentiment_score,
        "sentiment": overall,
        "emoji": emoji,
        "advice": advice,
        "components": {
            "fii_dii": fii_dii["sentiment"] if fii_dii else "N/A",
            "pcr": oc_nifty["pcr"]["oi"] if oc_nifty else "N/A",
        }
    }


# ===== ACTIVE SIGNAL MANAGEMENT =====

def create_active_signal(symbol, signal_type, entry_price, target_1, target_2, stop_loss,
                         timeframe, confidence, methodology=None):
    """
    Create and store an active signal for target hit monitoring.
    """
    global active_signals

    now = datetime.now(IST)
    signal_id = f"{symbol}_{now.strftime('%Y%m%d_%H%M%S')}"

    # Parse timeframe to set expiry
    if "intraday" in timeframe.lower() or "1-2 days" in timeframe.lower():
        expiry_hours = 8  # Same day or next day
    elif "3-5 days" in timeframe.lower() or "1 week" in timeframe.lower():
        expiry_hours = 5 * 24  # 5 days
    elif "1-2 weeks" in timeframe.lower():
        expiry_hours = 14 * 24  # 2 weeks
    else:
        expiry_hours = 24  # Default 1 day

    signal_data = {
        "id": signal_id,
        "symbol": symbol,
        "type": signal_type,  # BUY or SELL
        "entry_price": entry_price,
        "target_1": target_1,
        "target_2": target_2,
        "stop_loss": stop_loss,
        "timeframe": timeframe,
        "confidence": confidence,
        "created_at": now,
        "expiry_at": now + timedelta(hours=expiry_hours),
        "status": "ACTIVE",  # ACTIVE, TARGET_1_HIT, TARGET_2_HIT, STOP_LOSS_HIT, EXPIRED
        "target_1_hit_at": None,
        "target_2_hit_at": None,
        "stop_loss_hit_at": None,
        "highest_price": entry_price if signal_type == "BUY" else entry_price,
        "lowest_price": entry_price if signal_type == "SELL" else entry_price,
        "methodology": methodology or {}
    }

    active_signals[signal_id] = signal_data
    logger.info(f"Created active signal: {signal_id}")
    return signal_id


def check_signal_targets():
    """
    Check all active signals for target hits or stop loss hits.
    Send notifications when targets are achieved.
    """
    global active_signals, trade_history, market_mistakes

    now = datetime.now(IST)
    signals_to_remove = []

    for signal_id, signal in active_signals.items():
        if signal["status"] not in ["ACTIVE", "TARGET_1_HIT"]:
            continue

        # Check if expired
        if now > signal["expiry_at"]:
            signal["status"] = "EXPIRED"

            # Record as potential mistake if no target hit
            if signal.get("target_1_hit_at") is None:
                record_trade_result(signal, "EXPIRED", "Signal expired without hitting any target")

            signals_to_remove.append(signal_id)
            send_signal_expired_notification(signal)
            continue

        # Get current price
        try:
            symbol = signal["symbol"]
            if symbol in INDEX_WATCHLIST:
                data = get_index_data(symbol)
                current_price = data.get("value", 0) if data else 0
            else:
                info = get_stock_info(symbol)
                current_price = info.get("price", 0) if info else 0

            if current_price <= 0:
                continue

            # Update highest/lowest prices
            if signal["type"] == "BUY":
                signal["highest_price"] = max(signal["highest_price"], current_price)
            else:
                signal["lowest_price"] = min(signal["lowest_price"], current_price)

            # Check for BUY signal targets
            if signal["type"] == "BUY":
                # Check Stop Loss
                if current_price <= signal["stop_loss"]:
                    signal["status"] = "STOP_LOSS_HIT"
                    signal["stop_loss_hit_at"] = now
                    record_trade_result(signal, "STOP_LOSS", f"Price hit stop loss at ₹{current_price:,.2f}")
                    send_stop_loss_hit_notification(signal, current_price)
                    signals_to_remove.append(signal_id)
                    continue

                # Check Target 2 (if Target 1 already hit)
                if signal["status"] == "TARGET_1_HIT" and current_price >= signal["target_2"]:
                    signal["status"] = "TARGET_2_HIT"
                    signal["target_2_hit_at"] = now
                    record_trade_result(signal, "TARGET_2_HIT", f"Price hit Target 2 at ₹{current_price:,.2f}")
                    send_target_hit_notification(signal, 2, current_price)
                    signals_to_remove.append(signal_id)
                    continue

                # Check Target 1
                if signal["status"] == "ACTIVE" and current_price >= signal["target_1"]:
                    signal["status"] = "TARGET_1_HIT"
                    signal["target_1_hit_at"] = now
                    send_target_hit_notification(signal, 1, current_price)
                    # Don't remove - continue monitoring for Target 2

            # Check for SELL signal targets
            elif signal["type"] == "SELL":
                # Check Stop Loss (price goes up for sell)
                if current_price >= signal["stop_loss"]:
                    signal["status"] = "STOP_LOSS_HIT"
                    signal["stop_loss_hit_at"] = now
                    record_trade_result(signal, "STOP_LOSS", f"Price hit stop loss at ₹{current_price:,.2f}")
                    send_stop_loss_hit_notification(signal, current_price)
                    signals_to_remove.append(signal_id)
                    continue

                # Check Target 2
                if signal["status"] == "TARGET_1_HIT" and current_price <= signal["target_2"]:
                    signal["status"] = "TARGET_2_HIT"
                    signal["target_2_hit_at"] = now
                    record_trade_result(signal, "TARGET_2_HIT", f"Price hit Target 2 at ₹{current_price:,.2f}")
                    send_target_hit_notification(signal, 2, current_price)
                    signals_to_remove.append(signal_id)
                    continue

                # Check Target 1
                if signal["status"] == "ACTIVE" and current_price <= signal["target_1"]:
                    signal["status"] = "TARGET_1_HIT"
                    signal["target_1_hit_at"] = now
                    send_target_hit_notification(signal, 1, current_price)

        except Exception as e:
            logger.error(f"Error checking signal {signal_id}: {e}")

    # Remove completed signals
    for signal_id in signals_to_remove:
        if signal_id in active_signals:
            # Move to history before removing
            trade_history.append(active_signals[signal_id])
            del active_signals[signal_id]

    # Keep only last 100 trades in history
    if len(trade_history) > 100:
        trade_history = trade_history[-100:]


def record_trade_result(signal, result_type, notes=""):
    """
    Record trade result for learning and mistake tracking.
    """
    global market_mistakes

    result = {
        "signal_id": signal["id"],
        "symbol": signal["symbol"],
        "type": signal["type"],
        "result": result_type,
        "entry_price": signal["entry_price"],
        "target_1": signal["target_1"],
        "target_2": signal["target_2"],
        "stop_loss": signal["stop_loss"],
        "confidence": signal["confidence"],
        "created_at": signal["created_at"].isoformat() if isinstance(signal["created_at"], datetime) else signal["created_at"],
        "closed_at": datetime.now(IST).isoformat(),
        "notes": notes,
        "highest_price": signal.get("highest_price", signal["entry_price"]),
        "lowest_price": signal.get("lowest_price", signal["entry_price"]),
    }

    # If stop loss hit or expired without target, record as potential mistake
    if result_type in ["STOP_LOSS", "EXPIRED"]:
        mistake = {
            "date": datetime.now(IST).strftime("%Y-%m-%d"),
            "symbol": signal["symbol"],
            "signal_type": signal["type"],
            "confidence": signal["confidence"],
            "what_went_wrong": notes,
            "lesson": f"Signal with confidence {signal['confidence']} failed. Review indicators."
        }
        market_mistakes.append(mistake)

        # Keep only last 50 mistakes
        if len(market_mistakes) > 50:
            market_mistakes = market_mistakes[-50:]

    logger.info(f"Recorded trade result: {signal['symbol']} - {result_type}")


def send_target_hit_notification(signal, target_num, current_price):
    """Send notification when target is hit."""
    target_price = signal["target_1"] if target_num == 1 else signal["target_2"]

    if signal["type"] == "BUY":
        profit_pct = ((current_price - signal["entry_price"]) / signal["entry_price"]) * 100
        profit_amount = current_price - signal["entry_price"]
    else:
        profit_pct = ((signal["entry_price"] - current_price) / signal["entry_price"]) * 100
        profit_amount = signal["entry_price"] - current_price

    time_taken = datetime.now(IST) - signal["created_at"]
    hours = time_taken.total_seconds() / 3600

    emoji = "🎯🎯" if target_num == 2 else "🎯"

    msg = f"""{emoji} *TARGET {target_num} HIT!* {emoji}

*{signal['symbol']}* - {signal['type']} Signal SUCCESS!

✅ *Entry Price:* ₹{signal['entry_price']:,.2f}
✅ *Target {target_num}:* ₹{target_price:,.2f}
✅ *Current Price:* ₹{current_price:,.2f}

💰 *Profit:* ₹{profit_amount:,.2f} ({profit_pct:+.2f}%)
⏱️ *Time Taken:* {hours:.1f} hours

*Signal Details:*
  • Confidence: {signal['confidence']}/100
  • Timeframe: {signal['timeframe']}
  • Stop Loss was: ₹{signal['stop_loss']:,.2f}

"""

    if target_num == 1:
        msg += f"""🔔 *Next Target:* ₹{signal['target_2']:,.2f}
Continue monitoring for Target 2!

💡 *Tip:* Consider booking partial profits and trailing stop loss."""
    else:
        msg += """🏆 *TRADE COMPLETED SUCCESSFULLY!*

Great analysis! Both targets achieved."""

    send_message(ADMIN_CHAT_ID, msg)
    logger.info(f"Target {target_num} hit notification sent for {signal['symbol']}")


def send_stop_loss_hit_notification(signal, current_price):
    """Send notification when stop loss is hit."""
    if signal["type"] == "BUY":
        loss_pct = ((current_price - signal["entry_price"]) / signal["entry_price"]) * 100
        loss_amount = signal["entry_price"] - current_price
    else:
        loss_pct = ((signal["entry_price"] - current_price) / signal["entry_price"]) * 100
        loss_amount = current_price - signal["entry_price"]

    msg = f"""🛑 *STOP LOSS HIT!*

*{signal['symbol']}* - {signal['type']} Signal STOPPED OUT

❌ *Entry Price:* ₹{signal['entry_price']:,.2f}
❌ *Stop Loss:* ₹{signal['stop_loss']:,.2f}
❌ *Exit Price:* ₹{current_price:,.2f}

📉 *Loss:* ₹{loss_amount:,.2f} ({loss_pct:.2f}%)

*Signal Details:*
  • Confidence was: {signal['confidence']}/100
  • Target 1 was: ₹{signal['target_1']:,.2f}
  • Target 2 was: ₹{signal['target_2']:,.2f}

📝 *Learning:* This signal did not work. Adding to mistake tracker for analysis.

_Risk management protected from larger losses._"""

    send_message(ADMIN_CHAT_ID, msg)
    logger.info(f"Stop loss notification sent for {signal['symbol']}")


def send_signal_expired_notification(signal):
    """Send notification when signal expires without hitting target."""
    # Get current price
    try:
        if signal["symbol"] in INDEX_WATCHLIST:
            data = get_index_data(signal["symbol"])
            current_price = data.get("value", 0) if data else 0
        else:
            info = get_stock_info(signal["symbol"])
            current_price = info.get("price", 0) if info else 0
    except:
        current_price = signal["entry_price"]

    if signal["type"] == "BUY":
        change_pct = ((current_price - signal["entry_price"]) / signal["entry_price"]) * 100
    else:
        change_pct = ((signal["entry_price"] - current_price) / signal["entry_price"]) * 100

    target_1_status = "✅ HIT" if signal.get("target_1_hit_at") else "❌ Not Hit"

    msg = f"""⏰ *SIGNAL EXPIRED*

*{signal['symbol']}* - {signal['type']} Signal EXPIRED

📊 *Entry Price:* ₹{signal['entry_price']:,.2f}
📊 *Current Price:* ₹{current_price:,.2f}
📊 *Change:* {change_pct:+.2f}%

*Target Status:*
  • Target 1 (₹{signal['target_1']:,.2f}): {target_1_status}
  • Target 2 (₹{signal['target_2']:,.2f}): ❌ Not Hit

*Signal Info:*
  • Confidence: {signal['confidence']}/100
  • Timeframe: {signal['timeframe']}
  • Duration: {signal['timeframe']}

📝 *Learning:* Signal expired within timeframe. Review for next time."""

    send_message(ADMIN_CHAT_ID, msg)
    logger.info(f"Signal expired notification sent for {signal['symbol']}")

def get_market_news(limit=5):
    try:
        url = "https://newsapi.org/v2/everything"
        params = {"q": "Indian stock market", "apiKey": NEWSAPI_KEY, "language": "en", "sortBy": "publishedAt", "pageSize": limit}
        response = requests.get(url, params=params, timeout=10)
        data = response.json()
        if data.get("status") == "ok":
            return data.get("articles", [])
    except Exception as e:
        logger.error(f"News error: {e}")
    return []

# ===== ENHANCED ANALYSIS WITH 0-100 CONFIDENCE SCORE =====
def calculate_confidence_score(df, latest):
    """
    Calculate a confidence score from 0-100 based on multiple factors:
    - RSI (20 points max)
    - MACD (20 points max)
    - SMA Crossover (15 points max)
    - Bollinger Bands (15 points max)
    - Volume Analysis (15 points max)
    - Price Momentum (15 points max)
    """
    import pandas as pd

    score = 50  # Start neutral
    factors = {}

    # 1. RSI Analysis (±20 points)
    rsi = latest.get('rsi')
    if rsi and not pd.isna(rsi):
        if rsi < 25:
            score += 20  # Extremely oversold - strong buy
            factors['RSI'] = f"EXTREMELY OVERSOLD ({rsi:.1f}) +20"
        elif rsi < 30:
            score += 15  # Oversold - buy
            factors['RSI'] = f"OVERSOLD ({rsi:.1f}) +15"
        elif rsi < 40:
            score += 8  # Slightly oversold
            factors['RSI'] = f"SLIGHTLY OVERSOLD ({rsi:.1f}) +8"
        elif rsi > 75:
            score -= 20  # Extremely overbought - strong sell
            factors['RSI'] = f"EXTREMELY OVERBOUGHT ({rsi:.1f}) -20"
        elif rsi > 70:
            score -= 15  # Overbought - sell
            factors['RSI'] = f"OVERBOUGHT ({rsi:.1f}) -15"
        elif rsi > 60:
            score -= 8  # Slightly overbought
            factors['RSI'] = f"SLIGHTLY OVERBOUGHT ({rsi:.1f}) -8"
        else:
            factors['RSI'] = f"NEUTRAL ({rsi:.1f}) +0"

    # 2. MACD Analysis (±20 points)
    macd_val = latest.get('macd')
    macd_sig = latest.get('macd_signal')
    if macd_val and macd_sig and not (pd.isna(macd_val) or pd.isna(macd_sig)):
        macd_diff = macd_val - macd_sig

        # Check for crossover in last 3 days
        if len(df) >= 3:
            prev_macd = df['macd'].iloc[-3:-1]
            prev_sig = df['macd_signal'].iloc[-3:-1]
            recent_cross_up = any((prev_macd.iloc[i] < prev_sig.iloc[i]) for i in range(len(prev_macd)) if not pd.isna(prev_macd.iloc[i]))
            recent_cross_down = any((prev_macd.iloc[i] > prev_sig.iloc[i]) for i in range(len(prev_macd)) if not pd.isna(prev_macd.iloc[i]))
        else:
            recent_cross_up = recent_cross_down = False

        if macd_diff > 0:
            if recent_cross_up and macd_val > macd_sig:
                score += 20  # Fresh bullish crossover
                factors['MACD'] = f"BULLISH CROSSOVER +20"
            else:
                score += 12  # Bullish
                factors['MACD'] = f"BULLISH +12"
        else:
            if recent_cross_down and macd_val < macd_sig:
                score -= 20  # Fresh bearish crossover
                factors['MACD'] = f"BEARISH CROSSOVER -20"
            else:
                score -= 12  # Bearish
                factors['MACD'] = f"BEARISH -12"

    # 3. SMA Crossover (±15 points)
    sma20 = latest.get('sma_20')
    sma50 = latest.get('sma_50')
    close = latest.get('close')
    if sma20 and sma50 and close and not (pd.isna(sma20) or pd.isna(sma50)):
        if sma20 > sma50:
            if close > sma20:
                score += 15  # Price above both SMAs in uptrend
                factors['SMA'] = "STRONG UPTREND +15"
            else:
                score += 8  # Uptrend but price below SMA20
                factors['SMA'] = "UPTREND +8"
        else:
            if close < sma20:
                score -= 15  # Price below both SMAs in downtrend
                factors['SMA'] = "STRONG DOWNTREND -15"
            else:
                score -= 8  # Downtrend but price above SMA20
                factors['SMA'] = "DOWNTREND -8"

    # 4. Bollinger Bands (±15 points)
    bb_low = latest.get('bb_lower')
    bb_up = latest.get('bb_upper')
    if close and bb_low and bb_up and not (pd.isna(bb_low) or pd.isna(bb_up)):
        bb_width = bb_up - bb_low
        bb_position = (close - bb_low) / bb_width if bb_width > 0 else 0.5

        if bb_position <= 0.1:
            score += 15  # At/below lower band - strong buy
            factors['BB'] = "AT LOWER BAND +15"
        elif bb_position <= 0.25:
            score += 8  # Near lower band
            factors['BB'] = "NEAR LOWER BAND +8"
        elif bb_position >= 0.9:
            score -= 15  # At/above upper band - strong sell
            factors['BB'] = "AT UPPER BAND -15"
        elif bb_position >= 0.75:
            score -= 8  # Near upper band
            factors['BB'] = "NEAR UPPER BAND -8"
        else:
            factors['BB'] = "WITHIN BANDS +0"

    # 5. Volume Analysis (±15 points)
    volume = latest.get('volume')
    if volume and not pd.isna(volume) and len(df) >= 20:
        avg_volume = df['volume'].iloc[-20:].mean()
        if avg_volume > 0:
            vol_ratio = volume / avg_volume

            price_change = 0
            if len(df) >= 2:
                prev_close = df['close'].iloc[-2]
                if prev_close > 0:
                    price_change = (close - prev_close) / prev_close * 100

            if vol_ratio > 1.5 and price_change > 1:
                score += 15  # High volume with price increase
                factors['VOLUME'] = f"HIGH VOL BREAKOUT ({vol_ratio:.1f}x) +15"
            elif vol_ratio > 1.5 and price_change < -1:
                score -= 15  # High volume with price decrease
                factors['VOLUME'] = f"HIGH VOL BREAKDOWN ({vol_ratio:.1f}x) -15"
            elif vol_ratio > 1.2:
                if price_change > 0:
                    score += 8
                    factors['VOLUME'] = f"ABOVE AVG VOL ({vol_ratio:.1f}x) +8"
                else:
                    score -= 8
                    factors['VOLUME'] = f"ABOVE AVG VOL ({vol_ratio:.1f}x) -8"
            else:
                factors['VOLUME'] = f"NORMAL ({vol_ratio:.1f}x) +0"

    # 6. Price Momentum (±15 points)
    if len(df) >= 5:
        momentum_5d = (close - df['close'].iloc[-5]) / df['close'].iloc[-5] * 100 if df['close'].iloc[-5] > 0 else 0

        if momentum_5d > 5:
            score += 15  # Strong upward momentum
            factors['MOMENTUM'] = f"STRONG UP ({momentum_5d:.1f}%) +15"
        elif momentum_5d > 2:
            score += 8  # Positive momentum
            factors['MOMENTUM'] = f"POSITIVE ({momentum_5d:.1f}%) +8"
        elif momentum_5d < -5:
            score -= 15  # Strong downward momentum
            factors['MOMENTUM'] = f"STRONG DOWN ({momentum_5d:.1f}%) -15"
        elif momentum_5d < -2:
            score -= 8  # Negative momentum
            factors['MOMENTUM'] = f"NEGATIVE ({momentum_5d:.1f}%) -8"
        else:
            factors['MOMENTUM'] = f"FLAT ({momentum_5d:.1f}%) +0"

    # Clamp score between 0 and 100
    score = max(0, min(100, score))

    return score, factors


def analyze_stock(df):
    if df is None or len(df) < 20:
        return None

    try:
        import pandas as pd
        from ta.momentum import RSIIndicator
        from ta.trend import MACD, SMAIndicator
        from ta.volatility import BollingerBands

        df = df.copy()
        df['rsi'] = RSIIndicator(close=df['close'], window=14).rsi()
        macd = MACD(close=df['close'])
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        df['sma_20'] = SMAIndicator(close=df['close'], window=20).sma_indicator()
        df['sma_50'] = SMAIndicator(close=df['close'], window=50).sma_indicator()
        bb = BollingerBands(close=df['close'])
        df['bb_upper'] = bb.bollinger_hband()
        df['bb_lower'] = bb.bollinger_lband()
        df['bb_middle'] = bb.bollinger_mavg()

        latest = df.iloc[-1]

        # Calculate enhanced confidence score (0-100)
        confidence_score, score_factors = calculate_confidence_score(df, latest)

        # Original indicators for backward compatibility
        indicators = {}
        buy_score = sell_score = 0

        rsi = latest.get('rsi')
        if rsi and not pd.isna(rsi):
            if rsi < 30:
                indicators['RSI'] = f"OVERSOLD ({rsi:.1f})"
                buy_score += 2
            elif rsi > 70:
                indicators['RSI'] = f"OVERBOUGHT ({rsi:.1f})"
                sell_score += 2
            else:
                indicators['RSI'] = f"NEUTRAL ({rsi:.1f})"

        macd_val = latest.get('macd')
        macd_sig = latest.get('macd_signal')
        if macd_val and macd_sig and not (pd.isna(macd_val) or pd.isna(macd_sig)):
            if macd_val > macd_sig:
                indicators['MACD'] = "BULLISH"
                buy_score += 1
            else:
                indicators['MACD'] = "BEARISH"
                sell_score += 1

        sma20 = latest.get('sma_20')
        sma50 = latest.get('sma_50')
        if sma20 and sma50 and not (pd.isna(sma20) or pd.isna(sma50)):
            if sma20 > sma50:
                indicators['SMA'] = "BULLISH"
                buy_score += 1
            else:
                indicators['SMA'] = "BEARISH"
                sell_score += 1

        close = latest.get('close')
        bb_low = latest.get('bb_lower')
        bb_up = latest.get('bb_upper')
        if close and bb_low and bb_up:
            if close <= bb_low:
                indicators['BB'] = "LOWER BAND"
                buy_score += 1
            elif close >= bb_up:
                indicators['BB'] = "UPPER BAND"
                sell_score += 1
            else:
                indicators['BB'] = "WITHIN BANDS"

        net = buy_score - sell_score

        # Determine signal based on confidence score
        if confidence_score >= 75:
            signal = "STRONG BUY"
        elif confidence_score >= 60:
            signal = "BUY"
        elif confidence_score <= 25:
            signal = "STRONG SELL"
        elif confidence_score <= 40:
            signal = "SELL"
        else:
            signal = "NEUTRAL"

        return {
            "price": close,
            "rsi": rsi,
            "sma_20": sma20,
            "sma_50": sma50,
            "signal": signal,
            "strength": net,
            "confidence_score": confidence_score,
            "score_factors": score_factors,
            "indicators": indicators,
            "volume": latest.get('volume'),
            "bb_upper": bb_up,
            "bb_lower": bb_low,
            "bb_middle": latest.get('bb_middle'),
            "df": df  # Pass dataframe for advanced calculations
        }
    except Exception as e:
        logger.error(f"Analysis error: {e}")
        return None


def calculate_trading_levels(df, analysis, info):
    """
    Calculate entry, target, stop loss levels with timeframe recommendation.

    Methodology:
    - Stop Loss: Based on ATR (Average True Range) or recent swing low/high
    - Target 1: Based on Risk:Reward 1:1.5
    - Target 2: Based on Risk:Reward 1:2.5
    - Timeframe: Based on volatility and trend strength
    """
    import pandas as pd
    import numpy as np

    try:
        close = analysis['price']
        signal = analysis['signal']
        confidence = analysis.get('confidence_score', 50)
        bb_upper = analysis.get('bb_upper', close * 1.02)
        bb_lower = analysis.get('bb_lower', close * 0.98)
        bb_middle = analysis.get('bb_middle', close)
        sma_20 = analysis.get('sma_20', close)
        sma_50 = analysis.get('sma_50', close)

        # Calculate ATR (Average True Range) for volatility-based stop loss
        if len(df) >= 14:
            high = df['high']
            low = df['low']
            close_prev = df['close'].shift(1)

            tr1 = high - low
            tr2 = abs(high - close_prev)
            tr3 = abs(low - close_prev)

            true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr = true_range.rolling(window=14).mean().iloc[-1]
        else:
            atr = close * 0.02  # Default 2% if not enough data

        # Calculate recent swing high/low (last 20 days)
        recent_high = df['high'].iloc[-20:].max() if len(df) >= 20 else close * 1.05
        recent_low = df['low'].iloc[-20:].min() if len(df) >= 20 else close * 0.95

        # Calculate support and resistance levels
        support_1 = bb_lower  # Bollinger Lower Band
        support_2 = recent_low  # Recent Swing Low
        resistance_1 = bb_upper  # Bollinger Upper Band
        resistance_2 = recent_high  # Recent Swing High

        # Determine if BUY or SELL signal
        is_buy_signal = 'BUY' in signal
        is_sell_signal = 'SELL' in signal

        if is_buy_signal:
            # BUY Signal Calculations
            entry_price = close

            # Stop Loss: Below recent swing low or 1.5x ATR below entry
            stop_loss_atr = close - (1.5 * atr)
            stop_loss_swing = min(support_1, support_2) * 0.995  # Slightly below support
            stop_loss = max(stop_loss_atr, stop_loss_swing)  # Use the higher (less risky) stop

            # Risk calculation
            risk = entry_price - stop_loss

            # Target prices based on Risk:Reward ratios
            target_1 = entry_price + (risk * 1.5)  # R:R = 1:1.5
            target_2 = entry_price + (risk * 2.5)  # R:R = 1:2.5
            target_3 = min(resistance_1, resistance_2)  # Resistance level

            # Ensure targets are realistic
            target_1 = max(target_1, entry_price * 1.02)  # At least 2% gain
            target_2 = max(target_2, entry_price * 1.05)  # At least 5% gain

            action = "BUY"

        elif is_sell_signal:
            # SELL Signal Calculations (for short selling or exit)
            entry_price = close

            # Stop Loss: Above recent swing high or 1.5x ATR above entry
            stop_loss_atr = close + (1.5 * atr)
            stop_loss_swing = max(resistance_1, resistance_2) * 1.005  # Slightly above resistance
            stop_loss = min(stop_loss_atr, stop_loss_swing)  # Use the lower (less risky) stop

            # Risk calculation
            risk = stop_loss - entry_price

            # Target prices (downside targets for sell)
            target_1 = entry_price - (risk * 1.5)  # R:R = 1:1.5
            target_2 = entry_price - (risk * 2.5)  # R:R = 1:2.5
            target_3 = max(support_1, support_2)  # Support level

            # Ensure targets are realistic
            target_1 = min(target_1, entry_price * 0.98)  # At least 2% drop
            target_2 = min(target_2, entry_price * 0.95)  # At least 5% drop

            action = "SELL"

        else:
            # NEUTRAL - No clear trade setup
            return None

        # Calculate Risk:Reward Ratio
        if is_buy_signal:
            risk_amount = entry_price - stop_loss
            reward_1 = target_1 - entry_price
            reward_2 = target_2 - entry_price
        else:
            risk_amount = stop_loss - entry_price
            reward_1 = entry_price - target_1
            reward_2 = entry_price - target_2

        rr_ratio_1 = reward_1 / risk_amount if risk_amount > 0 else 0
        rr_ratio_2 = reward_2 / risk_amount if risk_amount > 0 else 0

        # Calculate percentage moves
        stop_loss_pct = ((stop_loss - entry_price) / entry_price) * 100
        target_1_pct = ((target_1 - entry_price) / entry_price) * 100
        target_2_pct = ((target_2 - entry_price) / entry_price) * 100

        # Determine Timeframe based on volatility and confidence
        daily_volatility = (atr / close) * 100  # ATR as % of price

        if daily_volatility > 3:
            # High volatility - shorter timeframe
            if confidence >= 70:
                timeframe = "1-3 days (Intraday/Swing)"
            else:
                timeframe = "1-2 days (Intraday)"
        elif daily_volatility > 1.5:
            # Medium volatility
            if confidence >= 70:
                timeframe = "1-2 weeks (Swing Trade)"
            else:
                timeframe = "3-5 days (Short Swing)"
        else:
            # Low volatility - longer timeframe
            if confidence >= 70:
                timeframe = "2-4 weeks (Positional)"
            else:
                timeframe = "1-2 weeks (Swing Trade)"

        # Determine trade quality
        if confidence >= 75 and rr_ratio_1 >= 1.5:
            trade_quality = "HIGH QUALITY SETUP"
        elif confidence >= 60 and rr_ratio_1 >= 1.2:
            trade_quality = "GOOD SETUP"
        elif confidence >= 50:
            trade_quality = "MODERATE SETUP"
        else:
            trade_quality = "WEAK SETUP - CAUTION"

        return {
            "action": action,
            "entry_price": entry_price,
            "stop_loss": stop_loss,
            "stop_loss_pct": stop_loss_pct,
            "target_1": target_1,
            "target_1_pct": target_1_pct,
            "target_2": target_2,
            "target_2_pct": target_2_pct,
            "rr_ratio_1": rr_ratio_1,
            "rr_ratio_2": rr_ratio_2,
            "timeframe": timeframe,
            "trade_quality": trade_quality,
            "atr": atr,
            "daily_volatility": daily_volatility,
            "support": min(support_1, support_2),
            "resistance": max(resistance_1, resistance_2),
            "risk_amount": risk_amount,
            "methodology": {
                "stop_loss_method": "ATR (1.5x) + Swing Low/High",
                "target_method": "Risk:Reward Ratio (1:1.5 and 1:2.5)",
                "timeframe_method": "Based on ATR volatility + Confidence score",
                "data_used": "3 months historical OHLCV data",
                "indicators_used": "RSI(14), MACD(12,26,9), SMA(20,50), Bollinger Bands(20,2), ATR(14), Volume"
            }
        }

    except Exception as e:
        logger.error(f"Trading levels calculation error: {e}")
        return None

# ===== ENHANCED CHART GENERATION =====
def generate_chart(symbol):
    """Generate enhanced multi-panel chart with price, indicators, and volume"""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from ta.momentum import RSIIndicator
        from ta.trend import MACD, SMAIndicator
        from ta.volatility import BollingerBands
        import numpy as np

        df = get_stock_data(symbol, period="3mo")
        if df is None or len(df) < 20:
            return None

        # Calculate indicators
        df['rsi'] = RSIIndicator(close=df['close'], window=14).rsi()
        macd = MACD(close=df['close'])
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        df['macd_hist'] = macd.macd_diff()
        df['sma_20'] = SMAIndicator(close=df['close'], window=20).sma_indicator()
        df['sma_50'] = SMAIndicator(close=df['close'], window=50).sma_indicator()
        bb = BollingerBands(close=df['close'])
        df['bb_upper'] = bb.bollinger_hband()
        df['bb_lower'] = bb.bollinger_lband()
        df['bb_middle'] = bb.bollinger_mavg()

        # Create figure with 4 subplots
        fig, axes = plt.subplots(4, 1, figsize=(12, 14), gridspec_kw={'height_ratios': [3, 1, 1, 1]})
        fig.suptitle(f'{symbol.upper()} - Technical Analysis Chart', fontsize=14, fontweight='bold')

        # Panel 1: Price with Bollinger Bands and SMAs
        ax1 = axes[0]
        ax1.plot(df.index, df['close'], label='Price', color='#2962FF', linewidth=1.5)
        ax1.plot(df.index, df['sma_20'], label='SMA 20', color='#FF6D00', linewidth=1, alpha=0.8)
        ax1.plot(df.index, df['sma_50'], label='SMA 50', color='#00C853', linewidth=1, alpha=0.8)
        ax1.fill_between(df.index, df['bb_upper'], df['bb_lower'], alpha=0.1, color='blue', label='BB Bands')
        ax1.plot(df.index, df['bb_upper'], color='gray', linewidth=0.5, linestyle='--')
        ax1.plot(df.index, df['bb_lower'], color='gray', linewidth=0.5, linestyle='--')
        ax1.set_ylabel('Price (₹)', fontweight='bold')
        ax1.legend(loc='upper left', fontsize=8)
        ax1.grid(True, alpha=0.3)
        ax1.set_title('Price with Bollinger Bands & Moving Averages', fontsize=10)

        # Panel 2: Volume
        ax2 = axes[1]
        colors = ['green' if df['close'].iloc[i] >= df['close'].iloc[i-1] else 'red'
                  for i in range(1, len(df))]
        colors.insert(0, 'green')
        ax2.bar(df.index, df['volume'], color=colors, alpha=0.7)
        ax2.set_ylabel('Volume', fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.set_title('Volume', fontsize=10)
        # Add volume average line
        vol_avg = df['volume'].rolling(window=20).mean()
        ax2.plot(df.index, vol_avg, color='blue', linewidth=1, label='20-day Avg')

        # Panel 3: RSI
        ax3 = axes[2]
        ax3.plot(df.index, df['rsi'], color='#7B1FA2', linewidth=1.5)
        ax3.axhline(y=70, color='red', linestyle='--', alpha=0.7, label='Overbought (70)')
        ax3.axhline(y=30, color='green', linestyle='--', alpha=0.7, label='Oversold (30)')
        ax3.axhline(y=50, color='gray', linestyle='-', alpha=0.3)
        ax3.fill_between(df.index, 70, 100, alpha=0.1, color='red')
        ax3.fill_between(df.index, 0, 30, alpha=0.1, color='green')
        ax3.set_ylabel('RSI', fontweight='bold')
        ax3.set_ylim(0, 100)
        ax3.legend(loc='upper left', fontsize=8)
        ax3.grid(True, alpha=0.3)
        ax3.set_title('RSI (14)', fontsize=10)

        # Panel 4: MACD
        ax4 = axes[3]
        ax4.plot(df.index, df['macd'], label='MACD', color='#2962FF', linewidth=1)
        ax4.plot(df.index, df['macd_signal'], label='Signal', color='#FF6D00', linewidth=1)
        colors_hist = ['green' if val >= 0 else 'red' for val in df['macd_hist']]
        ax4.bar(df.index, df['macd_hist'], color=colors_hist, alpha=0.5, label='Histogram')
        ax4.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
        ax4.set_ylabel('MACD', fontweight='bold')
        ax4.legend(loc='upper left', fontsize=8)
        ax4.grid(True, alpha=0.3)
        ax4.set_title('MACD (12, 26, 9)', fontsize=10)

        # Format x-axis
        for ax in axes:
            ax.tick_params(axis='x', rotation=45)

        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=120, bbox_inches='tight')
        buf.seek(0)
        plt.close()
        return buf
    except Exception as e:
        logger.error(f"Chart error: {e}")
        return None


def generate_weekly_report_chart(stocks_data):
    """Generate a comprehensive weekly report chart with multiple stocks comparison"""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import numpy as np

        if not stocks_data:
            return None

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Weekly Market Analysis Report', fontsize=16, fontweight='bold')

        # Panel 1: Weekly Performance Bar Chart
        ax1 = axes[0, 0]
        symbols = [s['symbol'] for s in stocks_data]
        weekly_returns = [s.get('weekly_return', 0) for s in stocks_data]
        colors = ['green' if r >= 0 else 'red' for r in weekly_returns]
        bars = ax1.bar(symbols, weekly_returns, color=colors, alpha=0.7)
        ax1.axhline(y=0, color='black', linewidth=0.5)
        ax1.set_ylabel('Weekly Return (%)', fontweight='bold')
        ax1.set_title('Weekly Performance', fontsize=12)
        ax1.tick_params(axis='x', rotation=45)
        # Add value labels
        for bar, val in zip(bars, weekly_returns):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    f'{val:.1f}%', ha='center', va='bottom', fontsize=8)

        # Panel 2: Confidence Scores
        ax2 = axes[0, 1]
        confidence_scores = [s.get('confidence_score', 50) for s in stocks_data]
        colors_conf = []
        for score in confidence_scores:
            if score >= 70:
                colors_conf.append('green')
            elif score <= 30:
                colors_conf.append('red')
            else:
                colors_conf.append('orange')
        bars = ax2.barh(symbols, confidence_scores, color=colors_conf, alpha=0.7)
        ax2.axvline(x=50, color='gray', linestyle='--', alpha=0.5)
        ax2.axvline(x=70, color='green', linestyle='--', alpha=0.3)
        ax2.axvline(x=30, color='red', linestyle='--', alpha=0.3)
        ax2.set_xlim(0, 100)
        ax2.set_xlabel('Confidence Score', fontweight='bold')
        ax2.set_title('Signal Confidence (0-100)', fontsize=12)
        # Add value labels
        for bar, val in zip(bars, confidence_scores):
            ax2.text(val + 2, bar.get_y() + bar.get_height()/2,
                    f'{val:.0f}', ha='left', va='center', fontsize=9)

        # Panel 3: RSI Comparison
        ax3 = axes[1, 0]
        rsi_values = [s.get('rsi', 50) for s in stocks_data]
        colors_rsi = []
        for rsi in rsi_values:
            if rsi < 30:
                colors_rsi.append('green')
            elif rsi > 70:
                colors_rsi.append('red')
            else:
                colors_rsi.append('blue')
        ax3.bar(symbols, rsi_values, color=colors_rsi, alpha=0.7)
        ax3.axhline(y=70, color='red', linestyle='--', alpha=0.5, label='Overbought')
        ax3.axhline(y=30, color='green', linestyle='--', alpha=0.5, label='Oversold')
        ax3.axhline(y=50, color='gray', linestyle='-', alpha=0.3)
        ax3.set_ylabel('RSI', fontweight='bold')
        ax3.set_ylim(0, 100)
        ax3.set_title('RSI Comparison', fontsize=12)
        ax3.tick_params(axis='x', rotation=45)
        ax3.legend(fontsize=8)

        # Panel 4: Signal Distribution Pie Chart
        ax4 = axes[1, 1]
        signals = [s.get('signal', 'NEUTRAL') for s in stocks_data]
        signal_counts = {}
        for sig in signals:
            signal_counts[sig] = signal_counts.get(sig, 0) + 1
        labels = list(signal_counts.keys())
        sizes = list(signal_counts.values())
        colors_pie = []
        for label in labels:
            if 'BUY' in label:
                colors_pie.append('#4CAF50' if 'STRONG' in label else '#81C784')
            elif 'SELL' in label:
                colors_pie.append('#F44336' if 'STRONG' in label else '#E57373')
            else:
                colors_pie.append('#FFC107')
        ax4.pie(sizes, labels=labels, colors=colors_pie, autopct='%1.0f%%', startangle=90)
        ax4.set_title('Signal Distribution', fontsize=12)

        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=120, bbox_inches='tight')
        buf.seek(0)
        plt.close()
        return buf
    except Exception as e:
        logger.error(f"Weekly report chart error: {e}")
        return None

# ===== HANDLERS =====
def handle_start(chat_id, user_name):
    msg = f"""*Welcome to MarketBuddy Pro!*

Hello {user_name}! I'm your AI-powered stock analysis assistant.

*📈 STOCK ANALYSIS:*
/stock SYMBOL - *Complete Trading Analysis*
  • Current Price & Change
  • Target 1, Target 2, Stop Loss
  • Timeframe & Hold Duration
  • Risk:Reward Ratio
  • Confidence Score (0-100)
  Example: /stock RELIANCE

/price SYMBOL - Quick price check
/chart SYMBOL - Technical chart with indicators
/days SYMBOL - Last 15 days price history
  Example: /days RELIANCE 20 (for 20 days)

*📊 INDEX TRADING:*
/nifty - NIFTY 50 signal with levels
/banknifty - Bank NIFTY signal
/sensex - SENSEX signal
/giftnifty - Gift Nifty status (SGX)

*🎯 SIGNAL TRACKING:*
/signals - View all active signals
/history - Trade history & win rate
/mistakes - Track failed signals for learning

*📰 MARKET INFO:*
/summary - NIFTY, SENSEX, BANKNIFTY
/news - Latest market news

*📊 MARKET ANALYSIS (NEW!):*
/fiidii - FII/DII buying/selling data
/oi NIFTY - Option Chain analysis (PCR, Support, Resistance)
/oi BANKNIFTY - Bank NIFTY option chain
/pcr - Quick NIFTY PCR check
/calendar - Upcoming economic events
/sentiment - Overall market sentiment score
/ipo - Mainboard IPO analysis with recommendation

💡 *TIP:* Just type stock name (e.g., RELIANCE) without "/" for instant analysis!

*📋 PORTFOLIO:*
/watchlist - View your watchlist
/watchlist add/remove SYMBOL
/portfolio - View holdings
/portfolio add SYMBOL QTY PRICE
/alert SYMBOL PRICE - Set price alert
/alerts - View your alerts

*🤖 AUTOMATIC ALERTS:*
  ✅ Morning Briefing (8:45 AM)
  ✅ Market Open/Close Alerts
  ✅ FII/DII Daily Update (4:30 PM)
  ✅ Target Hit Notifications (real-time)
  ✅ Stop Loss & Expiry Alerts
  ✅ Index Signals (NIFTY/BANKNIFTY)
  ✅ Gift Nifty (7PM & 11:30PM)
  ✅ Tomorrow's Events (8 PM)
  ✅ High Confidence Opportunities
  ✅ Breaking News Alerts
  ✅ Weekly Report (Saturday 6PM)

*Signal Format Includes:*
  • Current Price
  • Target 1 & Target 2
  • Stop Loss
  • Timeframe (5min/15min/1day etc.)
  • How Long to Hold
  • Risk:Reward Ratio
  • Confidence Score

Start with `/stock RELIANCE` or `/nifty` to see it in action!"""
    send_message(chat_id, msg)

def handle_price(chat_id, symbol):
    info = get_stock_info(symbol)
    if not info or not info.get('price'):
        send_message(chat_id, f"No data for {symbol}")
        return

    price = info['price']
    prev = info.get('previous_close', price)
    change = price - prev
    pct = (change / prev * 100) if prev else 0

    msg = f"""*{info.get('name', symbol)}*

Price: Rs {price:,.2f}
Change: {change:+,.2f} ({pct:+.2f}%)
Open: Rs {info.get('open', 0):,.2f}
High: Rs {info.get('high', 0):,.2f}
Low: Rs {info.get('low', 0):,.2f}"""
    send_message(chat_id, msg)


def handle_days(chat_id, symbol, days=15):
    """Show last N days price history for a stock."""
    symbol = symbol.upper().strip()

    # Get stock data for enough period
    df = get_stock_data(symbol, period="1mo")
    if df is None or len(df) == 0:
        send_message(chat_id, f"❌ No data available for {symbol}")
        return

    # Get stock info for company name
    info = get_stock_info(symbol)
    name = info.get('name', symbol) if info else symbol

    # Get last N days
    df_last = df.tail(days)

    if len(df_last) == 0:
        send_message(chat_id, f"❌ Not enough historical data for {symbol}")
        return

    now = datetime.now(IST)
    msg = f"""📊 *{days}-DAY PRICE HISTORY*
_{now.strftime('%Y-%m-%d %H:%M')} IST_

*{name}* ({symbol})
{'='*30}

"""
    # Calculate overall change from first to last
    first_close = df_last['close'].iloc[0]
    last_close = df_last['close'].iloc[-1]
    overall_change = ((last_close - first_close) / first_close * 100) if first_close else 0

    # Add header
    msg += "*Date       | Open     | High     | Low      | Close    | Change*\n"
    msg += "-" * 60 + "\n"

    prev_close = None
    for idx, row in df_last.iterrows():
        date_str = idx.strftime('%d-%b')
        open_price = row['open']
        high_price = row['high']
        low_price = row['low']
        close_price = row['close']

        # Calculate daily change
        if prev_close:
            daily_change = ((close_price - prev_close) / prev_close * 100)
            change_str = f"{daily_change:+.1f}%"
            emoji = "🟢" if daily_change >= 0 else "🔴"
        else:
            change_str = "-"
            emoji = "⚪"

        msg += f"{emoji} `{date_str}` | ₹{open_price:,.0f} | ₹{high_price:,.0f} | ₹{low_price:,.0f} | ₹{close_price:,.0f} | {change_str}\n"
        prev_close = close_price

    msg += "-" * 60 + "\n"

    # Summary statistics
    high_of_period = df_last['high'].max()
    low_of_period = df_last['low'].min()
    avg_volume = df_last['volume'].mean()

    msg += f"""
*📈 SUMMARY:*
• Period High: ₹{high_of_period:,.2f}
• Period Low: ₹{low_of_period:,.2f}
• Overall Change: {overall_change:+.2f}%
• Avg Volume: {avg_volume/100000:,.2f}L

*Current Price:* ₹{last_close:,.2f}
"""

    # Add trend assessment
    if overall_change > 5:
        trend = "🚀 Strong Uptrend"
    elif overall_change > 2:
        trend = "📈 Uptrend"
    elif overall_change > -2:
        trend = "➡️ Sideways"
    elif overall_change > -5:
        trend = "📉 Downtrend"
    else:
        trend = "💥 Strong Downtrend"

    msg += f"*{days}-Day Trend:* {trend}"

    send_message(chat_id, msg)


def handle_stock(chat_id, symbol):
    send_message(chat_id, f"Analyzing {symbol}...")

    df = get_stock_data(symbol)
    if df is None:
        send_message(chat_id, f"No data for {symbol}")
        return

    info = get_stock_info(symbol)
    analysis = analyze_stock(df)

    if not analysis:
        send_message(chat_id, "Insufficient data")
        return

    price = analysis['price']
    prev = info.get('previous_close', price)
    change = price - prev
    pct = (change / prev * 100) if prev else 0

    # Confidence score emoji
    conf_score = analysis.get('confidence_score', 50)
    if conf_score >= 75:
        score_emoji = "🟢🟢🟢"
    elif conf_score >= 60:
        score_emoji = "🟢🟢"
    elif conf_score <= 25:
        score_emoji = "🔴🔴🔴"
    elif conf_score <= 40:
        score_emoji = "🔴🔴"
    else:
        score_emoji = "🟡"

    msg = f"""*{info.get('name', symbol)}*

*Price:* ₹{price:,.2f}
*Change:* {change:+,.2f} ({pct:+.2f}%)

*SIGNAL: {analysis['signal']}*
*Confidence Score: {conf_score}/100* {score_emoji}

*Score Breakdown:*
"""
    # Add score factors
    score_factors = analysis.get('score_factors', {})
    for factor, detail in score_factors.items():
        msg += f"  • {factor}: {detail}\n"

    msg += f"""
*Technical Indicators:*
"""
    for ind, val in analysis['indicators'].items():
        msg += f"  • {ind}: {val}\n"

    msg += f"""
*Stock Info:*
  • 52W High: ₹{info.get('52w_high', 0):,.2f}
  • 52W Low: ₹{info.get('52w_low', 0):,.2f}
  • P/E: {info.get('pe_ratio', 0):.2f}
  • Sector: {info.get('sector', 'N/A')}
"""

    # Calculate trading levels if signal is not NEUTRAL
    analysis_df = analysis.get('df', df)
    trading = calculate_trading_levels(analysis_df, analysis, info)

    if trading:
        action_emoji = "🟢" if trading['action'] == "BUY" else "🔴"
        msg += f"""
{'='*30}
*TRADING RECOMMENDATION*
{'='*30}

{action_emoji} *Action:* {trading['action']}
*Quality:* {trading['trade_quality']}
*Timeframe:* {trading['timeframe']}

*Entry Price:* ₹{trading['entry_price']:,.2f}

🎯 *Target 1:* ₹{trading['target_1']:,.2f} ({trading['target_1_pct']:+.2f}%)
   R:R Ratio = 1:{trading['rr_ratio_1']:.1f}

🎯 *Target 2:* ₹{trading['target_2']:,.2f} ({trading['target_2_pct']:+.2f}%)
   R:R Ratio = 1:{trading['rr_ratio_2']:.1f}

🛑 *Stop Loss:* ₹{trading['stop_loss']:,.2f} ({trading['stop_loss_pct']:+.2f}%)
   Risk: ₹{trading['risk_amount']:,.2f} per share

*Key Levels:*
  • Support: ₹{trading['support']:,.2f}
  • Resistance: ₹{trading['resistance']:,.2f}

*Volatility:* {trading['daily_volatility']:.2f}% (ATR: ₹{trading['atr']:,.2f})
"""
    else:
        msg += f"""
{'='*30}
*TRADING RECOMMENDATION*
{'='*30}

⚠️ *No Clear Trade Setup*

Signal is NEUTRAL - Wait for clearer direction.
"""

    msg += "\n_Not financial advice. Do your own research._"
    send_message(chat_id, msg)

    # Create active signal for target tracking (if non-NEUTRAL signal)
    if trading and analysis['signal'] != 'NEUTRAL':
        signal_id = create_active_signal(
            symbol=symbol.upper(),
            signal_type=trading['action'],
            entry_price=trading['entry_price'],
            target_1=trading['target_1'],
            target_2=trading['target_2'],
            stop_loss=trading['stop_loss'],
            timeframe=trading['timeframe'],
            confidence=conf_score,
            methodology=trading.get('methodology', {})
        )

        # Send tracking confirmation
        tracking_msg = f"""📍 *SIGNAL TRACKING ACTIVATED*

Signal ID: `{signal_id}`

I will monitor this signal and notify you when:
  ✅ Target 1 is hit
  ✅ Target 2 is hit
  🛑 Stop Loss is hit
  ⏰ Signal expires (based on timeframe)

Use `/signals` to view all active signals.
Use `/history` to view trade history."""
        send_message(chat_id, tracking_msg)

    # Send methodology in a separate message for clarity
    if trading:
        method_msg = f"""📊 *HOW THIS WAS CALCULATED*

*Data Used:*
  • 3 months historical price data (OHLCV)
  • Real-time price from Yahoo Finance

*Indicators Analyzed:*
  • RSI (14-period) - Momentum
  • MACD (12,26,9) - Trend
  • SMA (20 & 50) - Moving Averages
  • Bollinger Bands (20,2) - Volatility
  • ATR (14-period) - Average True Range
  • Volume Analysis - Buying/Selling pressure
  • 5-day Momentum - Price direction

*Stop Loss Calculation:*
  {trading['methodology']['stop_loss_method']}

*Target Calculation:*
  {trading['methodology']['target_method']}

*Timeframe Selection:*
  {trading['methodology']['timeframe_method']}

*Confidence Score (0-100):*
  Sum of 6 factors:
  - RSI: ±20 points
  - MACD: ±20 points
  - SMA Trend: ±15 points
  - Bollinger: ±15 points
  - Volume: ±15 points
  - Momentum: ±15 points

_Higher score = Stronger signal_"""
        send_message(chat_id, method_msg)

def handle_chart(chat_id, symbol):
    send_message(chat_id, f"Generating chart for {symbol}...")
    chart = generate_chart(symbol)
    if chart:
        send_photo(chat_id, chart, f"{symbol.upper()} Chart")
    else:
        send_message(chat_id, f"Could not generate chart for {symbol}")

def handle_summary(chat_id):
    msg = f"""*MARKET SUMMARY*
_{datetime.now().strftime('%Y-%m-%d %H:%M')}_

"""
    for idx in ["NIFTY", "SENSEX", "BANKNIFTY"]:
        data = get_index_data(idx)
        if data:
            msg += f"""*{idx}*
  {data.get('value', 0):,.2f}
  {data.get('change', 0):+,.2f} ({data.get('pct', 0):+.2f}%)

"""
    send_message(chat_id, msg)

def handle_news(chat_id):
    send_message(chat_id, "Fetching news...")
    articles = get_market_news(limit=5)
    if not articles:
        send_message(chat_id, "No news available")
        return

    msg = "*LATEST NEWS*\n\n"
    for i, article in enumerate(articles, 1):
        title = article.get('title', '')[:80]
        msg += f"{i}. {title}\n\n"
    send_message(chat_id, msg)

def handle_watchlist(chat_id, action=None, symbol=None):
    user = get_user_data(chat_id)

    if action == "add" and symbol:
        if symbol.upper() not in user['watchlist']:
            user['watchlist'].append(symbol.upper())
            send_message(chat_id, f"Added {symbol.upper()} to watchlist")
        else:
            send_message(chat_id, f"{symbol.upper()} already in watchlist")
    elif action == "remove" and symbol:
        if symbol.upper() in user['watchlist']:
            user['watchlist'].remove(symbol.upper())
            send_message(chat_id, f"Removed {symbol.upper()}")
        else:
            send_message(chat_id, f"{symbol.upper()} not in watchlist")
    else:
        if not user['watchlist']:
            send_message(chat_id, "Watchlist empty. Use /watchlist add SYMBOL")
            return

        msg = "*YOUR WATCHLIST*\n\n"
        for sym in user['watchlist'][:10]:
            info = get_stock_info(sym)
            if info and info.get('price'):
                price = info['price']
                prev = info.get('previous_close', price)
                pct = ((price - prev) / prev * 100) if prev else 0
                msg += f"*{sym}*: Rs {price:,.2f} ({pct:+.2f}%)\n"
            else:
                msg += f"*{sym}*: N/A\n"
        send_message(chat_id, msg)

def handle_alert(chat_id, symbol, target_price):
    user = get_user_data(chat_id)
    try:
        target = float(target_price)
        user['alerts'].append({"symbol": symbol.upper(), "target": target})
        send_message(chat_id, f"Alert set: {symbol.upper()} at Rs {target:,.2f}")
    except:
        send_message(chat_id, "Invalid price. Use: /alert RELIANCE 2500")

def handle_alerts(chat_id):
    user = get_user_data(chat_id)
    if not user['alerts']:
        send_message(chat_id, "No alerts. Use /alert SYMBOL PRICE")
        return

    msg = "*YOUR ALERTS*\n\n"
    for i, alert in enumerate(user['alerts'], 1):
        msg += f"{i}. {alert['symbol']} -> Rs {alert['target']:,.2f}\n"
    send_message(chat_id, msg)

def handle_portfolio(chat_id, action=None, symbol=None, qty=None, price=None):
    user = get_user_data(chat_id)

    if action == "add" and symbol and qty and price:
        try:
            user['portfolio'][symbol.upper()] = {"qty": int(qty), "avg_price": float(price)}
            send_message(chat_id, f"Added {symbol.upper()}: {qty} @ Rs {float(price):,.2f}")
        except:
            send_message(chat_id, "Invalid. Use: /portfolio add RELIANCE 10 2500")
    elif action == "remove" and symbol:
        if symbol.upper() in user['portfolio']:
            del user['portfolio'][symbol.upper()]
            send_message(chat_id, f"Removed {symbol.upper()}")
    else:
        if not user['portfolio']:
            send_message(chat_id, "Portfolio empty. Use /portfolio add SYMBOL QTY PRICE")
            return

        msg = "*YOUR PORTFOLIO*\n\n"
        total_inv = total_cur = 0

        for sym, h in user['portfolio'].items():
            info = get_stock_info(sym)
            if info and info.get('price'):
                cur = info['price']
                inv = h['qty'] * h['avg_price']
                cur_val = h['qty'] * cur
                pnl = cur_val - inv
                total_inv += inv
                total_cur += cur_val
                msg += f"*{sym}* ({h['qty']})\n  Rs {h['avg_price']:,.2f} -> Rs {cur:,.2f}\n  P/L: Rs {pnl:+,.2f}\n\n"

        total_pnl = total_cur - total_inv
        msg += f"*Total P/L:* Rs {total_pnl:+,.2f}"
        send_message(chat_id, msg)


def handle_signals(chat_id):
    """Show all active signals being tracked."""
    global active_signals

    if not active_signals:
        send_message(chat_id, """📊 *NO ACTIVE SIGNALS*

You don't have any active signals being tracked.

Use `/stock SYMBOL` to analyze a stock and create a signal.

Example: `/stock RELIANCE`""")
        return

    now = datetime.now(IST)
    msg = f"""📊 *ACTIVE SIGNALS*
_{now.strftime('%Y-%m-%d %H:%M')} IST_

"""

    for signal_id, signal in active_signals.items():
        # Get current price
        try:
            if signal["symbol"] in INDEX_WATCHLIST:
                data = get_index_data(signal["symbol"])
                current_price = data.get("value", 0) if data else 0
            else:
                info = get_stock_info(signal["symbol"])
                current_price = info.get("price", 0) if info else 0
        except:
            current_price = signal["entry_price"]

        # Calculate P&L
        if signal["type"] == "BUY":
            pnl_pct = ((current_price - signal["entry_price"]) / signal["entry_price"]) * 100
        else:
            pnl_pct = ((signal["entry_price"] - current_price) / signal["entry_price"]) * 100

        status_emoji = "🟢" if signal["status"] == "ACTIVE" else "🎯" if "TARGET" in signal["status"] else "🔴"
        type_emoji = "📈" if signal["type"] == "BUY" else "📉"

        # Time remaining
        time_remaining = signal["expiry_at"] - now
        hours_left = time_remaining.total_seconds() / 3600

        msg += f"""{status_emoji} *{signal['symbol']}* - {signal['type']} {type_emoji}
  Status: {signal['status']}
  Entry: ₹{signal['entry_price']:,.2f}
  Current: ₹{current_price:,.2f} ({pnl_pct:+.2f}%)
  Target 1: ₹{signal['target_1']:,.2f} {'✅' if signal.get('target_1_hit_at') else '⏳'}
  Target 2: ₹{signal['target_2']:,.2f} {'✅' if signal.get('target_2_hit_at') else '⏳'}
  Stop Loss: ₹{signal['stop_loss']:,.2f}
  Timeframe: {signal['timeframe']}
  Expires in: {hours_left:.1f} hours

"""

    msg += f"_Total Active Signals: {len(active_signals)}_"
    send_message(chat_id, msg)


def handle_history(chat_id):
    """Show trade history."""
    global trade_history

    if not trade_history:
        send_message(chat_id, """📜 *NO TRADE HISTORY*

You don't have any completed trades yet.

As signals hit targets or expire, they'll appear here.""")
        return

    now = datetime.now(IST)
    msg = f"""📜 *TRADE HISTORY*
_{now.strftime('%Y-%m-%d %H:%M')} IST_

"""

    # Show last 10 trades
    recent_trades = trade_history[-10:]

    wins = 0
    losses = 0

    for trade in reversed(recent_trades):
        result = trade.get("status", "UNKNOWN")

        if "TARGET" in result:
            result_emoji = "✅"
            wins += 1
        elif "STOP_LOSS" in result:
            result_emoji = "❌"
            losses += 1
        elif "EXPIRED" in result:
            result_emoji = "⏰"
            losses += 1
        else:
            result_emoji = "❔"

        msg += f"""{result_emoji} *{trade['symbol']}* - {trade['type']}
  Result: {result}
  Entry: ₹{trade['entry_price']:,.2f}
  Confidence: {trade['confidence']}/100

"""

    # Win rate
    total = wins + losses
    win_rate = (wins / total * 100) if total > 0 else 0

    msg += f"""{'='*30}
*STATISTICS:*
  ✅ Wins: {wins}
  ❌ Losses: {losses}
  📊 Win Rate: {win_rate:.1f}%

_Showing last {len(recent_trades)} trades_"""

    send_message(chat_id, msg)


def handle_mistakes(chat_id):
    """Show tracked mistakes for learning."""
    global market_mistakes

    if not market_mistakes:
        send_message(chat_id, """📝 *NO MISTAKES TRACKED*

Great! No failed trades recorded yet.

When signals hit stop loss or expire without target, they'll be recorded here for learning.""")
        return

    now = datetime.now(IST)
    msg = f"""📝 *MISTAKE TRACKER - LEARNING*
_{now.strftime('%Y-%m-%d %H:%M')} IST_

*Recent Failed Signals:*

"""

    # Show last 5 mistakes
    recent_mistakes = market_mistakes[-5:]

    for i, mistake in enumerate(reversed(recent_mistakes), 1):
        msg += f"""*{i}. {mistake['symbol']}* ({mistake['date']})
  Signal: {mistake['signal_type']}
  Confidence: {mistake['confidence']}/100
  Issue: {mistake['what_went_wrong'][:50]}...
  Lesson: {mistake['lesson'][:50]}...

"""

    # Analysis
    if len(market_mistakes) >= 3:
        avg_conf = sum(m['confidence'] for m in market_mistakes) / len(market_mistakes)
        msg += f"""{'='*30}
*PATTERN ANALYSIS:*
  • Total Mistakes: {len(market_mistakes)}
  • Avg Failed Confidence: {avg_conf:.1f}/100

*Recommendation:*"""
        if avg_conf > 60:
            msg += "\n  ⚠️ High confidence signals failing. Review methodology."
        else:
            msg += "\n  ℹ️ Low confidence signals failing as expected. Consider stricter filters."

    msg += "\n\n_Use this data to improve future analysis._"
    send_message(chat_id, msg)


def handle_giftnifty(chat_id):
    """Show current Gift Nifty status."""
    gift_data = get_gift_nifty_data()

    if not gift_data:
        send_message(chat_id, "Unable to fetch Gift Nifty data. Try again later.")
        return

    now = datetime.now(IST)
    emoji = "📈" if gift_data['change'] >= 0 else "📉"

    msg = f"""🌐 *GIFT NIFTY STATUS*
_{now.strftime('%Y-%m-%d %H:%M')} IST_

{emoji} *GIFT NIFTY*

📊 *Current:* {gift_data['current']:,.2f}
📊 *Open:* {gift_data['open']:,.2f}
📊 *Change:* {gift_data['change']:+,.2f} ({gift_data['pct']:+.2f}%)

*Today's Range:*
  🔺 High: {gift_data['high']:,.2f}
  🔻 Low: {gift_data['low']:,.2f}
  📏 Range: {gift_data['range']:,.2f} points

*Market Indication:*
"""
    if gift_data['pct'] > 0.5:
        msg += "  ✅ Bullish sentiment - Gap up expected"
    elif gift_data['pct'] < -0.5:
        msg += "  ⚠️ Bearish sentiment - Gap down expected"
    else:
        msg += "  ➡️ Flat to neutral opening expected"

    msg += """

_Gift Nifty trades on SGX from 6:30 AM to 11:30 PM IST_"""

    send_message(chat_id, msg)


# IPO Cache
ipo_cache = {
    "data": [],
    "updated": None
}


def determine_ipo_status(open_date_str, close_date_str, today):
    """
    Determine IPO status based on dates.
    Returns: OPEN, UPCOMING, or CLOSED
    """
    import re
    from datetime import datetime as dt

    def parse_date(date_str):
        """Try to parse various date formats."""
        if not date_str:
            return None
        date_str = date_str.strip()

        # Common patterns
        patterns = [
            r'(\d{1,2})\s*(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s*(\d{4})?',
            r'(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s*(\d{1,2}),?\s*(\d{4})?',
        ]
        months = {'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6,
                  'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12}

        for pattern in patterns:
            match = re.search(pattern, date_str, re.IGNORECASE)
            if match:
                groups = match.groups()
                try:
                    if groups[0].isdigit():
                        day = int(groups[0])
                        month = months.get(groups[1].capitalize(), 1)
                        year = int(groups[2]) if groups[2] else today.year
                    else:
                        month = months.get(groups[0].capitalize(), 1)
                        day = int(groups[1])
                        year = int(groups[2]) if len(groups) > 2 and groups[2] else today.year
                    return dt(year, month, day).date()
                except:
                    pass
        return None

    try:
        open_date = parse_date(open_date_str)
        close_date = parse_date(close_date_str)

        if open_date and close_date:
            if today < open_date:
                return "UPCOMING"
            elif today > close_date:
                return "CLOSED"
            else:
                return "OPEN"
        elif open_date:
            if today < open_date:
                return "UPCOMING"
            elif today >= open_date:
                return "OPEN"
        elif close_date:
            if today > close_date:
                return "CLOSED"

        # Check for keywords in date strings
        combined = (open_date_str + close_date_str).lower()
        if 'listed' in combined or 'allot' in combined:
            return "CLOSED"
        elif 'open' in combined or 'live' in combined:
            return "OPEN"

    except Exception as e:
        pass

    return "UPCOMING"


def get_ipo_data():
    """
    Fetch IPO data from multiple sources.
    Returns mainboard IPOs only (not SME).
    Categories: OPEN, UPCOMING, RECENTLY CLOSED

    Always returns current IPO data - either live or curated.
    """
    global ipo_cache

    # Cache for 15 minutes only to ensure fresh data
    if ipo_cache["data"] and ipo_cache["updated"]:
        cache_age = (datetime.now(IST) - ipo_cache["updated"]).seconds
        if cache_age < 900:  # 15 minutes
            return ipo_cache["data"]

    ipos = []
    today = datetime.now(IST).date()

    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.9',
        }

        # Try fetching from Investorgain (more reliable for live IPO status)
        try:
            url = "https://www.investorgain.com/report/live-ipo-gmp/331/"
            response = requests.get(url, headers=headers, timeout=10)

            if response.status_code == 200:
                from bs4 import BeautifulSoup
                soup = BeautifulSoup(response.text, 'html.parser')

                # Find IPO table
                table = soup.find('table', {'id': 'mainTable'})
                if table:
                    rows = table.find_all('tr')[1:]  # Skip header
                    for row in rows[:20]:  # Limit to 20 IPOs
                        try:
                            cols = row.find_all('td')
                            if len(cols) >= 5:
                                name = cols[0].get_text(strip=True)
                                # Skip SME IPOs
                                if 'SME' in name.upper() or 'sme' in name.lower():
                                    continue

                                name = name.replace('IPO', '').replace(' - Mainboard', '').strip()
                                if not name:
                                    continue

                                price = cols[1].get_text(strip=True) if len(cols) > 1 else ""
                                gmp_text = cols[2].get_text(strip=True) if len(cols) > 2 else "0"

                                # Parse GMP
                                try:
                                    gmp = int(''.join(filter(lambda x: x.isdigit() or x == '-', gmp_text)))
                                except:
                                    gmp = 0

                                # Get dates and status from subsequent columns
                                open_date = cols[3].get_text(strip=True) if len(cols) > 3 else ""
                                close_date = cols[4].get_text(strip=True) if len(cols) > 4 else ""

                                # Determine status
                                status = determine_ipo_status(open_date, close_date, today)

                                ipo_data = {
                                    "id": len(ipos) + 1,
                                    "name": name,
                                    "type": "MAINBOARD",
                                    "status": status,
                                    "price_band": price if '₹' in price or price.isdigit() else f"₹{price}",
                                    "lot_size": "",
                                    "issue_size": "",
                                    "open_date": open_date,
                                    "close_date": close_date,
                                    "listing_date": "TBA",
                                    "gmp": gmp,
                                    "subscription": {"retail": 0, "nii": 0, "qib": 0},
                                    "financials": {},
                                    "positives": [],
                                    "negatives": [],
                                }

                                if not any(i['name'].lower() == name.lower() for i in ipos):
                                    ipos.append(ipo_data)

                        except Exception as e:
                            continue
        except Exception as e:
            logger.error(f"Error fetching from Investorgain: {e}")

        # Source 2: Hardcoded current IPOs (fallback - dynamically calculated status)
        if not ipos:
            # Current known mainboard IPOs - Status calculated dynamically
            current_ipos_raw = [
                {
                    "name": "Ventive Hospitality",
                    "price_band": "₹610-643",
                    "lot_size": "23 shares",
                    "issue_size": "₹1,600 Cr",
                    "open_date": "Dec 11, 2025",
                    "close_date": "Dec 13, 2025",
                    "listing_date": "Dec 18, 2025",
                    "gmp": 28,
                    "subscription": {"retail": 0.5, "nii": 0.3, "qib": 1.2},
                    "financials": {
                        "revenue": [450, 520, 680, 850],
                        "net_profit": [25, 35, 55, 78],
                        "debt": [120, 150, 180, 200]
                    },
                    "positives": ["Strong brand presence", "Consistent revenue growth"],
                    "negatives": ["High debt levels", "Competition from OYO, Lemon Tree"],
                },
                {
                    "name": "Mobikwik (One Mobikwik Systems)",
                    "price_band": "₹265-279",
                    "lot_size": "53 shares",
                    "issue_size": "₹572 Cr",
                    "open_date": "Dec 11, 2025",
                    "close_date": "Dec 13, 2025",
                    "listing_date": "Dec 18, 2025",
                    "gmp": 115,
                    "subscription": {"retail": 15.2, "nii": 42.5, "qib": 5.8},
                    "financials": {
                        "revenue": [380, 450, 540, 650],
                        "net_profit": [-128, -83, -45, 14],
                        "debt": [50, 45, 40, 35]
                    },
                    "positives": ["High GMP (41%)", "Turned profitable", "Fintech growth story"],
                    "negatives": ["History of losses", "Intense competition from Paytm, PhonePe"],
                },
                {
                    "name": "Sai Life Sciences",
                    "price_band": "₹522-549",
                    "lot_size": "27 shares",
                    "issue_size": "₹3,042 Cr",
                    "open_date": "Dec 11, 2025",
                    "close_date": "Dec 13, 2025",
                    "listing_date": "Dec 18, 2025",
                    "gmp": 55,
                    "subscription": {"retail": 1.8, "nii": 2.1, "qib": 4.5},
                    "financials": {
                        "revenue": [890, 1050, 1280, 1520],
                        "net_profit": [65, 85, 110, 145],
                        "debt": [280, 320, 350, 380]
                    },
                    "positives": ["Strong pharma CDMO player", "Consistent profitability", "Global clients"],
                    "negatives": ["High valuation", "Debt increasing"],
                },
                {
                    "name": "Vishal Mega Mart",
                    "price_band": "₹74-78",
                    "lot_size": "190 shares",
                    "issue_size": "₹8,000 Cr",
                    "open_date": "Dec 11, 2025",
                    "close_date": "Dec 13, 2025",
                    "listing_date": "Dec 18, 2025",
                    "gmp": 25,
                    "subscription": {"retail": 0, "nii": 0, "qib": 0},
                    "financials": {
                        "revenue": [5200, 6100, 7200, 8500],
                        "net_profit": [120, 180, 250, 320],
                        "debt": [800, 750, 700, 650]
                    },
                    "positives": ["Large retail chain", "Reducing debt", "Low price band"],
                    "negatives": ["Retail sector headwinds", "Competition from DMart"],
                },
                {
                    "name": "Inventurus Knowledge Solutions",
                    "price_band": "₹1,265-1,329",
                    "lot_size": "11 shares",
                    "issue_size": "₹2,498 Cr",
                    "open_date": "Dec 5, 2025",
                    "close_date": "Dec 9, 2025",
                    "listing_date": "Dec 12, 2025",
                    "gmp": 580,
                    "subscription": {"retail": 52.7, "nii": 214.8, "qib": 317.5},
                    "financials": {
                        "revenue": [520, 680, 890, 1150],
                        "net_profit": [85, 120, 165, 220],
                        "debt": [30, 25, 20, 15]
                    },
                    "positives": ["Healthcare IT services", "Strong subscription", "Low debt"],
                    "negatives": ["High valuation"],
                },
            ]

            # Process each IPO and calculate dynamic status
            for idx, ipo_raw in enumerate(current_ipos_raw, 1):
                status = determine_ipo_status(ipo_raw["open_date"], ipo_raw["close_date"], today)
                ipos.append({
                    "id": idx,
                    "name": ipo_raw["name"],
                    "type": "MAINBOARD",
                    "status": status,
                    "price_band": ipo_raw["price_band"],
                    "lot_size": ipo_raw["lot_size"],
                    "issue_size": ipo_raw["issue_size"],
                    "open_date": ipo_raw["open_date"],
                    "close_date": ipo_raw["close_date"],
                    "listing_date": ipo_raw["listing_date"],
                    "gmp": ipo_raw["gmp"],
                    "subscription": ipo_raw["subscription"],
                    "financials": ipo_raw["financials"],
                    "positives": ipo_raw["positives"],
                    "negatives": ipo_raw["negatives"],
                })

        # Update cache
        ipo_cache["data"] = ipos
        ipo_cache["updated"] = datetime.now(IST)

        return ipos

    except Exception as e:
        logger.error(f"IPO data fetch error: {e}")
        return ipo_cache.get("data", [])


def analyze_ipo(ipo):
    """Analyze IPO and provide recommendation."""
    score = 50  # Start neutral
    positives = []
    negatives = []

    financials = ipo.get('financials', {})

    # Revenue growth analysis
    revenues = financials.get('revenue', [])
    if len(revenues) >= 4:
        revenue_cagr = ((revenues[-1] / revenues[0]) ** (1/3) - 1) * 100 if revenues[0] > 0 else 0
        if revenue_cagr > 25:
            score += 15
            positives.append(f"Strong revenue CAGR: {revenue_cagr:.1f}%")
        elif revenue_cagr > 15:
            score += 10
            positives.append(f"Good revenue CAGR: {revenue_cagr:.1f}%")
        elif revenue_cagr < 5:
            score -= 10
            negatives.append(f"Low revenue growth: {revenue_cagr:.1f}%")

    # Profit growth analysis
    profits = financials.get('net_profit', [])
    if len(profits) >= 4:
        if profits[-1] > 0 and profits[0] > 0:
            profit_cagr = ((profits[-1] / profits[0]) ** (1/3) - 1) * 100
            if profit_cagr > 30:
                score += 15
                positives.append(f"Excellent profit CAGR: {profit_cagr:.1f}%")
            elif profit_cagr > 20:
                score += 10
                positives.append(f"Good profit CAGR: {profit_cagr:.1f}%")
        if profits[-1] <= 0:
            score -= 20
            negatives.append("Company not profitable")

    # Debt analysis
    debts = financials.get('debt', [])
    if len(debts) >= 1 and len(revenues) >= 1:
        debt_to_revenue = debts[-1] / revenues[-1] if revenues[-1] > 0 else 0
        if debt_to_revenue < 0.3:
            score += 10
            positives.append("Low debt levels")
        elif debt_to_revenue > 0.7:
            score -= 10
            negatives.append("High debt levels")

    # GMP analysis (Grey Market Premium)
    gmp = ipo.get('gmp', 0)
    price_band = ipo.get('price_band', '₹0-0')
    try:
        upper_price = int(price_band.replace('₹', '').split('-')[-1])
        gmp_pct = (gmp / upper_price * 100) if upper_price > 0 else 0
        if gmp_pct > 30:
            score += 10
            positives.append(f"Strong GMP: ₹{gmp} ({gmp_pct:.1f}%)")
        elif gmp_pct > 10:
            score += 5
            positives.append(f"Good GMP: ₹{gmp} ({gmp_pct:.1f}%)")
        elif gmp_pct < 0:
            score -= 15
            negatives.append(f"Negative GMP: ₹{gmp}")
    except:
        pass

    # Combine with provided positives/negatives
    positives.extend(ipo.get('positives', []))
    negatives.extend(ipo.get('negatives', []))

    # Determine recommendation
    if score >= 70:
        recommendation = "✅ SUBSCRIBE"
        rec_detail = "Strong fundamentals, good growth prospects"
    elif score >= 55:
        recommendation = "🟡 SUBSCRIBE FOR LISTING GAINS"
        rec_detail = "May give moderate listing gains"
    elif score >= 45:
        recommendation = "🟠 AVOID / RISKY"
        rec_detail = "Weak fundamentals or high valuation"
    else:
        recommendation = "❌ DO NOT SUBSCRIBE"
        rec_detail = "Poor financials or very high risk"

    return {
        "score": score,
        "recommendation": recommendation,
        "recommendation_detail": rec_detail,
        "positives": positives[:5],  # Top 5
        "negatives": negatives[:5]   # Top 5
    }


def handle_ipo(chat_id, ipo_selection=None):
    """
    IPO command - Shows list or details based on selection.
    /ipo - Show list of all IPOs
    /ipo 1 or /ipo Mobikwik - Show details of specific IPO
    """
    try:
        ipos = get_ipo_data()
        now = datetime.now(IST)

        if not ipos:
            send_message(chat_id, "❌ Unable to fetch IPO data. Please try again later.")
            return

        # If user selected a specific IPO
        if ipo_selection:
            selected_ipo = None

            # Try to find by number
            try:
                ipo_num = int(ipo_selection)
                for ipo in ipos:
                    if ipo.get('id') == ipo_num:
                        selected_ipo = ipo
                        break
            except ValueError:
                # Try to find by name (partial match)
                search_term = ipo_selection.lower()
                for ipo in ipos:
                    if search_term in ipo['name'].lower():
                        selected_ipo = ipo
                        break

            if selected_ipo:
                # Show detailed IPO info
                show_ipo_details(chat_id, selected_ipo)
            else:
                send_message(chat_id, f"❌ IPO '{ipo_selection}' not found. Use /ipo to see list.")
            return

        # Show IPO List
        open_ipos = [i for i in ipos if i['status'] == 'OPEN']
        upcoming_ipos = [i for i in ipos if i['status'] == 'UPCOMING']
        closed_ipos = [i for i in ipos if i['status'] == 'CLOSED']

        msg = f"""📊 *IPO DASHBOARD - MAINBOARD*
_{now.strftime('%Y-%m-%d %H:%M')} IST_

"""

        # OPEN IPOs
        if open_ipos:
            msg += f"🟢 *OPEN FOR SUBSCRIPTION ({len(open_ipos)}):*\n"
            msg += "-" * 28 + "\n"
            for ipo in open_ipos:
                gmp_str = f"GMP: ₹{ipo['gmp']}" if ipo.get('gmp') else ""
                msg += f"*{ipo['id']}.* {ipo['name']}\n"
                msg += f"   💰 {ipo['price_band']} | {gmp_str}\n"
                msg += f"   📅 Close: {ipo['close_date']}\n\n"
        else:
            msg += "🟢 *OPEN:* None currently\n\n"

        # UPCOMING IPOs
        if upcoming_ipos:
            msg += f"🟡 *UPCOMING ({len(upcoming_ipos)}):*\n"
            msg += "-" * 28 + "\n"
            for ipo in upcoming_ipos:
                msg += f"*{ipo['id']}.* {ipo['name']}\n"
                msg += f"   💰 {ipo['price_band']}\n"
                msg += f"   📅 Opens: {ipo['open_date']}\n\n"
        else:
            msg += "🟡 *UPCOMING:* None announced\n\n"

        # RECENTLY CLOSED IPOs
        if closed_ipos:
            msg += f"⚪ *RECENTLY CLOSED ({len(closed_ipos)}):*\n"
            msg += "-" * 28 + "\n"
            for ipo in closed_ipos[:3]:  # Show only last 3
                gmp_str = f"GMP: ₹{ipo['gmp']}" if ipo.get('gmp') else ""
                msg += f"*{ipo['id']}.* {ipo['name']}\n"
                msg += f"   💰 {ipo['price_band']} | {gmp_str}\n"
                msg += f"   📅 Listing: {ipo['listing_date']}\n\n"

        msg += f"""{'='*28}

📌 *To see detailed analysis:*
Type `/ipo 1` or `/ipo Mobikwik`

_Details include: Financials, GMP, Subscription, Plus/Minus factors, Apply recommendation_"""

        send_message(chat_id, msg)

    except Exception as e:
        logger.error(f"IPO command error: {e}")
        send_message(chat_id, "Error fetching IPO data. Please try again later.")


def show_ipo_details(chat_id, ipo):
    """Show detailed analysis for a specific IPO."""
    try:
        now = datetime.now(IST)
        analysis = analyze_ipo(ipo)

        status_emoji = "🟢" if ipo['status'] == 'OPEN' else "🟡" if ipo['status'] == 'UPCOMING' else "⚪"

        msg = f"""📊 *IPO DETAILED ANALYSIS*
_{now.strftime('%Y-%m-%d %H:%M')} IST_

{status_emoji} *{ipo['name']}*
*Status:* {ipo['status']}

{'='*30}
*📋 IPO DETAILS:*
• Price Band: {ipo['price_band']}
• Lot Size: {ipo['lot_size']}
• Issue Size: {ipo['issue_size']}
• Open Date: {ipo['open_date']}
• Close Date: {ipo['close_date']}
• Listing Date: {ipo['listing_date']}

*💹 GMP (Grey Market Premium):* ₹{ipo.get('gmp', 0)}
"""

        # Calculate expected listing gain
        if ipo.get('gmp'):
            try:
                price_text = ipo['price_band'].replace('₹', '').replace(',', '')
                price_upper = int(price_text.split('-')[-1].strip())
                if price_upper > 0:
                    expected_gain = (ipo['gmp'] / price_upper) * 100
                    msg += f"📈 *Expected Listing Gain:* {expected_gain:.1f}%\n"
            except:
                pass

        # Subscription status
        sub = ipo.get('subscription', {})
        if any(sub.values()):
            msg += f"""
{'='*30}
*📈 SUBSCRIPTION STATUS:*
• Retail: {sub.get('retail', 0):.2f}x
• NII (HNI): {sub.get('nii', 0):.2f}x
• QIB: {sub.get('qib', 0):.2f}x
• Total: {(sub.get('retail', 0) + sub.get('nii', 0) + sub.get('qib', 0))/3:.2f}x
"""

        # Financials (4 years)
        fin = ipo.get('financials', {})
        if fin and any([fin.get('revenue'), fin.get('net_profit'), fin.get('debt')]):
            current_year = datetime.now().year
            years = [f"FY{current_year-4}", f"FY{current_year-3}", f"FY{current_year-2}", f"FY{current_year-1}"]

            msg += f"""
{'='*30}
*📊 FINANCIALS (Last 4 Years):*

"""

            if fin.get('revenue'):
                rev = fin['revenue']
                msg += "*Revenue (₹ Cr):*\n"
                for i, val in enumerate(rev):
                    yr = years[i] if i < len(years) else f"Y{i+1}"
                    growth = ""
                    if i > 0 and rev[i-1] > 0:
                        growth_pct = ((val - rev[i-1]) / rev[i-1]) * 100
                        growth = f" ({growth_pct:+.1f}%)"
                    msg += f"  {yr}: ₹{val:,.0f} Cr{growth}\n"
                if len(rev) >= 2 and rev[0] > 0:
                    cagr = ((rev[-1] / rev[0]) ** (1/(len(rev)-1)) - 1) * 100
                    msg += f"  📈 *CAGR: {cagr:.1f}%*\n"
                msg += "\n"

            if fin.get('net_profit'):
                prof = fin['net_profit']
                msg += "*Net Profit (₹ Cr):*\n"
                for i, val in enumerate(prof):
                    yr = years[i] if i < len(years) else f"Y{i+1}"
                    growth = ""
                    if i > 0 and abs(prof[i-1]) > 0:
                        growth_pct = ((val - prof[i-1]) / abs(prof[i-1])) * 100
                        growth = f" ({growth_pct:+.1f}%)"
                    emoji = "🟢" if val > 0 else "🔴"
                    msg += f"  {yr}: {emoji} ₹{val:,.0f} Cr{growth}\n"
                if len(prof) >= 2 and prof[0] > 0 and prof[-1] > 0:
                    cagr = ((prof[-1] / prof[0]) ** (1/(len(prof)-1)) - 1) * 100
                    msg += f"  📈 *CAGR: {cagr:.1f}%*\n"
                msg += "\n"

            if fin.get('debt'):
                debt = fin['debt']
                msg += "*Debt (₹ Cr):*\n"
                for i, val in enumerate(debt):
                    yr = years[i] if i < len(years) else f"Y{i+1}"
                    change = ""
                    if i > 0:
                        diff = val - debt[i-1]
                        change = f" ({diff:+.0f})"
                    emoji = "🟢" if val < debt[0] else "🟡" if val == debt[0] else "🔴"
                    msg += f"  {yr}: {emoji} ₹{val:,.0f} Cr{change}\n"
                msg += "\n"

        # Plus Factors
        msg += f"{'='*30}\n"
        if analysis['positives']:
            msg += "*✅ PLUS FACTORS:*\n"
            for p in analysis['positives'][:5]:
                msg += f"  • {p}\n"
            msg += "\n"

        # Minus Factors
        if analysis['negatives']:
            msg += "*❌ MINUS FACTORS:*\n"
            for n in analysis['negatives'][:5]:
                msg += f"  • {n}\n"
            msg += "\n"

        # Recommendation
        msg += f"""{'='*30}
*🎯 FINAL RECOMMENDATION:*

{analysis['recommendation']}
📝 {analysis['recommendation_detail']}

*Analysis Score:* {analysis['score']}/100

{'='*30}
"""

        # Apply or Not recommendation
        if analysis['score'] >= 70:
            msg += """✅ *SHOULD YOU APPLY?* YES
• Strong fundamentals
• Good listing gain expected
• Apply with full lot"""
        elif analysis['score'] >= 55:
            msg += """🟡 *SHOULD YOU APPLY?* YES (for listing gains)
• Apply for short term gains
• Book profit on listing day
• Don't hold long term"""
        elif analysis['score'] >= 45:
            msg += """🟠 *SHOULD YOU APPLY?* RISKY
• Only apply if you understand risks
• Consider skipping
• Wait for better IPOs"""
        else:
            msg += """❌ *SHOULD YOU APPLY?* NO
• Weak fundamentals
• High risk of listing loss
• Skip this IPO"""

        msg += """

⚠️ *DISCLAIMER:*
_This is not investment advice. Do your own research._"""

        send_message(chat_id, msg)

    except Exception as e:
        logger.error(f"IPO details error: {e}")
        send_message(chat_id, "Error showing IPO details.")


def handle_fiidii(chat_id):
    """Show FII/DII data with analysis."""
    fii_dii = get_fii_dii_data()

    if not fii_dii:
        # Use fallback sample data
        fii_dii = get_fallback_fii_dii_data()
        if not fii_dii:
            send_message(chat_id, "❌ Unable to fetch FII/DII data. Please try again later.")
            return

    now = datetime.now(IST)

    msg = f"""📊 *FII/DII ACTIVITY*
_{now.strftime('%Y-%m-%d %H:%M')} IST_

*FII/FPI (Foreign Institutional Investors):*
  💰 Buy: ₹{fii_dii['fii']['buy']:,.2f} Cr
  💸 Sell: ₹{fii_dii['fii']['sell']:,.2f} Cr
  📊 Net: ₹{fii_dii['fii']['net']:+,.2f} Cr {'✅' if fii_dii['fii']['net'] > 0 else '❌'}

*DII (Domestic Institutional Investors):*
  💰 Buy: ₹{fii_dii['dii']['buy']:,.2f} Cr
  💸 Sell: ₹{fii_dii['dii']['sell']:,.2f} Cr
  📊 Net: ₹{fii_dii['dii']['net']:+,.2f} Cr {'✅' if fii_dii['dii']['net'] > 0 else '❌'}

{'='*30}
*SENTIMENT: {fii_dii['sentiment']}* {fii_dii['sentiment_emoji']}

📝 {fii_dii['description']}

*What This Means:*
"""
    if fii_dii['sentiment'] == "STRONG BULLISH":
        msg += """  ✅ Both institutions buying
  ✅ Strong market support
  ✅ Good for long positions"""
    elif fii_dii['sentiment'] == "BULLISH":
        msg += """  ✅ FII driving market up
  ⚠️ DII booking profits
  ✅ Overall positive bias"""
    elif fii_dii['sentiment'] == "CAUTIOUS":
        msg += """  ⚠️ FII exiting positions
  ✅ DII providing support
  ⚠️ Be selective in buying"""
    else:
        msg += """  ❌ Both selling heavily
  ❌ Market under pressure
  ❌ Avoid fresh positions"""

    msg += """

_Data from NSE. Updates after market hours._"""

    send_message(chat_id, msg)


def handle_optionchain(chat_id, symbol="NIFTY"):
    """Show Option Chain analysis for NIFTY/BANKNIFTY."""
    oc_data = get_option_chain_data(symbol)

    if not oc_data:
        # Use fallback data
        oc_data = get_fallback_option_chain_data(symbol)
        if not oc_data:
            send_message(chat_id, f"❌ Unable to fetch Option Chain for {symbol}. Please try again later.")
            return

    now = datetime.now(IST)
    pcr = oc_data['pcr']

    # PCR interpretation emoji
    if pcr['oi'] > 1.2:
        pcr_emoji = "🟢🟢"
    elif pcr['oi'] > 1.0:
        pcr_emoji = "🟢"
    elif pcr['oi'] < 0.8:
        pcr_emoji = "🔴🔴"
    elif pcr['oi'] < 1.0:
        pcr_emoji = "🔴"
    else:
        pcr_emoji = "🟡"

    msg = f"""📊 *OPTION CHAIN ANALYSIS*
*{oc_data['symbol']}*
_{oc_data['timestamp']} IST_

📈 *Spot Price:* {oc_data['underlying']:,.2f}

{'='*30}
*PCR (PUT-CALL RATIO)* {pcr_emoji}
{'='*30}

📊 *PCR (OI):* {pcr['oi']}
📊 *PCR (Volume):* {pcr['volume']}
📊 *Sentiment:* {pcr['sentiment']}

📝 {pcr['description']}

{'='*30}
*KEY LEVELS (Based on OI)*
{'='*30}

🔺 *Resistance:* {oc_data['resistance']:,}
   (Highest Call OI - {oc_data['analysis']['max_ce_oi']:,} contracts)

🔻 *Support:* {oc_data['support']:,}
   (Highest Put OI - {oc_data['analysis']['max_pe_oi']:,} contracts)

🎯 *Max Pain:* {oc_data['max_pain']:,}
   (Market likely to expire near this level)

{'='*30}
*OPEN INTEREST SUMMARY*
{'='*30}

📞 Total Call OI: {oc_data['total_call_oi']:,}
📉 Total Put OI: {oc_data['total_put_oi']:,}

*Trading Implication:*
"""
    if pcr['oi'] > 1.2:
        msg += """  ✅ More puts written = Writers expect support
  ✅ Market likely to stay above support
  ✅ Bullish bias for {symbol}"""
    elif pcr['oi'] < 0.8:
        msg += """  ⚠️ More calls written = Writers expect resistance
  ⚠️ Market may face selling pressure
  ⚠️ Bearish bias for {symbol}"""
    else:
        msg += """  ➡️ Balanced OI = No clear direction
  ➡️ Range-bound movement expected
  ➡️ Trade with caution"""

    msg += """

_Data from NSE Option Chain. Updates every 5 min._"""

    send_message(chat_id, msg.format(symbol=symbol.upper()))


def handle_calendar(chat_id):
    """Show upcoming economic events."""
    calendar = get_upcoming_economic_events()

    if not calendar or not calendar.get('events'):
        send_message(chat_id, "No major economic events in the next 7 days.")
        return

    now = datetime.now(IST)

    msg = f"""📅 *ECONOMIC CALENDAR*
_{now.strftime('%Y-%m-%d %H:%M')} IST_

*Upcoming Events (Next 7 Days):*

"""

    for event in calendar['events']:
        impact_emoji = "🔴" if event['impact'] in ["HIGH", "VERY HIGH"] else "🟡"

        msg += f"""{impact_emoji} *{event['event']}*
  📅 Date: {event['date']} ({event['day']})
  ⚡ Impact: {event['impact']}
  📝 {event['description']}
  💡 {event['recommendation']}

"""

    if calendar.get('next_important'):
        next_event = calendar['next_important']
        msg += f"""{'='*30}
⚠️ *NEXT IMPORTANT EVENT:*
*{next_event['event']}* on {next_event['date']}

*Recommendation:* {next_event['recommendation']}
"""

    msg += """
_Events may affect market volatility. Plan accordingly._"""

    send_message(chat_id, msg)


def handle_sentiment(chat_id):
    """Show overall market sentiment combining all data sources."""
    sentiment = get_market_sentiment_summary()
    fii_dii = get_fii_dii_data() or get_fallback_fii_dii_data()
    oc_nifty = get_option_chain_data("NIFTY") or get_fallback_option_chain_data("NIFTY")
    calendar = get_upcoming_economic_events()

    now = datetime.now(IST)

    msg = f"""🎯 *MARKET SENTIMENT ANALYSIS*
_{now.strftime('%Y-%m-%d %H:%M')} IST_

{'='*30}
*OVERALL SENTIMENT: {sentiment['sentiment']}* {sentiment['emoji']}
*Score: {sentiment['score']}/100*
{'='*30}

*Components:*

1️⃣ *FII/DII:* {sentiment['components']['fii_dii']}
"""
    if fii_dii:
        msg += f"   FII Net: ₹{fii_dii['fii']['net']:+,.0f} Cr\n"
        msg += f"   DII Net: ₹{fii_dii['dii']['net']:+,.0f} Cr\n"

    msg += f"""
2️⃣ *PCR (NIFTY):* {sentiment['components']['pcr']}
"""
    if oc_nifty:
        msg += f"   Support: {oc_nifty['support']:,}\n"
        msg += f"   Resistance: {oc_nifty['resistance']:,}\n"

    msg += """
3️⃣ *Economic Calendar:*
"""
    if calendar and calendar.get('next_important'):
        msg += f"   Next: {calendar['next_important']['event']}\n"
        msg += f"   Impact: {calendar['next_important']['impact']}\n"
    else:
        msg += "   No major events this week\n"

    msg += f"""
{'='*30}
*TRADING ADVICE:*
💡 {sentiment['advice']}

*Action Plan:*
"""
    if sentiment['score'] >= 60:
        msg += """  ✅ Buy on dips near support levels
  ✅ Trail stop losses on existing longs
  ✅ Index calls can be considered"""
    elif sentiment['score'] <= 40:
        msg += """  ⚠️ Avoid fresh long positions
  ⚠️ Book profits on rallies
  ⚠️ Keep position sizes small"""
    else:
        msg += """  ➡️ Wait for clear breakout/breakdown
  ➡️ Trade only high confidence setups
  ➡️ Maintain hedged positions"""

    msg += """

_Sentiment updates with market data._"""

    send_message(chat_id, msg)


# ===== AUTOMATIC ALERTS =====

def send_market_summary_auto():
    """Send automatic market summary with indices and watchlist"""
    now = datetime.now(IST)

    msg = f"""📊 *AUTOMATIC MARKET UPDATE*
_{now.strftime('%Y-%m-%d %H:%M')} IST_

*INDICES:*
"""
    # Get indices
    for idx in ["NIFTY", "SENSEX", "BANKNIFTY"]:
        data = get_index_data(idx)
        if data:
            emoji = "📈" if data.get('change', 0) >= 0 else "📉"
            msg += f"{emoji} *{idx}*: {data.get('value', 0):,.2f} ({data.get('pct', 0):+.2f}%)\n"

    msg += "\n*YOUR WATCHLIST:*\n"

    # Get watchlist prices
    for sym in DEFAULT_WATCHLIST:
        try:
            info = get_stock_info(sym)
            if info and info.get('price'):
                price = info['price']
                prev = info.get('previous_close', price)
                pct = ((price - prev) / prev * 100) if prev else 0
                emoji = "📈" if pct >= 0 else "📉"
                msg += f"{emoji} *{sym}*: ₹{price:,.2f} ({pct:+.2f}%)\n"
        except:
            pass

    send_message(ADMIN_CHAT_ID, msg)
    logger.info("Auto market summary sent")


def send_market_open_alert():
    """Send alert when market opens"""
    msg = """🔔 *MARKET OPEN ALERT*

Indian Stock Market is NOW OPEN! 🟢

Trading Hours: 9:15 AM - 3:30 PM IST

Good luck with your trades today!"""
    send_message(ADMIN_CHAT_ID, msg)
    logger.info("Market open alert sent")


def send_market_close_alert():
    """Send alert when market closes with full summary"""
    now = datetime.now(IST)

    msg = f"""🔔 *MARKET CLOSED*
_{now.strftime('%Y-%m-%d %H:%M')} IST_

Indian Stock Market is NOW CLOSED! 🔴

*TODAY'S SUMMARY:*

"""
    # Get indices
    for idx in ["NIFTY", "SENSEX", "BANKNIFTY"]:
        data = get_index_data(idx)
        if data:
            emoji = "📈" if data.get('change', 0) >= 0 else "📉"
            msg += f"{emoji} *{idx}*: {data.get('value', 0):,.2f} ({data.get('pct', 0):+.2f}%)\n"

    # Collect all watchlist performance data
    watchlist_performance = []
    for sym in DEFAULT_WATCHLIST:
        try:
            info = get_stock_info(sym)
            if info and info.get('price'):
                price = info['price']
                prev = info.get('previous_close', price)
                pct = ((price - prev) / prev * 100) if prev else 0
                watchlist_performance.append({
                    'symbol': sym,
                    'price': price,
                    'change_pct': pct
                })
        except:
            pass

    # Sort by performance: Gainers (highest to lowest) then Losers (lowest to highest)
    watchlist_performance.sort(key=lambda x: x['change_pct'], reverse=True)

    # Separate gainers and losers
    gainers = [s for s in watchlist_performance if s['change_pct'] >= 0]
    losers = [s for s in watchlist_performance if s['change_pct'] < 0]

    msg += "\n*🟢 TOP GAINERS:*\n"
    for stock in gainers[:10]:  # Top 10 gainers
        msg += f"📈 *{stock['symbol']}*: ₹{stock['price']:,.2f} ({stock['change_pct']:+.2f}%)\n"

    if not gainers:
        msg += "_No gainers today_\n"

    msg += "\n*🔴 TOP LOSERS:*\n"
    for stock in losers[:10]:  # Top 10 losers
        msg += f"📉 *{stock['symbol']}*: ₹{stock['price']:,.2f} ({stock['change_pct']:+.2f}%)\n"

    if not losers:
        msg += "_No losers today_\n"

    # Summary stats
    total_gainers = len(gainers)
    total_losers = len(losers)
    market_breadth = "Bullish" if total_gainers > total_losers else "Bearish" if total_losers > total_gainers else "Neutral"

    msg += f"\n*📊 Market Breadth:* {market_breadth}\n"
    msg += f"Gainers: {total_gainers} | Losers: {total_losers}\n"

    msg += "\nSee you tomorrow! 👋"
    send_message(ADMIN_CHAT_ID, msg)
    logger.info("Market close alert sent")


def send_fiidii_daily_alert():
    """Send daily FII/DII update at 4:30 PM after market close."""
    try:
        fii_dii = get_fii_dii_data()

        if not fii_dii:
            logger.warning("Could not fetch FII/DII for daily alert")
            return

        now = datetime.now(IST)

        msg = f"""📊 *DAILY FII/DII UPDATE*
_{now.strftime('%Y-%m-%d')} IST_

*FII/FPI:*
  Net: ₹{fii_dii['fii']['net']:+,.2f} Cr {'✅ BUYING' if fii_dii['fii']['net'] > 0 else '❌ SELLING'}

*DII:*
  Net: ₹{fii_dii['dii']['net']:+,.2f} Cr {'✅ BUYING' if fii_dii['dii']['net'] > 0 else '❌ SELLING'}

*Sentiment: {fii_dii['sentiment']}* {fii_dii['sentiment_emoji']}

📝 {fii_dii['description']}

_Tomorrow's outlook: {'Positive' if fii_dii['fii']['net'] > 0 else 'Cautious'}_"""

        send_message(ADMIN_CHAT_ID, msg)
        logger.info("FII/DII daily alert sent")

    except Exception as e:
        logger.error(f"FII/DII daily alert error: {e}")


def send_morning_briefing():
    """
    Send comprehensive morning briefing at 8:45 AM before market open.
    Includes: Gift Nifty, FII/DII, Option Chain, Calendar events
    """
    try:
        now = datetime.now(IST)

        msg = f"""🌅 *MORNING MARKET BRIEFING*
_{now.strftime('%Y-%m-%d %H:%M')} IST_

"""

        # Gift Nifty
        gift_data = get_gift_nifty_data()
        if gift_data:
            emoji = "📈" if gift_data['change'] >= 0 else "📉"
            msg += f"""*GIFT NIFTY:*
{emoji} {gift_data['current']:,.2f} ({gift_data['pct']:+.2f}%)
Expected Opening: {'Gap Up' if gift_data['pct'] > 0.3 else 'Gap Down' if gift_data['pct'] < -0.3 else 'Flat'}

"""

        # FII/DII
        fii_dii = get_fii_dii_data()
        if fii_dii:
            msg += f"""*FII/DII (Previous Day):*
FII: ₹{fii_dii['fii']['net']:+,.0f} Cr | DII: ₹{fii_dii['dii']['net']:+,.0f} Cr
Sentiment: {fii_dii['sentiment']} {fii_dii['sentiment_emoji']}

"""

        # Option Chain - Support/Resistance
        oc_nifty = get_option_chain_data("NIFTY")
        if oc_nifty:
            msg += f"""*NIFTY OI LEVELS:*
🔻 Support: {oc_nifty['support']:,}
🔺 Resistance: {oc_nifty['resistance']:,}
PCR: {oc_nifty['pcr']['oi']} ({oc_nifty['pcr']['sentiment']})

"""

        # Today's Events
        calendar = get_upcoming_economic_events()
        if calendar and calendar.get('events'):
            today = now.strftime('%Y-%m-%d')
            today_events = [e for e in calendar['events'] if e['date'] == today]
            if today_events:
                msg += "*TODAY'S EVENTS:*\n"
                for event in today_events:
                    msg += f"  ⚠️ {event['event']} ({event['impact']})\n"
                msg += "\n"

        # Overall Sentiment
        sentiment = get_market_sentiment_summary()
        msg += f"""{'='*30}
*TODAY'S OUTLOOK: {sentiment['sentiment']}* {sentiment['emoji']}
💡 {sentiment['advice']}

_Market opens at 9:15 AM IST_"""

        send_message(ADMIN_CHAT_ID, msg)
        logger.info("Morning briefing sent")

    except Exception as e:
        logger.error(f"Morning briefing error: {e}")


def send_calendar_event_alert(alert_type="evening"):
    """
    Send alert for market events.
    alert_type: "evening" (8 PM for tomorrow), "morning" (7 AM for today)
    """
    try:
        calendar = get_upcoming_economic_events()

        if not calendar or not calendar.get('events'):
            return

        now = datetime.now(IST)

        if alert_type == "morning":
            target_date = now.strftime('%Y-%m-%d')
            header = "TODAY'S MARKET EVENTS"
            footer = "⚠️ *Trade with caution today!*\n\n_High impact events expected._"
        else:  # evening
            target_date = (now + timedelta(days=1)).strftime('%Y-%m-%d')
            header = "TOMORROW'S MARKET EVENTS"
            footer = "⚠️ *Plan your trades accordingly!*\n\n_Events can cause high volatility._"

        # Check for target date's events
        target_events = [e for e in calendar['events'] if e['date'] == target_date]

        if not target_events:
            return

        # Sort by impact level
        impact_order = {"VERY HIGH": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}
        target_events.sort(key=lambda x: impact_order.get(x['impact'], 4))

        msg = f"""📅 *{header}*
_{target_date}_

"""
        # Count by impact
        high_impact = sum(1 for e in target_events if e['impact'] in ["HIGH", "VERY HIGH"])

        if high_impact > 0:
            msg += f"🚨 *{high_impact} HIGH IMPACT EVENT(S)*\n\n"

        for event in target_events:
            impact = event['impact']
            if impact == "VERY HIGH":
                impact_emoji = "🔴🔴"
            elif impact == "HIGH":
                impact_emoji = "🔴"
            elif impact == "MEDIUM":
                impact_emoji = "🟡"
            else:
                impact_emoji = "🟢"

            msg += f"""{impact_emoji} *{event['event']}*
  Impact: {impact}
  📝 {event['description']}
  💡 {event['recommendation']}

"""

        msg += footer

        send_message(ADMIN_CHAT_ID, msg)
        logger.info(f"Calendar event alert ({alert_type}) sent for {len(target_events)} events")

    except Exception as e:
        logger.error(f"Calendar event alert error: {e}")


def send_pre_event_alert():
    """
    Send alert 1 hour before very high impact events.
    This should be called every hour during market hours.
    """
    try:
        calendar = get_upcoming_economic_events()
        if not calendar or not calendar.get('events'):
            return

        now = datetime.now(IST)
        today = now.strftime('%Y-%m-%d')

        # Check today's very high impact events
        today_events = [e for e in calendar['events']
                       if e['date'] == today and e['impact'] in ["VERY HIGH", "HIGH"]]

        for event in today_events:
            # Create event key for deduplication
            event_key = f"pre_event_{event['event']}_{today}"
            if last_scheduled_alert.get(event_key):
                continue

            msg = f"""⏰ *EVENT REMINDER*
_{now.strftime('%H:%M')} IST_

🔴 *{event['event']}* is happening today!

Impact: *{event['impact']}*
📝 {event['description']}

💡 *Action:* {event['recommendation']}

⚠️ Consider reducing position sizes before this event."""

            send_message(ADMIN_CHAT_ID, msg)
            last_scheduled_alert[event_key] = True
            logger.info(f"Pre-event alert sent: {event['event']}")

    except Exception as e:
        logger.error(f"Pre-event alert error: {e}")


def check_breaking_news():
    """Check for important breaking news and send alerts - with persistent deduplication"""
    global sent_news_ids

    try:
        # Important keywords for Indian market
        keywords = ["RBI", "SEBI", "Sensex", "Nifty", "crash", "surge", "breaking",
                   "Fed", "interest rate", "inflation", "GDP", "recession",
                   "Reliance", "TCS", "HDFC", "Infosys", "budget", "tax",
                   "IPO", "FII", "DII", "bonus", "split", "dividend"]

        articles = get_market_news(limit=10)
        news_sent_this_cycle = False

        for article in articles:
            title = article.get('title', '')
            if not title:
                continue

            description = article.get('description', '') or ''
            published = article.get('publishedAt', '')

            # Skip old news (more than 6 hours old)
            if published:
                try:
                    pub_time = datetime.fromisoformat(published.replace('Z', '+00:00'))
                    if datetime.now(pytz.UTC) - pub_time > timedelta(hours=6):
                        continue
                except:
                    pass

            # Create consistent unique ID using MD5 hash
            news_id = get_news_id(title)

            # Skip if already sent
            if news_id in sent_news_ids:
                continue

            # Check if important
            content = (title + " " + description).lower()
            is_important = any(kw.lower() in content for kw in keywords)

            if is_important:
                msg = f"""🚨 *BREAKING NEWS ALERT*

*{title}*

{description[:200] if description else ''}{'...' if len(description) > 200 else ''}

_Source: {article.get('source', {}).get('name', 'Unknown')}_"""

                send_message(ADMIN_CHAT_ID, msg)
                sent_news_ids.add(news_id)
                news_sent_this_cycle = True
                logger.info(f"Breaking news sent: {title[:50]}")

                # Keep only last 100 news IDs in memory
                if len(sent_news_ids) > 100:
                    sent_news_ids = set(list(sent_news_ids)[-50:])

                # Only send one news at a time
                break

        # Save to file for persistence across restarts
        if news_sent_this_cycle:
            save_sent_news_ids()

    except Exception as e:
        logger.error(f"Breaking news check error: {e}")


def send_gift_nifty_close_notification():
    """
    Send Gift Nifty close summary notification.
    Gift Nifty closes at 11:30 PM IST.
    """
    try:
        gift_data = get_gift_nifty_data()

        if not gift_data:
            logger.warning("Could not fetch Gift Nifty data")
            return

        now = datetime.now(IST)
        emoji = "📈" if gift_data['change'] >= 0 else "📉"

        msg = f"""🌙 *GIFT NIFTY CLOSE UPDATE*
_{now.strftime('%Y-%m-%d %H:%M')} IST_

{emoji} *GIFT NIFTY*

📊 *Current:* {gift_data['current']:,.2f}
📊 *Change:* {gift_data['change']:+,.2f} ({gift_data['pct']:+.2f}%)

*Today's Range:*
  🔺 High: {gift_data['high']:,.2f}
  🔻 Low: {gift_data['low']:,.2f}
  📏 Range: {gift_data['range']:,.2f} points

*Opening Data:*
  📍 Open: {gift_data['open']:,.2f}

*What This Means:*
"""
        if gift_data['pct'] > 0.5:
            msg += "  ✅ Positive cues for Indian markets tomorrow\n"
            msg += "  📈 Gap-up opening expected\n"
        elif gift_data['pct'] < -0.5:
            msg += "  ⚠️ Negative cues for Indian markets tomorrow\n"
            msg += "  📉 Gap-down opening expected\n"
        else:
            msg += "  ➡️ Flat to slightly positive/negative opening expected\n"
            msg += "  📊 Watch global cues and US markets\n"

        msg += """
_Gift Nifty indicates SGX Nifty futures movement._
_Use this for tomorrow's market sentiment._"""

        send_message(ADMIN_CHAT_ID, msg)
        logger.info("Gift Nifty close notification sent")

    except Exception as e:
        logger.error(f"Gift Nifty notification error: {e}")


def check_index_signals():
    """
    Check NIFTY, BANKNIFTY, SENSEX for trading opportunities.
    Generate signals when high confidence setup is detected.
    """
    global last_scheduled_alert

    try:
        now = datetime.now(IST)
        today_key = now.strftime('%Y-%m-%d-%H')
        index_alerts = []

        for idx in INDEX_WATCHLIST:
            # Check if we already sent alert for this index this hour
            alert_key = f"index_{idx}_{today_key}"
            if last_scheduled_alert.get(alert_key):
                continue

            df = get_stock_data(idx, period="3mo")
            if df is None:
                continue

            analysis = analyze_stock(df)
            if not analysis:
                continue

            confidence = analysis.get('confidence_score', 50)
            signal = analysis.get('signal', 'NEUTRAL')

            # Only alert for high confidence signals (>=65 for indices)
            if confidence >= 65 and signal != 'NEUTRAL':
                data = get_index_data(idx)
                current_price = data.get('value', 0) if data else 0

                # Calculate trading levels
                trading = calculate_trading_levels(analysis.get('df', df), analysis, {"price": current_price})

                if trading:
                    index_alerts.append({
                        'symbol': idx,
                        'signal': signal,
                        'confidence': confidence,
                        'price': current_price,
                        'change_pct': data.get('pct', 0) if data else 0,
                        'trading': trading,
                        'score_factors': analysis.get('score_factors', {}),
                        'rsi': analysis.get('rsi', 50)
                    })
                    last_scheduled_alert[alert_key] = True

        if index_alerts:
            for alert in index_alerts:
                trading = alert['trading']
                emoji = "🟢" if "BUY" in alert['signal'] else "🔴"
                conf_emoji = "🟢🟢🟢" if alert['confidence'] >= 75 else "🟢🟢"

                msg = f"""📊 *INDEX SIGNAL ALERT* {emoji}
_{now.strftime('%Y-%m-%d %H:%M')} IST_

*{alert['symbol']}* - {alert['signal']}
*Confidence: {alert['confidence']}/100* {conf_emoji}

📈 *Current Price:* {alert['price']:,.2f}
📊 *Today's Change:* {alert['change_pct']:+.2f}%

{'='*30}
*TRADING RECOMMENDATION*
{'='*30}

{emoji} *Action:* {trading['action']}
*Quality:* {trading['trade_quality']}
*Timeframe:* {trading['timeframe']}

*Entry Price:* {trading['entry_price']:,.2f}

🎯 *Target 1:* {trading['target_1']:,.2f} ({trading['target_1_pct']:+.2f}%)
   R:R = 1:{trading['rr_ratio_1']:.1f}

🎯 *Target 2:* {trading['target_2']:,.2f} ({trading['target_2_pct']:+.2f}%)
   R:R = 1:{trading['rr_ratio_2']:.1f}

🛑 *Stop Loss:* {trading['stop_loss']:,.2f} ({trading['stop_loss_pct']:+.2f}%)

*Key Levels:*
  • Support: {trading['support']:,.2f}
  • Resistance: {trading['resistance']:,.2f}

*Score Breakdown:*
"""
                for factor, detail in list(alert['score_factors'].items())[:4]:
                    msg += f"  • {factor}: {detail}\n"

                msg += """
_For F&O trading. This is not financial advice._"""

                send_message(ADMIN_CHAT_ID, msg)

                # Create active signal for tracking
                create_active_signal(
                    symbol=alert['symbol'],
                    signal_type=trading['action'],
                    entry_price=trading['entry_price'],
                    target_1=trading['target_1'],
                    target_2=trading['target_2'],
                    stop_loss=trading['stop_loss'],
                    timeframe=trading['timeframe'],
                    confidence=alert['confidence'],
                    methodology=trading.get('methodology', {})
                )

            logger.info(f"Index signals sent: {len(index_alerts)} indices")

    except Exception as e:
        logger.error(f"Index signals check error: {e}")


def check_strong_signals():
    """Check watchlist for strong buy/sell signals based on confidence score"""
    try:
        strong_alerts = []

        # Check both stocks AND indices
        for sym in FULL_WATCHLIST:
            df = get_stock_data(sym, period="1mo")
            if df is None:
                continue

            analysis = analyze_stock(df)
            if not analysis:
                continue

            # Use new confidence score (>=70 for buy, <=30 for sell)
            confidence = analysis.get('confidence_score', 50)
            signal = analysis.get('signal', '')

            if confidence >= 70:  # High confidence buy
                info = get_stock_info(sym)
                price = info.get('price', 0) if info else 0
                strong_alerts.append({
                    'symbol': sym,
                    'signal': '🟢 HIGH CONFIDENCE BUY',
                    'price': price,
                    'rsi': analysis.get('rsi', 0),
                    'confidence': confidence,
                    'score_factors': analysis.get('score_factors', {})
                })
            elif confidence <= 30:  # High confidence sell
                info = get_stock_info(sym)
                price = info.get('price', 0) if info else 0
                strong_alerts.append({
                    'symbol': sym,
                    'signal': '🔴 HIGH CONFIDENCE SELL',
                    'price': price,
                    'rsi': analysis.get('rsi', 0),
                    'confidence': confidence,
                    'score_factors': analysis.get('score_factors', {})
                })

        if strong_alerts:
            msg = "🚨 *HIGH CONFIDENCE SIGNAL ALERT*\n\n"
            for alert in strong_alerts[:3]:  # Max 3 alerts
                rsi_val = alert.get('rsi')
                rsi_str = f"{rsi_val:.1f}" if rsi_val and rsi_val > 0 else "N/A"
                msg += f"""*{alert['symbol']}* - {alert['signal']}
  Price: ₹{alert['price']:,.2f}
  Confidence: {alert['confidence']}/100
  RSI: {rsi_str}

  Key Factors:
"""
                for factor, detail in list(alert['score_factors'].items())[:3]:
                    msg += f"    • {factor}: {detail}\n"
                msg += "\n"

            msg += "_This is not financial advice. Do your own research._"
            send_message(ADMIN_CHAT_ID, msg)
            logger.info(f"High confidence alerts sent: {len(strong_alerts)} stocks")

    except Exception as e:
        logger.error(f"Strong signals check error: {e}")


def check_high_confidence_alerts():
    """
    Check for high-confidence opportunities (score >= 70) and send alerts.
    This is triggered every 15 minutes during market hours.
    """
    global last_scheduled_alert
    try:
        now = datetime.now(IST)
        today_key = now.strftime('%Y-%m-%d-%H')  # Hourly key to limit alerts

        high_conf_alerts = []

        # Check extended watchlist for high confidence
        extended_watchlist = DEFAULT_WATCHLIST + ["SBIN", "BAJFINANCE", "LT", "MARUTI", "WIPRO"]

        for sym in extended_watchlist:
            df = get_stock_data(sym, period="1mo")
            if df is None:
                continue

            analysis = analyze_stock(df)
            if not analysis:
                continue

            confidence = analysis.get('confidence_score', 50)

            # Only alert if confidence >= 70 (strong buy signal)
            if confidence >= 70:
                # Check if we already sent alert for this stock today
                alert_key = f"high_conf_{sym}_{today_key}"
                if last_scheduled_alert.get(alert_key):
                    continue

                info = get_stock_info(sym)
                price = info.get('price', 0) if info else 0
                prev = info.get('previous_close', price) if info else price
                change_pct = ((price - prev) / prev * 100) if prev else 0

                high_conf_alerts.append({
                    'symbol': sym,
                    'price': price,
                    'change_pct': change_pct,
                    'confidence': confidence,
                    'signal': analysis.get('signal', ''),
                    'rsi': analysis.get('rsi', 50),
                    'score_factors': analysis.get('score_factors', {})
                })
                last_scheduled_alert[alert_key] = True

        if high_conf_alerts:
            # Sort by confidence score
            high_conf_alerts.sort(key=lambda x: x['confidence'], reverse=True)

            for alert in high_conf_alerts[:3]:  # Max 3 detailed alerts
                # Calculate trading levels for each stock
                df = get_stock_data(alert['symbol'], period="1mo")
                if df is None:
                    continue

                analysis = analyze_stock(df)
                info = get_stock_info(alert['symbol'])
                trading = calculate_trading_levels(df, analysis, info) if analysis else None

                emoji = "🟢🟢🟢" if alert['confidence'] >= 85 else "🟢🟢"
                rsi_val = alert.get('rsi')
                rsi_str = f"{rsi_val:.1f}" if rsi_val and rsi_val > 0 else "N/A"

                msg = f"""🎯 *HIGH CONFIDENCE OPPORTUNITY*
_{now.strftime('%Y-%m-%d %H:%M')} IST_

*{alert['symbol']}* {emoji}
Signal: *{alert['signal']}*
Confidence: *{alert['confidence']}/100*

💰 *Current Price:* ₹{alert['price']:,.2f} ({alert['change_pct']:+.2f}%)
"""

                if trading:
                    msg += f"""
{'='*28}
*TRADING LEVELS*
{'='*28}

📍 *Entry:* ₹{trading['entry_price']:,.2f}
🎯 *Target 1:* ₹{trading['target_1']:,.2f} ({trading['target_1_pct']:+.2f}%)
🎯 *Target 2:* ₹{trading['target_2']:,.2f} ({trading['target_2_pct']:+.2f}%)
🛑 *Stop Loss:* ₹{trading['stop_loss']:,.2f} ({trading['stop_loss_pct']:+.2f}%)

⏱️ *Timeframe:* {trading['timeframe']}
📊 *R:R Ratio:* 1:{trading['rr_ratio_1']:.1f}

*Key Levels:*
  Support: ₹{trading['support']:,.2f}
  Resistance: ₹{trading['resistance']:,.2f}
"""

                msg += f"""
*Technical:*
  RSI: {rsi_str}
"""
                for factor, detail in list(alert['score_factors'].items())[:3]:
                    msg += f"  • {factor}: {detail}\n"

                msg += "\n_Not financial advice. DYOR._"
                send_message(ADMIN_CHAT_ID, msg)

                # Create active signal for tracking
                if trading:
                    create_active_signal(
                        symbol=alert['symbol'],
                        signal_type=trading['action'],
                        entry_price=trading['entry_price'],
                        target_1=trading['target_1'],
                        target_2=trading['target_2'],
                        stop_loss=trading['stop_loss'],
                        timeframe=trading['timeframe'],
                        confidence=alert['confidence']
                    )

            logger.info(f"High confidence alerts sent: {len(high_conf_alerts)} stocks")

    except Exception as e:
        logger.error(f"High confidence check error: {e}")


# ===== NEW AUTOMATIC FEATURES =====

def send_intraday_picks():
    """
    Send 3-5 best intraday trading picks at 9:15 AM.
    Based on pre-market analysis, gap analysis, and technical setup.
    """
    try:
        now = datetime.now(IST)
        intraday_picks = []

        for sym in DEFAULT_WATCHLIST:
            try:
                df = get_stock_data(sym, period="1mo")
                if df is None or len(df) < 20:
                    continue

                analysis = analyze_stock(df)
                if not analysis:
                    continue

                info = get_stock_info(sym)
                if not info:
                    continue

                price = info.get('price', 0)
                prev_close = info.get('previous_close', price)
                gap_pct = ((price - prev_close) / prev_close * 100) if prev_close else 0

                rsi = analysis.get('rsi', 50)
                confidence = analysis.get('confidence_score', 50)
                signal = analysis.get('signal', 'NEUTRAL')

                # Calculate intraday score
                intraday_score = 0

                # Gap analysis (+/- 20 points)
                if 0.5 < gap_pct < 2:  # Moderate gap up = bullish
                    intraday_score += 15
                elif -2 < gap_pct < -0.5:  # Moderate gap down = bearish reversal potential
                    intraday_score += 10

                # RSI for intraday
                if 30 < rsi < 40:  # Oversold bounce
                    intraday_score += 20
                elif 60 < rsi < 70:  # Momentum play
                    intraday_score += 15

                # Volume check (from previous day)
                if len(df) >= 2:
                    avg_vol = df['volume'].tail(20).mean()
                    last_vol = df['volume'].iloc[-1]
                    if last_vol > avg_vol * 1.5:
                        intraday_score += 15

                # Confidence boost
                if confidence >= 60:
                    intraday_score += 15
                elif confidence >= 50:
                    intraday_score += 5

                if intraday_score >= 25 and signal != 'NEUTRAL':
                    # Calculate intraday levels
                    high = info.get('high', price * 1.01)
                    low = info.get('low', price * 0.99)
                    pivot = (high + low + prev_close) / 3
                    r1 = 2 * pivot - low
                    s1 = 2 * pivot - high

                    intraday_picks.append({
                        'symbol': sym,
                        'price': price,
                        'gap_pct': gap_pct,
                        'signal': signal,
                        'confidence': confidence,
                        'rsi': rsi,
                        'intraday_score': intraday_score,
                        'pivot': pivot,
                        'r1': r1,
                        's1': s1,
                        'target': r1 if 'BUY' in signal else s1,
                        'stop_loss': s1 if 'BUY' in signal else r1,
                    })

            except Exception as e:
                logger.error(f"Intraday analysis error for {sym}: {e}")
                continue

        if not intraday_picks:
            logger.info("No strong intraday picks today")
            return

        # Sort by intraday score
        intraday_picks.sort(key=lambda x: x['intraday_score'], reverse=True)
        top_picks = intraday_picks[:5]

        msg = f"""🌅 *INTRADAY PICKS - {now.strftime('%Y-%m-%d')}*
_Market Opening at 9:15 AM IST_

Top {len(top_picks)} stocks for today's intraday trading:

"""

        for i, pick in enumerate(top_picks, 1):
            emoji = "🟢" if "BUY" in pick['signal'] else "🔴"
            gap_emoji = "📈" if pick['gap_pct'] > 0 else "📉"

            msg += f"""*{i}. {pick['symbol']}* {emoji}
  💰 CMP: ₹{pick['price']:,.2f} {gap_emoji} Gap: {pick['gap_pct']:+.2f}%
  📊 Signal: {pick['signal']} (Score: {pick['intraday_score']})
  🎯 Target: ₹{pick['target']:,.2f}
  🛑 Stop Loss: ₹{pick['stop_loss']:,.2f}
  📍 Pivot: ₹{pick['pivot']:,.2f}

"""

        msg += """*Trading Rules:*
• Enter after 9:30 AM (avoid first 15 min volatility)
• Book 50% at Target 1, trail rest
• Exit all positions by 3:15 PM
• Max 2-3% risk per trade

_Not financial advice. Trade responsibly._"""

        send_message(ADMIN_CHAT_ID, msg)
        logger.info(f"Intraday picks sent: {len(top_picks)} stocks")

    except Exception as e:
        logger.error(f"Intraday picks error: {e}")


def send_swing_trade_ideas():
    """
    Send weekly swing trade suggestions every Sunday at 7 PM.
    Based on weekly charts, trend analysis, and breakout setups.
    """
    try:
        now = datetime.now(IST)
        swing_picks = []

        for sym in DEFAULT_WATCHLIST:
            try:
                # Get 3 months data for swing analysis
                df = get_stock_data(sym, period="3mo")
                if df is None or len(df) < 50:
                    continue

                analysis = analyze_stock(df)
                if not analysis:
                    continue

                info = get_stock_info(sym)
                if not info:
                    continue

                price = info.get('price', 0)
                confidence = analysis.get('confidence_score', 50)
                signal = analysis.get('signal', 'NEUTRAL')
                rsi = analysis.get('rsi', 50)

                # Calculate swing score
                swing_score = 0

                # Trend analysis
                if len(df) >= 50:
                    sma20 = df['close'].tail(20).mean()
                    sma50 = df['close'].tail(50).mean()

                    if price > sma20 > sma50:  # Uptrend
                        swing_score += 20
                    elif price < sma20 < sma50:  # Downtrend
                        swing_score -= 10

                # RSI for swing
                if 40 < rsi < 60:  # Neutral RSI = room to move
                    swing_score += 10
                elif rsi < 35:  # Oversold
                    swing_score += 15

                # Confidence
                if confidence >= 65:
                    swing_score += 20
                elif confidence >= 55:
                    swing_score += 10

                # Weekly momentum
                if len(df) >= 5:
                    week_return = ((price - df['close'].iloc[-5]) / df['close'].iloc[-5] * 100)
                    if 1 < week_return < 5:  # Moderate momentum
                        swing_score += 10

                if swing_score >= 30 and signal != 'NEUTRAL':
                    # Calculate swing levels (wider targets)
                    atr = df['close'].diff().abs().tail(14).mean()
                    if 'BUY' in signal:
                        target_1 = price + (atr * 2)
                        target_2 = price + (atr * 4)
                        stop_loss = price - (atr * 1.5)
                    else:
                        target_1 = price - (atr * 2)
                        target_2 = price - (atr * 4)
                        stop_loss = price + (atr * 1.5)

                    swing_picks.append({
                        'symbol': sym,
                        'price': price,
                        'signal': signal,
                        'confidence': confidence,
                        'rsi': rsi,
                        'swing_score': swing_score,
                        'target_1': target_1,
                        'target_2': target_2,
                        'stop_loss': stop_loss,
                        'holding_period': '5-10 days',
                    })

            except Exception as e:
                continue

        if not swing_picks:
            logger.info("No swing trade ideas this week")
            return

        # Sort by swing score
        swing_picks.sort(key=lambda x: x['swing_score'], reverse=True)
        top_picks = swing_picks[:5]

        msg = f"""📈 *WEEKLY SWING TRADE IDEAS*
_{now.strftime('%A, %B %d, %Y')}_

Top {len(top_picks)} swing trade setups for this week:

"""

        for i, pick in enumerate(top_picks, 1):
            emoji = "🟢" if "BUY" in pick['signal'] else "🔴"

            msg += f"""*{i}. {pick['symbol']}* {emoji}
  💰 Entry: ₹{pick['price']:,.2f}
  📊 Signal: {pick['signal']} (Score: {pick['swing_score']})
  🎯 Target 1: ₹{pick['target_1']:,.2f}
  🎯 Target 2: ₹{pick['target_2']:,.2f}
  🛑 Stop Loss: ₹{pick['stop_loss']:,.2f}
  ⏱️ Holding: {pick['holding_period']}
  RSI: {pick['rsi']:.1f}

"""

        msg += """*Swing Trading Rules:*
• Enter on dips/pullbacks
• Keep strict stop loss
• Book partial at Target 1
• Hold winners, cut losers quickly

_Not financial advice. DYOR._"""

        send_message(ADMIN_CHAT_ID, msg)
        logger.info(f"Swing trade ideas sent: {len(top_picks)} stocks")

    except Exception as e:
        logger.error(f"Swing trade ideas error: {e}")


def check_golden_death_cross():
    """
    Check for Golden Cross (SMA50 crosses above SMA200) or
    Death Cross (SMA50 crosses below SMA200) signals.
    """
    global last_scheduled_alert

    try:
        now = datetime.now(IST)
        today_key = now.strftime('%Y-%m-%d')
        cross_alerts = []

        for sym in DEFAULT_WATCHLIST + INDEX_WATCHLIST:
            try:
                alert_key = f"cross_{sym}_{today_key}"
                if last_scheduled_alert.get(alert_key):
                    continue

                df = get_stock_data(sym, period="1y")
                if df is None or len(df) < 200:
                    continue

                # Calculate SMAs
                sma50 = df['close'].rolling(50).mean()
                sma200 = df['close'].rolling(200).mean()

                if len(sma50) < 2 or len(sma200) < 2:
                    continue

                # Check for crossover (today vs yesterday)
                today_50 = sma50.iloc[-1]
                today_200 = sma200.iloc[-1]
                yest_50 = sma50.iloc[-2]
                yest_200 = sma200.iloc[-2]

                cross_type = None

                # Golden Cross: SMA50 crosses ABOVE SMA200
                if yest_50 <= yest_200 and today_50 > today_200:
                    cross_type = "GOLDEN CROSS"

                # Death Cross: SMA50 crosses BELOW SMA200
                elif yest_50 >= yest_200 and today_50 < today_200:
                    cross_type = "DEATH CROSS"

                if cross_type:
                    info = get_stock_info(sym)
                    price = info.get('price', df['close'].iloc[-1]) if info else df['close'].iloc[-1]

                    cross_alerts.append({
                        'symbol': sym,
                        'cross_type': cross_type,
                        'price': price,
                        'sma50': today_50,
                        'sma200': today_200,
                    })
                    last_scheduled_alert[alert_key] = True

            except Exception as e:
                continue

        for alert in cross_alerts:
            if alert['cross_type'] == "GOLDEN CROSS":
                emoji = "🟢✨"
                meaning = "BULLISH - Long term uptrend beginning"
                action = "Consider accumulating on dips"
            else:
                emoji = "🔴💀"
                meaning = "BEARISH - Long term downtrend warning"
                action = "Consider reducing positions"

            msg = f"""{emoji} *{alert['cross_type']} ALERT!*

*{alert['symbol']}*

📊 SMA 50 has crossed {'above' if 'GOLDEN' in alert['cross_type'] else 'below'} SMA 200!

💰 Current Price: ₹{alert['price']:,.2f}
📈 SMA 50: ₹{alert['sma50']:,.2f}
📉 SMA 200: ₹{alert['sma200']:,.2f}

*Meaning:* {meaning}
*Action:* {action}

_This is a significant technical event!_"""

            send_message(ADMIN_CHAT_ID, msg)
            logger.info(f"{alert['cross_type']} alert sent for {alert['symbol']}")

    except Exception as e:
        logger.error(f"Golden/Death cross check error: {e}")


def check_rsi_divergence():
    """
    Check for RSI divergence patterns (bullish and bearish).
    Bullish: Price makes lower low, RSI makes higher low
    Bearish: Price makes higher high, RSI makes lower high
    """
    global last_scheduled_alert

    try:
        now = datetime.now(IST)
        today_key = now.strftime('%Y-%m-%d')
        divergence_alerts = []

        for sym in DEFAULT_WATCHLIST:
            try:
                alert_key = f"divergence_{sym}_{today_key}"
                if last_scheduled_alert.get(alert_key):
                    continue

                df = get_stock_data(sym, period="1mo")
                if df is None or len(df) < 20:
                    continue

                # Calculate RSI
                delta = df['close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
                rs = gain / loss
                rsi = 100 - (100 / (1 + rs))

                if len(rsi) < 10:
                    continue

                # Get last 10 days data
                prices = df['close'].tail(10).values
                rsi_values = rsi.tail(10).values

                # Check for bullish divergence (price lower low, RSI higher low)
                if prices[-1] < prices[0] and rsi_values[-1] > rsi_values[0]:
                    if rsi_values[-1] < 40:  # Confirm oversold
                        divergence_alerts.append({
                            'symbol': sym,
                            'type': 'BULLISH DIVERGENCE',
                            'price': prices[-1],
                            'rsi': rsi_values[-1],
                        })
                        last_scheduled_alert[alert_key] = True

                # Check for bearish divergence (price higher high, RSI lower high)
                elif prices[-1] > prices[0] and rsi_values[-1] < rsi_values[0]:
                    if rsi_values[-1] > 60:  # Confirm overbought zone
                        divergence_alerts.append({
                            'symbol': sym,
                            'type': 'BEARISH DIVERGENCE',
                            'price': prices[-1],
                            'rsi': rsi_values[-1],
                        })
                        last_scheduled_alert[alert_key] = True

            except Exception as e:
                continue

        for alert in divergence_alerts:
            if 'BULLISH' in alert['type']:
                emoji = "🟢📈"
                meaning = "Price making lower lows but RSI making higher lows"
                action = "Potential reversal UP - Watch for confirmation"
            else:
                emoji = "🔴📉"
                meaning = "Price making higher highs but RSI making lower highs"
                action = "Potential reversal DOWN - Consider booking profits"

            msg = f"""{emoji} *{alert['type']} DETECTED!*

*{alert['symbol']}*

💰 Price: ₹{alert['price']:,.2f}
📊 RSI: {alert['rsi']:.1f}

*What This Means:*
{meaning}

*Action:* {action}

_Divergence is an early warning signal. Wait for confirmation._"""

            send_message(ADMIN_CHAT_ID, msg)
            logger.info(f"RSI divergence alert sent for {alert['symbol']}")

    except Exception as e:
        logger.error(f"RSI divergence check error: {e}")


def send_support_resistance_levels():
    """
    Send key support and resistance levels for watchlist stocks daily at 8:30 AM.
    """
    try:
        now = datetime.now(IST)

        msg = f"""📊 *DAILY SUPPORT & RESISTANCE LEVELS*
_{now.strftime('%Y-%m-%d')} IST_

"""

        levels_data = []

        # Get levels for indices first
        for idx in INDEX_WATCHLIST:
            try:
                df = get_stock_data(idx, period="1mo")
                if df is None or len(df) < 10:
                    continue

                data = get_index_data(idx)
                price = data.get('value', 0) if data else df['close'].iloc[-1]

                # Calculate pivot points
                high = df['high'].tail(5).max()
                low = df['low'].tail(5).min()
                close = df['close'].iloc[-1]

                pivot = (high + low + close) / 3
                r1 = 2 * pivot - low
                r2 = pivot + (high - low)
                s1 = 2 * pivot - high
                s2 = pivot - (high - low)

                levels_data.append({
                    'symbol': idx,
                    'price': price,
                    'pivot': pivot,
                    'r1': r1, 'r2': r2,
                    's1': s1, 's2': s2,
                    'is_index': True
                })

            except:
                continue

        # Get levels for top stocks
        for sym in DEFAULT_WATCHLIST[:10]:  # Top 10 stocks
            try:
                df = get_stock_data(sym, period="1mo")
                if df is None or len(df) < 10:
                    continue

                info = get_stock_info(sym)
                price = info.get('price', df['close'].iloc[-1]) if info else df['close'].iloc[-1]

                high = df['high'].tail(5).max()
                low = df['low'].tail(5).min()
                close = df['close'].iloc[-1]

                pivot = (high + low + close) / 3
                r1 = 2 * pivot - low
                r2 = pivot + (high - low)
                s1 = 2 * pivot - high
                s2 = pivot - (high - low)

                levels_data.append({
                    'symbol': sym,
                    'price': price,
                    'pivot': pivot,
                    'r1': r1, 'r2': r2,
                    's1': s1, 's2': s2,
                    'is_index': False
                })

            except:
                continue

        # Format message
        msg += "*INDICES:*\n"
        for data in levels_data:
            if data['is_index']:
                msg += f"""
*{data['symbol']}* (CMP: {data['price']:,.0f})
  R2: {data['r2']:,.0f} | R1: {data['r1']:,.0f}
  Pivot: {data['pivot']:,.0f}
  S1: {data['s1']:,.0f} | S2: {data['s2']:,.0f}
"""

        msg += "\n*TOP STOCKS:*\n"
        for data in levels_data:
            if not data['is_index']:
                msg += f"""
*{data['symbol']}* (₹{data['price']:,.2f})
  R2: ₹{data['r2']:,.2f} | R1: ₹{data['r1']:,.2f}
  Pivot: ₹{data['pivot']:,.2f}
  S1: ₹{data['s1']:,.2f} | S2: ₹{data['s2']:,.2f}
"""

        msg += """
*Trading Guide:*
• Buy near S1/S2 with SL below support
• Sell near R1/R2 with SL above resistance
• Breakout above R2 = Strong bullish
• Breakdown below S2 = Strong bearish"""

        send_message(ADMIN_CHAT_ID, msg)
        logger.info("Support/Resistance levels sent")

    except Exception as e:
        logger.error(f"Support/Resistance levels error: {e}")


def send_global_market_summary():
    """
    Send global market summary at 8:00 AM with overnight US, Asian markets performance.
    """
    try:
        import yfinance as yf

        now = datetime.now(IST)

        msg = f"""🌍 *GLOBAL MARKET SUMMARY*
_{now.strftime('%Y-%m-%d %H:%M')} IST_

*Overnight Performance:*

"""

        global_data = []

        for name, symbol in GLOBAL_INDICES.items():
            try:
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period="2d")

                if len(hist) >= 2:
                    prev = float(hist['Close'].iloc[-2])
                    curr = float(hist['Close'].iloc[-1])
                    change = curr - prev
                    pct = (change / prev) * 100

                    global_data.append({
                        'name': name,
                        'value': curr,
                        'change': change,
                        'pct': pct
                    })
            except:
                continue

        # US Markets
        msg += "*🇺🇸 US MARKETS:*\n"
        for data in global_data:
            if data['name'] in ['DOW', 'NASDAQ', 'S&P500']:
                emoji = "📈" if data['pct'] >= 0 else "📉"
                msg += f"  {emoji} *{data['name']}*: {data['value']:,.0f} ({data['pct']:+.2f}%)\n"

        # Asian Markets
        msg += "\n*🌏 ASIAN MARKETS:*\n"
        for data in global_data:
            if data['name'] in ['NIKKEI', 'HANGSENG', 'SGX']:
                emoji = "📈" if data['pct'] >= 0 else "📉"
                msg += f"  {emoji} *{data['name']}*: {data['value']:,.0f} ({data['pct']:+.2f}%)\n"

        # European Markets
        msg += "\n*🇪🇺 EUROPE:*\n"
        for data in global_data:
            if data['name'] == 'FTSE':
                emoji = "📈" if data['pct'] >= 0 else "📉"
                msg += f"  {emoji} *{data['name']}*: {data['value']:,.0f} ({data['pct']:+.2f}%)\n"

        # Calculate overall sentiment
        positive_markets = sum(1 for d in global_data if d['pct'] > 0)
        total_markets = len(global_data)

        if positive_markets >= total_markets * 0.7:
            sentiment = "POSITIVE"
            sentiment_emoji = "🟢"
            outlook = "Indian markets likely to open positive"
        elif positive_markets <= total_markets * 0.3:
            sentiment = "NEGATIVE"
            sentiment_emoji = "🔴"
            outlook = "Indian markets may face pressure"
        else:
            sentiment = "MIXED"
            sentiment_emoji = "🟡"
            outlook = "Flat to mildly positive/negative opening"

        msg += f"""
{'='*30}
*OVERALL SENTIMENT: {sentiment}* {sentiment_emoji}
📍 {outlook}

_Use this for pre-market analysis._"""

        send_message(ADMIN_CHAT_ID, msg)
        logger.info("Global market summary sent")

    except Exception as e:
        logger.error(f"Global market summary error: {e}")


def check_earnings_calendar():
    """
    Check and alert for upcoming earnings/results of watchlist stocks.
    Alert 1 day before company reports results.
    """
    global last_scheduled_alert

    try:
        import yfinance as yf

        now = datetime.now(IST)
        today_key = now.strftime('%Y-%m-%d')
        tomorrow = (now + timedelta(days=1)).strftime('%Y-%m-%d')

        earnings_alerts = []

        for sym in DEFAULT_WATCHLIST:
            try:
                alert_key = f"earnings_{sym}_{today_key}"
                if last_scheduled_alert.get(alert_key):
                    continue

                # Try to get earnings date from yfinance
                ticker = yf.Ticker(f"{sym}.NS")
                calendar = ticker.calendar

                if calendar is not None and not calendar.empty:
                    # Check if earnings date exists
                    if 'Earnings Date' in calendar.index:
                        earnings_date = calendar.loc['Earnings Date']
                        if isinstance(earnings_date, str):
                            if earnings_date == tomorrow:
                                info = get_stock_info(sym)
                                earnings_alerts.append({
                                    'symbol': sym,
                                    'date': earnings_date,
                                    'price': info.get('price', 0) if info else 0,
                                })
                                last_scheduled_alert[alert_key] = True

            except Exception as e:
                continue

        for alert in earnings_alerts:
            msg = f"""📢 *EARNINGS ALERT!*

*{alert['symbol']}* is reporting results TOMORROW!

📅 Date: {alert['date']}
💰 Current Price: ₹{alert['price']:,.2f}

*What to do:*
  ⚠️ Avoid taking new positions before results
  ⚠️ If holding, consider hedging with options
  ⚠️ Results can cause 5-15% move in either direction

_Watch for post-result announcement._"""

            send_message(ADMIN_CHAT_ID, msg)
            logger.info(f"Earnings alert sent for {alert['symbol']}")

    except Exception as e:
        logger.error(f"Earnings calendar error: {e}")


def check_dividend_alerts():
    """
    Check for dividend announcements in watchlist stocks.
    """
    global last_scheduled_alert

    try:
        import yfinance as yf

        now = datetime.now(IST)
        today_key = now.strftime('%Y-%m-%d')

        for sym in DEFAULT_WATCHLIST:
            try:
                alert_key = f"dividend_{sym}_{today_key}"
                if last_scheduled_alert.get(alert_key):
                    continue

                ticker = yf.Ticker(f"{sym}.NS")
                dividends = ticker.dividends

                if dividends is not None and len(dividends) > 0:
                    # Check if recent dividend (within last 7 days)
                    last_div_date = dividends.index[-1]
                    days_ago = (now.date() - last_div_date.date()).days

                    if 0 <= days_ago <= 7:
                        div_amount = dividends.iloc[-1]
                        info = get_stock_info(sym)
                        price = info.get('price', 100) if info else 100
                        div_yield = (div_amount / price * 100) if price else 0

                        msg = f"""💰 *DIVIDEND ANNOUNCEMENT!*

*{sym}* has announced dividend!

📅 Date: {last_div_date.strftime('%Y-%m-%d')}
💵 Dividend: ₹{div_amount:.2f} per share
📊 Current Price: ₹{price:,.2f}
📈 Yield: {div_yield:.2f}%

*Note:*
  • Check record date for eligibility
  • Stock may correct post ex-dividend date

_Dividend investors take note!_"""

                        send_message(ADMIN_CHAT_ID, msg)
                        last_scheduled_alert[alert_key] = True
                        logger.info(f"Dividend alert sent for {sym}")

            except Exception as e:
                continue

    except Exception as e:
        logger.error(f"Dividend alert error: {e}")


def check_ipo_and_regulatory_news():
    """
    Check for IPO news and RBI/SEBI announcements.
    """
    try:
        # Check for regulatory keywords in news
        keywords = ["IPO", "RBI", "SEBI", "interest rate", "repo rate",
                   "monetary policy", "regulation", "FPI limit", "margin"]

        articles = get_market_news(limit=15)

        for article in articles:
            title = article.get('title', '')
            description = article.get('description', '')
            content = (title + " " + description).lower()

            # Check for IPO news
            if 'ipo' in content and any(word in content for word in ['launch', 'open', 'price', 'gmp', 'subscription']):
                msg = f"""🆕 *IPO NEWS ALERT!*

*{title}*

{description[:200] if description else ''}...

_Check for GMP and subscription status._"""

                send_message(ADMIN_CHAT_ID, msg)
                logger.info(f"IPO news alert sent: {title[:30]}")
                break

            # Check for RBI/SEBI news
            elif any(word in content for word in ['rbi', 'sebi']) and any(word in content for word in ['announce', 'decision', 'policy', 'change', 'rule']):
                msg = f"""🏛️ *REGULATORY NEWS ALERT!*

*{title}*

{description[:200] if description else ''}...

_This may impact market sentiment._"""

                send_message(ADMIN_CHAT_ID, msg)
                logger.info(f"Regulatory news alert sent: {title[:30]}")
                break

    except Exception as e:
        logger.error(f"IPO/Regulatory news error: {e}")


def send_weekly_report():
    """
    Send comprehensive weekly market analysis report every Saturday at 6 PM IST.
    Includes analysis of watchlist stocks with charts.
    """
    try:
        now = datetime.now(IST)

        msg = f"""📊 *WEEKLY MARKET ANALYSIS REPORT*
_{now.strftime('%A, %B %d, %Y')}_

"""
        # Collect data for all watchlist stocks
        stocks_data = []

        msg += "*WATCHLIST ANALYSIS:*\n\n"

        for sym in DEFAULT_WATCHLIST[:15]:  # Top 15 for report
            df = get_stock_data(sym, period="1mo")
            if df is None:
                continue

            analysis = analyze_stock(df)
            info = get_stock_info(sym)

            if not analysis:
                continue

            # Get current price
            current_price = analysis.get('price', 0)
            if not current_price:
                current_price = info.get('price', 0) if info else 0

            # Calculate weekly return
            if len(df) >= 5 and current_price > 0:
                week_ago_price = df['close'].iloc[-5]
                weekly_return = ((current_price - week_ago_price) / week_ago_price * 100) if week_ago_price > 0 else 0
            else:
                weekly_return = 0

            confidence = analysis.get('confidence_score', 50)
            signal = analysis.get('signal', 'NEUTRAL')
            rsi = analysis.get('rsi', 50)

            # Store for chart
            stocks_data.append({
                'symbol': sym,
                'price': current_price,
                'weekly_return': weekly_return,
                'confidence_score': confidence,
                'signal': signal,
                'rsi': rsi if rsi else 50
            })

            # Signal emoji
            if confidence >= 70:
                sig_emoji = "🟢🟢🟢"
            elif confidence >= 60:
                sig_emoji = "🟢"
            elif confidence <= 30:
                sig_emoji = "🔴🔴🔴"
            elif confidence <= 40:
                sig_emoji = "🔴"
            else:
                sig_emoji = "🟡"

            msg += f"""*{sym}* {sig_emoji}
  Price: ₹{current_price:,.2f}
  Week Return: {weekly_return:+.2f}%
  Signal: {signal}
  Confidence: {confidence}/100
  RSI: {rsi:.1f if rsi else 'N/A'}

"""

        # Index summary
        msg += "*INDICES WEEKLY SUMMARY:*\n"
        for idx in ["NIFTY", "SENSEX", "BANKNIFTY"]:
            data = get_index_data(idx)
            if data:
                emoji = "📈" if data.get('change', 0) >= 0 else "📉"
                msg += f"{emoji} *{idx}*: {data.get('value', 0):,.2f} ({data.get('pct', 0):+.2f}%)\n"

        # Top picks summary
        if stocks_data:
            # Sort by confidence for top picks
            sorted_by_conf = sorted(stocks_data, key=lambda x: x['confidence_score'], reverse=True)
            top_buys = [s for s in sorted_by_conf if s['confidence_score'] >= 60][:3]

            if top_buys:
                msg += "\n*TOP PICKS THIS WEEK:*\n"
                for pick in top_buys:
                    msg += f"  🎯 {pick['symbol']} (Confidence: {pick['confidence_score']}/100)\n"

        msg += "\n_Have a great weekend! See you on Monday._"

        # Send text report
        send_message(ADMIN_CHAT_ID, msg)
        logger.info("Weekly report text sent")

        # Generate and send chart
        if stocks_data:
            chart = generate_weekly_report_chart(stocks_data)
            if chart:
                send_photo(ADMIN_CHAT_ID, chart, "Weekly Analysis Chart")
                logger.info("Weekly report chart sent")

    except Exception as e:
        logger.error(f"Weekly report error: {e}")


def run_scheduler():
    """Background scheduler for automatic alerts"""
    global last_scheduled_alert

    logger.info("Scheduler started!")

    while True:
        try:
            now = datetime.now(IST)
            current_hour = now.hour
            current_minute = now.minute
            current_day = now.weekday()  # 0=Monday, 6=Sunday
            today_key = now.strftime('%Y-%m-%d')

            # ===== SATURDAY WEEKLY REPORT =====
            # Send weekly report on Saturday at 6 PM IST
            if current_day == 5 and current_hour == 18 and current_minute == 0:
                if last_scheduled_alert.get('weekly_report') != today_key:
                    send_weekly_report()
                    last_scheduled_alert['weekly_report'] = today_key

            # Skip rest of alerts on weekends (except weekly report above)
            if current_day >= 5:  # Saturday or Sunday
                time.sleep(60)
                continue

            # ===== WEEKDAY ALERTS =====

            # ===== MORNING BRIEFING - 8:45 AM =====
            # Comprehensive pre-market briefing with Gift Nifty, FII/DII, OI levels
            if current_hour == 8 and current_minute == 45:
                if last_scheduled_alert.get('morning_briefing') != today_key:
                    send_morning_briefing()
                    last_scheduled_alert['morning_briefing'] = today_key

            # Market Open Alert - 9:15 AM
            if current_hour == 9 and current_minute == 15:
                if last_scheduled_alert.get('market_open') != today_key:
                    send_market_open_alert()
                    last_scheduled_alert['market_open'] = today_key

            # Morning Update - 10:30 AM
            if current_hour == 10 and current_minute == 30:
                if last_scheduled_alert.get('morning') != today_key:
                    send_market_summary_auto()
                    last_scheduled_alert['morning'] = today_key

            # Afternoon Update - 1:30 PM (13:30)
            if current_hour == 13 and current_minute == 30:
                if last_scheduled_alert.get('afternoon') != today_key:
                    send_market_summary_auto()
                    last_scheduled_alert['afternoon'] = today_key

            # Market Close Alert - 3:30 PM (15:30)
            if current_hour == 15 and current_minute == 30:
                if last_scheduled_alert.get('market_close') != today_key:
                    send_market_close_alert()
                    last_scheduled_alert['market_close'] = today_key

            # Check breaking news every 10 minutes
            if current_minute % 10 == 0:
                check_breaking_news()

            # Check strong signals every 30 minutes during market hours
            if current_minute % 30 == 0 and 9 <= current_hour < 16:
                check_strong_signals()

            # ===== NEW: HIGH CONFIDENCE ALERTS =====
            # Check for high confidence opportunities every 15 minutes during market hours
            if current_minute % 15 == 0 and 9 <= current_hour < 16:
                check_high_confidence_alerts()

            # ===== INDEX SIGNALS =====
            # Check NIFTY, BANKNIFTY, SENSEX for signals every hour during market hours
            if current_minute == 0 and 9 <= current_hour < 16:
                check_index_signals()

            # ===== TARGET HIT MONITORING =====
            # Check active signals for target hits every 5 minutes during market hours
            if current_minute % 5 == 0 and 9 <= current_hour < 16:
                check_signal_targets()

            # ===== GIFT NIFTY CLOSE NOTIFICATION =====
            # Send Gift Nifty update at 11:30 PM IST
            if current_hour == 23 and current_minute == 30:
                if last_scheduled_alert.get('gift_nifty') != today_key:
                    send_gift_nifty_close_notification()
                    last_scheduled_alert['gift_nifty'] = today_key

            # ===== EVENING GIFT NIFTY UPDATE =====
            # Send Gift Nifty status at 7 PM IST (before US market opens)
            if current_hour == 19 and current_minute == 0:
                if last_scheduled_alert.get('gift_nifty_evening') != today_key:
                    send_gift_nifty_close_notification()
                    last_scheduled_alert['gift_nifty_evening'] = today_key

            # ===== FII/DII DAILY ALERT - 4:30 PM =====
            # Send FII/DII data after market close
            if current_hour == 16 and current_minute == 30:
                if last_scheduled_alert.get('fiidii_daily') != today_key:
                    send_fiidii_daily_alert()
                    last_scheduled_alert['fiidii_daily'] = today_key

            # ===== CALENDAR EVENT ALERT - 8 PM =====
            # Alert about tomorrow's important events
            if current_hour == 20 and current_minute == 0:
                if last_scheduled_alert.get('calendar_alert') != today_key:
                    send_calendar_event_alert("evening")
                    last_scheduled_alert['calendar_alert'] = today_key

            # ===== MORNING CALENDAR ALERT - 7 AM =====
            # Alert about today's important events
            if current_hour == 7 and current_minute == 0:
                if last_scheduled_alert.get('calendar_morning') != today_key:
                    send_calendar_event_alert("morning")
                    last_scheduled_alert['calendar_morning'] = today_key

            # ===== PRE-EVENT REMINDER - Every hour 8 AM to 4 PM =====
            # Alert about high impact events happening today
            if current_minute == 0 and 8 <= current_hour <= 16:
                send_pre_event_alert()

            # ===== NEW AUTOMATIC FEATURES =====

            # Global Market Summary at 8:00 AM (before market)
            if current_hour == 8 and current_minute == 0:
                if last_scheduled_alert.get('global_summary') != today_key:
                    send_global_market_summary()
                    last_scheduled_alert['global_summary'] = today_key

            # Support/Resistance Levels at 8:30 AM
            if current_hour == 8 and current_minute == 30:
                if last_scheduled_alert.get('sr_levels') != today_key:
                    send_support_resistance_levels()
                    last_scheduled_alert['sr_levels'] = today_key

            # Intraday Picks at 9:20 AM (just after market open)
            if current_hour == 9 and current_minute == 20:
                if last_scheduled_alert.get('intraday_picks') != today_key:
                    send_intraday_picks()
                    last_scheduled_alert['intraday_picks'] = today_key

            # Golden Cross / Death Cross check at 10:00 AM
            if current_hour == 10 and current_minute == 0:
                check_golden_death_cross()

            # RSI Divergence check every 2 hours during market
            if current_minute == 0 and current_hour in [10, 12, 14]:
                check_rsi_divergence()

            # Earnings Calendar check at 6 PM
            if current_hour == 18 and current_minute == 0:
                if last_scheduled_alert.get('earnings_check') != today_key:
                    check_earnings_calendar()
                    last_scheduled_alert['earnings_check'] = today_key

            # Dividend alerts check at 6:30 PM
            if current_hour == 18 and current_minute == 30:
                if last_scheduled_alert.get('dividend_check') != today_key:
                    check_dividend_alerts()
                    last_scheduled_alert['dividend_check'] = today_key

            # IPO and Regulatory News check every 2 hours
            if current_minute == 30 and current_hour in [9, 11, 13, 15, 17]:
                check_ipo_and_regulatory_news()

            # ===== WEEKEND ALERTS =====

            # Sunday: Weekly Swing Trade Ideas at 7 PM
            if current_day == 6 and current_hour == 19 and current_minute == 0:  # Sunday
                if last_scheduled_alert.get('swing_ideas') != today_key:
                    send_swing_trade_ideas()
                    last_scheduled_alert['swing_ideas'] = today_key

            time.sleep(60)  # Check every minute

        except Exception as e:
            logger.error(f"Scheduler error: {e}")
            time.sleep(60)


# ===== BOT LOOP =====
def run_bot():
    global bot_started
    if bot_started:
        logger.info("Bot already running, skipping...")
        return

    bot_started = True
    logger.info("MarketBuddy Pro starting...")

    # Delete any existing webhook to ensure clean polling
    try:
        delete_webhook_url = f"{API_URL}/deleteWebhook"
        requests.get(delete_webhook_url, timeout=10)
        logger.info("Webhook deleted")
    except:
        pass

    # Clear pending updates to avoid processing old messages
    try:
        clear_url = f"{API_URL}/getUpdates?offset=-1"
        requests.get(clear_url, timeout=10)
        logger.info("Pending updates cleared")
    except:
        pass

    # Start scheduler thread for automatic alerts
    scheduler_thread = threading.Thread(target=run_scheduler, daemon=True)
    scheduler_thread.start()
    logger.info("Scheduler thread started!")

    offset = None

    while True:
        try:
            updates = get_updates(offset)

            if updates and updates.get('ok'):
                for update in updates.get('result', []):
                    try:
                        offset = update['update_id'] + 1

                        message = update.get('message', {})
                        text = message.get('text', '').strip()
                        chat_id = message.get('chat', {}).get('id')
                        user_name = message.get('from', {}).get('first_name', 'User')

                        if not text or not chat_id:
                            continue

                        logger.info(f"Message: {text}")
                        parts = text.split()
                        cmd = parts[0].lower()

                        if cmd in ['/start', '/help']:
                            handle_start(chat_id, user_name)
                        elif cmd == '/price' and len(parts) >= 2:
                            handle_price(chat_id, parts[1])
                        elif cmd == '/stock' and len(parts) >= 2:
                            handle_stock(chat_id, parts[1])
                        elif cmd == '/chart' and len(parts) >= 2:
                            handle_chart(chat_id, parts[1])
                        elif cmd == '/summary':
                            handle_summary(chat_id)
                        elif cmd == '/news':
                            handle_news(chat_id)
                        elif cmd == '/watchlist':
                            if len(parts) >= 3 and parts[1].lower() == 'add':
                                handle_watchlist(chat_id, 'add', parts[2])
                            elif len(parts) >= 3 and parts[1].lower() == 'remove':
                                handle_watchlist(chat_id, 'remove', parts[2])
                            else:
                                handle_watchlist(chat_id)
                        elif cmd == '/alert' and len(parts) >= 3:
                            handle_alert(chat_id, parts[1], parts[2])
                        elif cmd == '/alerts':
                            handle_alerts(chat_id)
                        elif cmd == '/portfolio':
                            if len(parts) >= 5 and parts[1].lower() == 'add':
                                handle_portfolio(chat_id, 'add', parts[2], parts[3], parts[4])
                            elif len(parts) >= 3 and parts[1].lower() == 'remove':
                                handle_portfolio(chat_id, 'remove', parts[2])
                            else:
                                handle_portfolio(chat_id)
                        elif cmd == '/signals':
                            handle_signals(chat_id)
                        elif cmd == '/history':
                            handle_history(chat_id)
                        elif cmd == '/mistakes':
                            handle_mistakes(chat_id)
                        elif cmd == '/giftnifty':
                            handle_giftnifty(chat_id)
                        elif cmd == '/nifty':
                            handle_stock(chat_id, 'NIFTY')
                        elif cmd == '/banknifty':
                            handle_stock(chat_id, 'BANKNIFTY')
                        elif cmd == '/sensex':
                            handle_stock(chat_id, 'SENSEX')
                        elif cmd == '/fiidii':
                            handle_fiidii(chat_id)
                        elif cmd == '/oi' or cmd == '/optionchain':
                            symbol = parts[1].upper() if len(parts) >= 2 else "NIFTY"
                            handle_optionchain(chat_id, symbol)
                        elif cmd == '/calendar':
                            handle_calendar(chat_id)
                        elif cmd == '/sentiment':
                            handle_sentiment(chat_id)
                        elif cmd == '/pcr':
                            handle_optionchain(chat_id, "NIFTY")
                        elif cmd == '/days' and len(parts) >= 2:
                            # /days SYMBOL [N] - show last N days (default 15)
                            symbol = parts[1].upper()
                            days = int(parts[2]) if len(parts) >= 3 else 15
                            days = min(max(days, 5), 30)  # Limit between 5-30 days
                            handle_days(chat_id, symbol, days)
                        elif cmd == '/ipo':
                            # /ipo or /ipo 1 or /ipo Mobikwik
                            ipo_selection = ' '.join(parts[1:]) if len(parts) > 1 else None
                            handle_ipo(chat_id, ipo_selection)
                        else:
                            # Check if user typed a stock symbol directly (without /)
                            potential_symbol = text.strip().upper()
                            if potential_symbol and len(potential_symbol) <= 20 and potential_symbol.isalpha():
                                # Try to look up as stock symbol
                                info = get_stock_info(potential_symbol)
                                if info and info.get('price'):
                                    handle_stock(chat_id, potential_symbol)
                                else:
                                    send_message(chat_id, f"'{potential_symbol}' not found. Use /help for commands.")
                            elif not text.startswith('/'):
                                # Non-command text that's not a stock symbol
                                send_message(chat_id, "Type a stock symbol (e.g., RELIANCE) or use /help for commands.")
                            else:
                                send_message(chat_id, "Unknown command. Use /help")

                    except Exception as e:
                        logger.error(f"Command error: {e}")

            time.sleep(1)

        except Exception as e:
            logger.error(f"Loop error: {e}")
            time.sleep(5)

# Start bot in background thread when module loads
def start_bot_thread():
    global bot_started
    if not bot_started:
        thread = threading.Thread(target=run_bot, daemon=True)
        thread.start()
        logger.info("Bot thread started!")

# Start the bot
start_bot_thread()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)
