#!/usr/bin/env python3
"""
MarketBuddy Pro - Advanced Stock Analysis Telegram Bot
Features: Watchlist, Alerts, Charts, News, FII/DII, Option Chain, Screener & more
"""

import os
import requests
import logging
import time
import json
import io
from datetime import datetime, timedelta
from flask import Flask
import threading

import yfinance as yf
import pandas as pd
import numpy as np
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.trend import MACD, SMAIndicator, EMAIndicator
from ta.volatility import BollingerBands, AverageTrueRange
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mplfinance as mpf

# Flask app
app = Flask(__name__)

@app.route('/')
def home():
    return "MarketBuddy Pro is running!"

@app.route('/health')
def health():
    return "OK"

# Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ===== CONFIG =====
BOT_TOKEN = os.environ.get("BOT_TOKEN", "8241472299:AAGhewPu-VZFXpuLyVBlhRZYrGPgxpCb7mI")
ADMIN_CHAT_ID = os.environ.get("ADMIN_CHAT_ID", "2110121880")
NEWSAPI_KEY = os.environ.get("NEWSAPI_KEY", "5df4112f47a84dd6b7aa6e09a5be71db")
API_URL = f"https://api.telegram.org/bot{BOT_TOKEN}"

# Indian Indices
INDIAN_INDICES = {
    "NIFTY": "^NSEI", "NIFTY50": "^NSEI",
    "SENSEX": "^BSESN",
    "BANKNIFTY": "^NSEBANK",
    "NIFTYIT": "^CNXIT",
    "NIFTYPHARMA": "^CNXPHARMA"
}

# Default Watchlist
DEFAULT_WATCHLIST = ["RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK",
                     "HINDUNILVR", "SBIN", "BHARTIARTL", "KOTAKBANK", "ITC",
                     "LT", "AXISBANK", "BAJFINANCE", "MARUTI", "TITAN"]

# User Data Storage (in-memory, resets on restart)
user_data = {}  # {chat_id: {"watchlist": [], "portfolio": {}, "alerts": []}}

# ===== TELEGRAM API =====
def send_message(chat_id, text, parse_mode="Markdown"):
    try:
        url = f"{API_URL}/sendMessage"
        data = {"chat_id": chat_id, "text": text, "parse_mode": parse_mode}
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

# ===== USER DATA MANAGEMENT =====
def get_user_data(chat_id):
    chat_id = str(chat_id)
    if chat_id not in user_data:
        user_data[chat_id] = {
            "watchlist": DEFAULT_WATCHLIST.copy()[:5],
            "portfolio": {},
            "alerts": []
        }
    return user_data[chat_id]

# ===== DATA FETCHERS =====

def format_symbol(symbol):
    """Format symbol for yfinance"""
    symbol = symbol.upper().strip()
    if symbol in INDIAN_INDICES:
        return INDIAN_INDICES[symbol]
    elif '.' not in symbol and '^' not in symbol:
        return f"{symbol}.NS"
    return symbol

def get_stock_data(symbol, period="3mo"):
    try:
        symbol = format_symbol(symbol)
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=period)
        if df.empty:
            return pd.DataFrame()
        df.columns = [c.lower() for c in df.columns]
        return df
    except Exception as e:
        logger.error(f"Fetch error {symbol}: {e}")
        return pd.DataFrame()

def get_stock_info(symbol):
    try:
        symbol = format_symbol(symbol)
        ticker = yf.Ticker(symbol)
        info = ticker.info
        return {
            "name": info.get("longName", info.get("shortName", symbol)),
            "symbol": symbol,
            "price": info.get("currentPrice", info.get("regularMarketPrice", 0)),
            "previous_close": info.get("previousClose", 0),
            "open": info.get("regularMarketOpen", 0),
            "high": info.get("dayHigh", info.get("regularMarketDayHigh", 0)),
            "low": info.get("dayLow", info.get("regularMarketDayLow", 0)),
            "volume": info.get("volume", info.get("regularMarketVolume", 0)),
            "market_cap": info.get("marketCap", 0),
            "pe_ratio": info.get("trailingPE", 0),
            "pb_ratio": info.get("priceToBook", 0),
            "dividend_yield": info.get("dividendYield", 0),
            "52w_high": info.get("fiftyTwoWeekHigh", 0),
            "52w_low": info.get("fiftyTwoWeekLow", 0),
            "sector": info.get("sector", "N/A"),
            "industry": info.get("industry", "N/A"),
        }
    except Exception as e:
        logger.error(f"Info error {symbol}: {e}")
        return {}

def get_index_data(index_name="NIFTY"):
    try:
        symbol = INDIAN_INDICES.get(index_name.upper(), "^NSEI")
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period="2d")
        if len(hist) >= 2:
            prev = float(hist['Close'].iloc[-2])
            curr = float(hist['Close'].iloc[-1])
            change = curr - prev
            pct = (change / prev) * 100
            return {
                "name": index_name.upper(),
                "value": curr,
                "change": change,
                "pct": pct,
                "prev": prev
            }
    except Exception as e:
        logger.error(f"Index error: {e}")
    return {}

# ===== NSE DATA =====
def get_nse_data(endpoint):
    """Fetch data from NSE India"""
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Accept-Language": "en-US,en;q=0.9",
            "Accept-Encoding": "gzip, deflate, br"
        }
        session = requests.Session()
        session.get("https://www.nseindia.com", headers=headers, timeout=10)
        response = session.get(f"https://www.nseindia.com/api/{endpoint}", headers=headers, timeout=10)
        return response.json()
    except Exception as e:
        logger.error(f"NSE error: {e}")
        return None

def get_fii_dii_data():
    """Get FII/DII data from NSE"""
    try:
        data = get_nse_data("fiidiiTradeReact")
        if data:
            return data
    except:
        pass
    return None

def get_option_chain(symbol="NIFTY"):
    """Get option chain data"""
    try:
        data = get_nse_data(f"option-chain-indices?symbol={symbol}")
        if data and 'records' in data:
            records = data['records']
            return {
                "spot_price": records.get('underlyingValue', 0),
                "pcr": calculate_pcr(records.get('data', [])),
                "max_pain": calculate_max_pain(records.get('data', [])),
                "timestamp": records.get('timestamp', '')
            }
    except Exception as e:
        logger.error(f"Option chain error: {e}")
    return None

def calculate_pcr(data):
    """Calculate Put-Call Ratio"""
    try:
        total_ce_oi = sum(item.get('CE', {}).get('openInterest', 0) for item in data if 'CE' in item)
        total_pe_oi = sum(item.get('PE', {}).get('openInterest', 0) for item in data if 'PE' in item)
        if total_ce_oi > 0:
            return round(total_pe_oi / total_ce_oi, 2)
    except:
        pass
    return 0

def calculate_max_pain(data):
    """Calculate max pain (simplified)"""
    try:
        strike_oi = {}
        for item in data:
            strike = item.get('strikePrice', 0)
            ce_oi = item.get('CE', {}).get('openInterest', 0)
            pe_oi = item.get('PE', {}).get('openInterest', 0)
            strike_oi[strike] = ce_oi + pe_oi
        if strike_oi:
            return max(strike_oi, key=strike_oi.get)
    except:
        pass
    return 0

def get_top_gainers_losers():
    """Get top gainers and losers"""
    try:
        data = get_nse_data("equity-stockIndices?index=NIFTY%2050")
        if data and 'data' in data:
            stocks = data['data'][1:]  # Skip index data
            sorted_stocks = sorted(stocks, key=lambda x: x.get('pChange', 0), reverse=True)
            gainers = sorted_stocks[:5]
            losers = sorted_stocks[-5:][::-1]
            return {"gainers": gainers, "losers": losers}
    except Exception as e:
        logger.error(f"Gainers/Losers error: {e}")
    return None

# ===== NEWS API =====
def get_market_news(query="Indian stock market", limit=5):
    """Fetch news from NewsAPI"""
    try:
        url = "https://newsapi.org/v2/everything"
        params = {
            "q": query,
            "apiKey": NEWSAPI_KEY,
            "language": "en",
            "sortBy": "publishedAt",
            "pageSize": limit
        }
        response = requests.get(url, params=params, timeout=10)
        data = response.json()
        if data.get("status") == "ok":
            return data.get("articles", [])
    except Exception as e:
        logger.error(f"News error: {e}")
    return []

# ===== TECHNICAL ANALYSIS =====
def analyze_stock(df):
    if df.empty or len(df) < 20:
        return None

    df = df.copy()
    df.columns = [c.lower() for c in df.columns]

    try:
        # RSI
        df['rsi'] = RSIIndicator(close=df['close'], window=14).rsi()

        # MACD
        macd = MACD(close=df['close'])
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        df['macd_hist'] = macd.macd_diff()

        # Moving Averages
        df['sma_20'] = SMAIndicator(close=df['close'], window=20).sma_indicator()
        df['sma_50'] = SMAIndicator(close=df['close'], window=50).sma_indicator()
        df['sma_200'] = SMAIndicator(close=df['close'], window=200).sma_indicator()
        df['ema_12'] = EMAIndicator(close=df['close'], window=12).ema_indicator()
        df['ema_26'] = EMAIndicator(close=df['close'], window=26).ema_indicator()

        # Bollinger Bands
        bb = BollingerBands(close=df['close'])
        df['bb_upper'] = bb.bollinger_hband()
        df['bb_middle'] = bb.bollinger_mavg()
        df['bb_lower'] = bb.bollinger_lband()

        # ATR
        df['atr'] = AverageTrueRange(high=df['high'], low=df['low'], close=df['close']).average_true_range()

        # Stochastic
        stoch = StochasticOscillator(high=df['high'], low=df['low'], close=df['close'])
        df['stoch_k'] = stoch.stoch()
        df['stoch_d'] = stoch.stoch_signal()

    except Exception as e:
        logger.error(f"Analysis error: {e}")

    latest = df.iloc[-1]
    prev = df.iloc[-2] if len(df) > 1 else latest

    indicators = {}
    signals = {"buy": [], "sell": []}
    buy_score = sell_score = 0

    # RSI Analysis
    rsi = latest.get('rsi')
    if rsi and not pd.isna(rsi):
        if rsi < 30:
            indicators['RSI'] = f"🟢 OVERSOLD ({rsi:.1f})"
            signals['buy'].append("RSI Oversold")
            buy_score += 2
        elif rsi > 70:
            indicators['RSI'] = f"🔴 OVERBOUGHT ({rsi:.1f})"
            signals['sell'].append("RSI Overbought")
            sell_score += 2
        elif rsi < 40:
            indicators['RSI'] = f"🟡 WEAK ({rsi:.1f})"
            buy_score += 1
        elif rsi > 60:
            indicators['RSI'] = f"🟡 STRONG ({rsi:.1f})"
            sell_score += 1
        else:
            indicators['RSI'] = f"⚪ NEUTRAL ({rsi:.1f})"

    # MACD Analysis
    macd_val = latest.get('macd')
    macd_sig = latest.get('macd_signal')
    prev_macd = prev.get('macd', 0)
    prev_sig = prev.get('macd_signal', 0)

    if macd_val and macd_sig and not (pd.isna(macd_val) or pd.isna(macd_sig)):
        if prev_macd <= prev_sig and macd_val > macd_sig:
            indicators['MACD'] = "🟢 BULLISH CROSSOVER"
            signals['buy'].append("MACD Bullish Crossover")
            buy_score += 2
        elif prev_macd >= prev_sig and macd_val < macd_sig:
            indicators['MACD'] = "🔴 BEARISH CROSSOVER"
            signals['sell'].append("MACD Bearish Crossover")
            sell_score += 2
        elif macd_val > macd_sig:
            indicators['MACD'] = "🟢 BULLISH"
            buy_score += 1
        else:
            indicators['MACD'] = "🔴 BEARISH"
            sell_score += 1

    # SMA Analysis
    sma20 = latest.get('sma_20')
    sma50 = latest.get('sma_50')
    sma200 = latest.get('sma_200')
    close = latest.get('close')

    if sma20 and sma50 and not (pd.isna(sma20) or pd.isna(sma50)):
        if sma20 > sma50:
            indicators['SMA 20/50'] = "🟢 BULLISH"
            buy_score += 1
        else:
            indicators['SMA 20/50'] = "🔴 BEARISH"
            sell_score += 1

    if close and sma200 and not pd.isna(sma200):
        if close > sma200:
            indicators['SMA 200'] = "🟢 ABOVE (Bullish)"
            buy_score += 1
        else:
            indicators['SMA 200'] = "🔴 BELOW (Bearish)"
            sell_score += 1

    # Bollinger Bands
    bb_low = latest.get('bb_lower')
    bb_up = latest.get('bb_upper')
    if close and bb_low and bb_up and not (pd.isna(bb_low) or pd.isna(bb_up)):
        if close <= bb_low:
            indicators['Bollinger'] = "🟢 LOWER BAND (Oversold)"
            signals['buy'].append("At Bollinger Lower Band")
            buy_score += 1
        elif close >= bb_up:
            indicators['Bollinger'] = "🔴 UPPER BAND (Overbought)"
            signals['sell'].append("At Bollinger Upper Band")
            sell_score += 1
        else:
            indicators['Bollinger'] = "⚪ WITHIN BANDS"

    # Stochastic
    stoch_k = latest.get('stoch_k')
    stoch_d = latest.get('stoch_d')
    if stoch_k and not pd.isna(stoch_k):
        if stoch_k < 20:
            indicators['Stochastic'] = f"🟢 OVERSOLD ({stoch_k:.1f})"
            buy_score += 1
        elif stoch_k > 80:
            indicators['Stochastic'] = f"🔴 OVERBOUGHT ({stoch_k:.1f})"
            sell_score += 1
        else:
            indicators['Stochastic'] = f"⚪ NEUTRAL ({stoch_k:.1f})"

    # Calculate overall signal
    net = buy_score - sell_score
    if net >= 5:
        signal = "🟢 STRONG BUY"
    elif net >= 2:
        signal = "🟢 BUY"
    elif net <= -5:
        signal = "🔴 STRONG SELL"
    elif net <= -2:
        signal = "🔴 SELL"
    else:
        signal = "🟡 NEUTRAL"

    # Support & Resistance
    recent_high = df['high'].tail(20).max()
    recent_low = df['low'].tail(20).min()
    pivot = (recent_high + recent_low + close) / 3
    r1 = 2 * pivot - recent_low
    s1 = 2 * pivot - recent_high

    return {
        "price": close,
        "change": close - latest.get('open', close),
        "change_pct": ((close - latest.get('open', close)) / latest.get('open', close) * 100) if latest.get('open') else 0,
        "volume": latest.get('volume', 0),
        "rsi": rsi,
        "macd": macd_val,
        "sma_20": sma20,
        "sma_50": sma50,
        "sma_200": sma200,
        "signal": signal,
        "strength": net,
        "indicators": indicators,
        "buy_signals": signals['buy'],
        "sell_signals": signals['sell'],
        "support": s1,
        "resistance": r1,
        "pivot": pivot,
        "atr": latest.get('atr')
    }

# ===== CHART GENERATION =====
def generate_chart(symbol, period="3mo"):
    """Generate candlestick chart with indicators"""
    try:
        sym = format_symbol(symbol)
        ticker = yf.Ticker(sym)
        df = ticker.history(period=period)

        if df.empty or len(df) < 20:
            return None

        # Calculate indicators for chart
        df['SMA20'] = df['Close'].rolling(window=20).mean()
        df['SMA50'] = df['Close'].rolling(window=50).mean()

        # Bollinger Bands
        df['BB_Middle'] = df['Close'].rolling(window=20).mean()
        df['BB_Std'] = df['Close'].rolling(window=20).std()
        df['BB_Upper'] = df['BB_Middle'] + 2 * df['BB_Std']
        df['BB_Lower'] = df['BB_Middle'] - 2 * df['BB_Std']

        # Create chart
        fig, axes = plt.subplots(3, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [3, 1, 1]})

        # Price chart with candlesticks
        ax1 = axes[0]
        ax1.set_title(f"{symbol.upper()} - Price Chart", fontsize=14, fontweight='bold')

        # Plot price as line (simplified)
        ax1.plot(df.index, df['Close'], label='Close', color='blue', linewidth=1.5)
        ax1.plot(df.index, df['SMA20'], label='SMA 20', color='orange', linewidth=1)
        ax1.plot(df.index, df['SMA50'], label='SMA 50', color='red', linewidth=1)
        ax1.fill_between(df.index, df['BB_Upper'], df['BB_Lower'], alpha=0.2, color='gray', label='Bollinger Bands')
        ax1.legend(loc='upper left', fontsize=8)
        ax1.grid(True, alpha=0.3)
        ax1.set_ylabel('Price')

        # Volume
        ax2 = axes[1]
        colors = ['green' if df['Close'].iloc[i] >= df['Open'].iloc[i] else 'red' for i in range(len(df))]
        ax2.bar(df.index, df['Volume'], color=colors, alpha=0.7)
        ax2.set_ylabel('Volume')
        ax2.grid(True, alpha=0.3)

        # RSI
        ax3 = axes[2]
        rsi = RSIIndicator(close=df['Close'], window=14).rsi()
        ax3.plot(df.index, rsi, color='purple', linewidth=1)
        ax3.axhline(y=70, color='red', linestyle='--', alpha=0.5)
        ax3.axhline(y=30, color='green', linestyle='--', alpha=0.5)
        ax3.fill_between(df.index, 30, 70, alpha=0.1, color='gray')
        ax3.set_ylabel('RSI')
        ax3.set_ylim(0, 100)
        ax3.grid(True, alpha=0.3)

        plt.tight_layout()

        # Save to buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        plt.close()

        return buf
    except Exception as e:
        logger.error(f"Chart error: {e}")
        return None

# ===== SCREENER =====
def screen_stocks(criteria="oversold"):
    """Screen stocks based on criteria"""
    results = []
    stocks_to_check = DEFAULT_WATCHLIST[:10]

    for symbol in stocks_to_check:
        try:
            df = get_stock_data(symbol)
            if df.empty:
                continue

            analysis = analyze_stock(df)
            if not analysis:
                continue

            rsi = analysis.get('rsi', 50)

            if criteria == "oversold" and rsi and rsi < 30:
                results.append({"symbol": symbol, "rsi": rsi, "price": analysis['price']})
            elif criteria == "overbought" and rsi and rsi > 70:
                results.append({"symbol": symbol, "rsi": rsi, "price": analysis['price']})
            elif criteria == "bullish" and analysis['strength'] >= 3:
                results.append({"symbol": symbol, "signal": analysis['signal'], "price": analysis['price']})
            elif criteria == "bearish" and analysis['strength'] <= -3:
                results.append({"symbol": symbol, "signal": analysis['signal'], "price": analysis['price']})

        except Exception as e:
            logger.error(f"Screen error {symbol}: {e}")

    return results

# ===== COMMAND HANDLERS =====

def handle_start(chat_id, user_name):
    msg = f"""🚀 *Welcome to MarketBuddy Pro!*

Hello {user_name}! I'm your advanced stock analysis assistant.

📊 *ANALYSIS COMMANDS*
/stock SYMBOL - Full technical analysis
/price SYMBOL - Quick price check
/chart SYMBOL - Candlestick chart
/compare STOCK1 STOCK2 - Compare stocks

📈 *MARKET DATA*
/summary - NIFTY, SENSEX, BANKNIFTY
/topgainers - Top 5 gainers
/toplosers - Top 5 losers
/fii - FII/DII data
/options NIFTY - Option chain data

📰 *NEWS & ALERTS*
/news - Latest market news
/alert SYMBOL PRICE - Set price alert
/alerts - View your alerts

📋 *PORTFOLIO*
/watchlist - Your watchlist
/watchlist add SYMBOL - Add to watchlist
/watchlist remove SYMBOL - Remove
/portfolio - Your holdings
/portfolio add SYMBOL QTY PRICE - Add holding

🔍 *SCREENER*
/screen oversold - Find oversold stocks
/screen overbought - Find overbought
/screen bullish - Strong buy signals

💡 *Try:* /stock RELIANCE or /chart TCS"""

    send_message(chat_id, msg)

def handle_price(chat_id, symbol):
    """Quick price check"""
    info = get_stock_info(symbol)
    if not info or not info.get('price'):
        send_message(chat_id, f"❌ No data for {symbol}")
        return

    price = info['price']
    prev = info.get('previous_close', price)
    change = price - prev
    pct = (change / prev * 100) if prev else 0
    emoji = "📈" if change >= 0 else "📉"

    msg = f"""*{info.get('name', symbol)}*

💰 *Price:* ₹{price:,.2f}
{emoji} *Change:* {change:+,.2f} ({pct:+.2f}%)

📊 Open: ₹{info.get('open', 0):,.2f}
📈 High: ₹{info.get('high', 0):,.2f}
📉 Low: ₹{info.get('low', 0):,.2f}
📦 Volume: {info.get('volume', 0):,}"""

    send_message(chat_id, msg)

def handle_stock(chat_id, symbol):
    """Full stock analysis"""
    send_message(chat_id, f"🔄 Analyzing *{symbol}*...")

    df = get_stock_data(symbol, period="6mo")
    if df.empty:
        send_message(chat_id, f"❌ No data for {symbol}")
        return

    info = get_stock_info(symbol)
    analysis = analyze_stock(df)

    if not analysis:
        send_message(chat_id, "❌ Insufficient data")
        return

    price = analysis['price']
    prev = info.get('previous_close', price)
    change = price - prev
    pct = (change / prev * 100) if prev else 0
    emoji = "📈" if change >= 0 else "📉"

    msg = f"""📊 *{info.get('name', symbol)}*
{'='*30}

💰 *Price:* ₹{price:,.2f}
{emoji} *Change:* {change:+,.2f} ({pct:+.2f}%)

*{analysis['signal']}*
Signal Strength: {analysis['strength']}

📈 *TECHNICAL INDICATORS*
"""

    for ind, val in analysis['indicators'].items():
        msg += f"• {ind}: {val}\n"

    if analysis.get('buy_signals'):
        msg += f"\n✅ *Buy Signals:* {', '.join(analysis['buy_signals'])}"

    if analysis.get('sell_signals'):
        msg += f"\n❌ *Sell Signals:* {', '.join(analysis['sell_signals'])}"

    msg += f"""

📐 *LEVELS*
• Support: ₹{analysis.get('support', 0):,.2f}
• Resistance: ₹{analysis.get('resistance', 0):,.2f}
• Pivot: ₹{analysis.get('pivot', 0):,.2f}

📊 *MORE INFO*
• 52W High: ₹{info.get('52w_high', 0):,.2f}
• 52W Low: ₹{info.get('52w_low', 0):,.2f}
• P/E: {info.get('pe_ratio', 0):.2f}
• Sector: {info.get('sector', 'N/A')}

_Updated: {datetime.now().strftime('%H:%M:%S')}_"""

    send_message(chat_id, msg)

def handle_chart(chat_id, symbol):
    """Send chart image"""
    send_message(chat_id, f"📊 Generating chart for *{symbol}*...")

    chart = generate_chart(symbol)
    if chart:
        send_photo(chat_id, chart, f"📈 *{symbol.upper()}* - Technical Chart")
    else:
        send_message(chat_id, f"❌ Could not generate chart for {symbol}")

def handle_summary(chat_id):
    """Market summary"""
    msg = f"""📊 *MARKET SUMMARY*
_{datetime.now().strftime('%Y-%m-%d %H:%M')}_

"""
    for idx in ["NIFTY", "SENSEX", "BANKNIFTY"]:
        data = get_index_data(idx)
        if data:
            emoji = "📈" if data.get('change', 0) >= 0 else "📉"
            msg += f"""*{idx}*
  {data.get('value', 0):,.2f}
  {emoji} {data.get('change', 0):+,.2f} ({data.get('pct', 0):+.2f}%)

"""
    send_message(chat_id, msg)

def handle_news(chat_id):
    """Market news"""
    send_message(chat_id, "📰 Fetching latest news...")

    articles = get_market_news(limit=5)
    if not articles:
        send_message(chat_id, "❌ No news available")
        return

    msg = "📰 *LATEST MARKET NEWS*\n\n"
    for i, article in enumerate(articles, 1):
        title = article.get('title', '')[:100]
        source = article.get('source', {}).get('name', '')
        msg += f"{i}. *{title}*\n   _{source}_\n\n"

    send_message(chat_id, msg)

def handle_topgainers(chat_id):
    """Top gainers"""
    send_message(chat_id, "📈 Fetching top gainers...")

    data = get_top_gainers_losers()
    if not data or not data.get('gainers'):
        send_message(chat_id, "❌ Data not available")
        return

    msg = "📈 *TOP 5 GAINERS*\n\n"
    for stock in data['gainers'][:5]:
        symbol = stock.get('symbol', '')
        ltp = stock.get('lastPrice', 0)
        change = stock.get('pChange', 0)
        msg += f"🟢 *{symbol}*: ₹{ltp:,.2f} (+{change:.2f}%)\n"

    send_message(chat_id, msg)

def handle_toplosers(chat_id):
    """Top losers"""
    send_message(chat_id, "📉 Fetching top losers...")

    data = get_top_gainers_losers()
    if not data or not data.get('losers'):
        send_message(chat_id, "❌ Data not available")
        return

    msg = "📉 *TOP 5 LOSERS*\n\n"
    for stock in data['losers'][:5]:
        symbol = stock.get('symbol', '')
        ltp = stock.get('lastPrice', 0)
        change = stock.get('pChange', 0)
        msg += f"🔴 *{symbol}*: ₹{ltp:,.2f} ({change:.2f}%)\n"

    send_message(chat_id, msg)

def handle_options(chat_id, symbol="NIFTY"):
    """Option chain data"""
    send_message(chat_id, f"📊 Fetching {symbol} options data...")

    data = get_option_chain(symbol)
    if not data:
        send_message(chat_id, "❌ Option data not available")
        return

    msg = f"""📊 *{symbol} OPTIONS DATA*

💰 *Spot Price:* ₹{data.get('spot_price', 0):,.2f}
📊 *PCR:* {data.get('pcr', 0)}
🎯 *Max Pain:* ₹{data.get('max_pain', 0):,}

_PCR > 1 = Bullish, PCR < 1 = Bearish_
_{data.get('timestamp', '')}_"""

    send_message(chat_id, msg)

def handle_fii(chat_id):
    """FII/DII data"""
    send_message(chat_id, "📊 Fetching FII/DII data...")

    data = get_fii_dii_data()
    if not data:
        send_message(chat_id, "❌ FII/DII data not available. Try again later.")
        return

    msg = "📊 *FII/DII ACTIVITY*\n\n"
    # Format based on actual NSE response
    send_message(chat_id, msg + "Data fetched from NSE")

def handle_watchlist(chat_id, action=None, symbol=None):
    """Manage watchlist"""
    user = get_user_data(chat_id)

    if action == "add" and symbol:
        if symbol.upper() not in user['watchlist']:
            user['watchlist'].append(symbol.upper())
            send_message(chat_id, f"✅ Added *{symbol.upper()}* to watchlist")
        else:
            send_message(chat_id, f"⚠️ *{symbol.upper()}* already in watchlist")
    elif action == "remove" and symbol:
        if symbol.upper() in user['watchlist']:
            user['watchlist'].remove(symbol.upper())
            send_message(chat_id, f"✅ Removed *{symbol.upper()}* from watchlist")
        else:
            send_message(chat_id, f"⚠️ *{symbol.upper()}* not in watchlist")
    else:
        # Show watchlist
        if not user['watchlist']:
            send_message(chat_id, "📋 Your watchlist is empty\n\nUse: /watchlist add SYMBOL")
            return

        msg = "📋 *YOUR WATCHLIST*\n\n"
        for sym in user['watchlist']:
            info = get_stock_info(sym)
            if info and info.get('price'):
                price = info['price']
                prev = info.get('previous_close', price)
                pct = ((price - prev) / prev * 100) if prev else 0
                emoji = "📈" if pct >= 0 else "📉"
                msg += f"{emoji} *{sym}*: ₹{price:,.2f} ({pct:+.2f}%)\n"
            else:
                msg += f"⚪ *{sym}*: N/A\n"

        msg += "\n_/watchlist add SYMBOL_\n_/watchlist remove SYMBOL_"
        send_message(chat_id, msg)

def handle_alert(chat_id, symbol, target_price):
    """Set price alert"""
    user = get_user_data(chat_id)

    try:
        target = float(target_price)
        user['alerts'].append({
            "symbol": symbol.upper(),
            "target": target,
            "created": datetime.now().isoformat()
        })
        send_message(chat_id, f"✅ Alert set!\n\n*{symbol.upper()}* at ₹{target:,.2f}")
    except:
        send_message(chat_id, "❌ Invalid price. Use: /alert RELIANCE 2500")

def handle_alerts(chat_id):
    """Show alerts"""
    user = get_user_data(chat_id)

    if not user['alerts']:
        send_message(chat_id, "🔔 No alerts set\n\nUse: /alert SYMBOL PRICE")
        return

    msg = "🔔 *YOUR ALERTS*\n\n"
    for i, alert in enumerate(user['alerts'], 1):
        msg += f"{i}. *{alert['symbol']}* → ₹{alert['target']:,.2f}\n"

    send_message(chat_id, msg)

def handle_portfolio(chat_id, action=None, symbol=None, qty=None, price=None):
    """Manage portfolio"""
    user = get_user_data(chat_id)

    if action == "add" and symbol and qty and price:
        try:
            user['portfolio'][symbol.upper()] = {
                "qty": int(qty),
                "avg_price": float(price)
            }
            send_message(chat_id, f"✅ Added *{symbol.upper()}*\nQty: {qty}, Avg: ₹{float(price):,.2f}")
        except:
            send_message(chat_id, "❌ Invalid. Use: /portfolio add RELIANCE 10 2500")
    elif action == "remove" and symbol:
        if symbol.upper() in user['portfolio']:
            del user['portfolio'][symbol.upper()]
            send_message(chat_id, f"✅ Removed *{symbol.upper()}* from portfolio")
        else:
            send_message(chat_id, f"⚠️ *{symbol.upper()}* not in portfolio")
    else:
        # Show portfolio
        if not user['portfolio']:
            send_message(chat_id, "💼 Portfolio empty\n\nUse: /portfolio add SYMBOL QTY PRICE")
            return

        msg = "💼 *YOUR PORTFOLIO*\n\n"
        total_invested = 0
        total_current = 0

        for sym, holding in user['portfolio'].items():
            info = get_stock_info(sym)
            if info and info.get('price'):
                current = info['price']
                qty = holding['qty']
                avg = holding['avg_price']
                invested = qty * avg
                current_val = qty * current
                pnl = current_val - invested
                pnl_pct = (pnl / invested * 100) if invested else 0

                total_invested += invested
                total_current += current_val

                emoji = "📈" if pnl >= 0 else "📉"
                msg += f"*{sym}* ({qty} shares)\n"
                msg += f"  Avg: ₹{avg:,.2f} → ₹{current:,.2f}\n"
                msg += f"  {emoji} P/L: ₹{pnl:+,.2f} ({pnl_pct:+.2f}%)\n\n"

        total_pnl = total_current - total_invested
        total_pct = (total_pnl / total_invested * 100) if total_invested else 0
        emoji = "📈" if total_pnl >= 0 else "📉"

        msg += f"━━━━━━━━━━━━━━━\n"
        msg += f"💰 *Invested:* ₹{total_invested:,.2f}\n"
        msg += f"💎 *Current:* ₹{total_current:,.2f}\n"
        msg += f"{emoji} *Total P/L:* ₹{total_pnl:+,.2f} ({total_pct:+.2f}%)"

        send_message(chat_id, msg)

def handle_screen(chat_id, criteria):
    """Stock screener"""
    send_message(chat_id, f"🔍 Screening for *{criteria}* stocks...")

    results = screen_stocks(criteria)

    if not results:
        send_message(chat_id, f"❌ No {criteria} stocks found")
        return

    msg = f"🔍 *{criteria.upper()} STOCKS*\n\n"
    for stock in results[:10]:
        if 'rsi' in stock:
            msg += f"• *{stock['symbol']}*: ₹{stock['price']:,.2f} (RSI: {stock['rsi']:.1f})\n"
        else:
            msg += f"• *{stock['symbol']}*: ₹{stock['price']:,.2f}\n"

    send_message(chat_id, msg)

def handle_compare(chat_id, symbol1, symbol2):
    """Compare two stocks"""
    send_message(chat_id, f"🔄 Comparing *{symbol1}* vs *{symbol2}*...")

    info1 = get_stock_info(symbol1)
    info2 = get_stock_info(symbol2)

    df1 = get_stock_data(symbol1)
    df2 = get_stock_data(symbol2)

    a1 = analyze_stock(df1) if not df1.empty else None
    a2 = analyze_stock(df2) if not df2.empty else None

    if not info1.get('price') or not info2.get('price'):
        send_message(chat_id, "❌ Could not fetch data")
        return

    msg = f"""📊 *COMPARISON*

*{info1.get('name', symbol1)}*
💰 Price: ₹{info1.get('price', 0):,.2f}
📊 P/E: {info1.get('pe_ratio', 0):.2f}
📈 52W High: ₹{info1.get('52w_high', 0):,.2f}
📉 52W Low: ₹{info1.get('52w_low', 0):,.2f}
{a1['signal'] if a1 else 'N/A'}

*{info2.get('name', symbol2)}*
💰 Price: ₹{info2.get('price', 0):,.2f}
📊 P/E: {info2.get('pe_ratio', 0):.2f}
📈 52W High: ₹{info2.get('52w_high', 0):,.2f}
📉 52W Low: ₹{info2.get('52w_low', 0):,.2f}
{a2['signal'] if a2 else 'N/A'}"""

    send_message(chat_id, msg)

# ===== SCHEDULED TASKS =====
def check_alerts():
    """Check price alerts"""
    for chat_id, data in user_data.items():
        alerts_to_remove = []
        for i, alert in enumerate(data.get('alerts', [])):
            try:
                info = get_stock_info(alert['symbol'])
                if info and info.get('price'):
                    current = info['price']
                    target = alert['target']
                    if (target > 0 and current >= target) or (target < 0 and current <= abs(target)):
                        send_message(chat_id, f"🔔 *ALERT TRIGGERED!*\n\n*{alert['symbol']}* reached ₹{current:,.2f}\nTarget was: ₹{target:,.2f}")
                        alerts_to_remove.append(i)
            except:
                pass

        for i in reversed(alerts_to_remove):
            data['alerts'].pop(i)

def send_market_summary():
    """Send daily market summary at 8 AM"""
    now = datetime.now()
    if now.hour == 8 and now.minute < 5:
        handle_summary(ADMIN_CHAT_ID)

# ===== BOT LOOP =====
def run_bot():
    logger.info("🚀 MarketBuddy Pro starting...")

    # Try to send startup message, but don't crash if it fails
    try:
        send_message(ADMIN_CHAT_ID, "🚀 *MarketBuddy Pro Started!*\n\nAll features active.\nUse /start for help.")
        logger.info("Startup message sent successfully")
    except Exception as e:
        logger.error(f"Failed to send startup message: {e}")

    offset = None
    last_alert_check = time.time()
    last_summary = time.time()

    logger.info("Starting bot loop...")

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

                        logger.info(f"Message from {user_name}: {text}")
                        parts = text.split()
                        cmd = parts[0].lower()

                        # Handle commands
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

                        elif cmd == '/topgainers':
                            handle_topgainers(chat_id)

                        elif cmd == '/toplosers':
                            handle_toplosers(chat_id)

                        elif cmd == '/options':
                            symbol = parts[1] if len(parts) >= 2 else "NIFTY"
                            handle_options(chat_id, symbol)

                        elif cmd == '/fii':
                            handle_fii(chat_id)

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

                        elif cmd == '/screen' and len(parts) >= 2:
                            handle_screen(chat_id, parts[1])

                        elif cmd == '/compare' and len(parts) >= 3:
                            handle_compare(chat_id, parts[1], parts[2])

                        else:
                            send_message(chat_id, "❓ Unknown command. Use /help")

                    except Exception as cmd_error:
                        logger.error(f"Command error: {cmd_error}")
                        try:
                            send_message(chat_id, "❌ Error processing command. Try again.")
                        except:
                            pass

            # Check alerts every 5 minutes
            if time.time() - last_alert_check > 300:
                last_alert_check = time.time()
                check_alerts()

            # Daily summary at 8 AM
            now = datetime.now()
            if now.hour == 8 and now.minute < 2 and time.time() - last_summary > 3600:
                last_summary = time.time()
                handle_summary(ADMIN_CHAT_ID)

        except Exception as e:
            logger.error(f"Loop error: {e}")
            time.sleep(5)

# ===== MAIN =====
if __name__ == "__main__":
    bot_thread = threading.Thread(target=run_bot, daemon=True)
    bot_thread.start()

    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)
