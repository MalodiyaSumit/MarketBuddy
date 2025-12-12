#!/usr/bin/env python3
"""
MarketBuddy Pro - Advanced Stock Analysis Telegram Bot
"""

import os
import requests
import logging
import time
import io
from datetime import datetime
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

# Indian Indices
INDIAN_INDICES = {
    "NIFTY": "^NSEI", "NIFTY50": "^NSEI",
    "SENSEX": "^BSESN",
    "BANKNIFTY": "^NSEBANK",
}

# Default Watchlist
DEFAULT_WATCHLIST = ["RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK"]

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
            "bb_lower": bb_low
        }
    except Exception as e:
        logger.error(f"Analysis error: {e}")
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

*Available Commands:*

/stock SYMBOL - Detailed analysis with confidence score
  Example: /stock RELIANCE or /stock TCS

/price SYMBOL - Quick price check

/chart SYMBOL - Enhanced technical chart
  (Price, Volume, RSI, MACD, Bollinger Bands)

/summary - Get NIFTY, SENSEX & BANKNIFTY

/news - Latest market news

/watchlist - View your watchlist
/watchlist add SYMBOL - Add stock
/watchlist remove SYMBOL - Remove stock

/portfolio - View your holdings
/portfolio add SYMBOL QTY PRICE

/alert SYMBOL PRICE - Set price alert
/alerts - View your alerts

/help - Show this help message

*New Features:*
  Advanced Confidence Score (0-100)
  6-Factor Analysis (RSI, MACD, SMA, BB, Volume, Momentum)
  High-Confidence Alerts (score >= 70)
  Enhanced Multi-Panel Charts

*Automatic Alerts:*
  Market Open/Close (9:15 AM / 3:30 PM)
  Morning & Afternoon Updates (10:30 AM / 1:30 PM)
  High Confidence Opportunities (every 15 min)
  Breaking News Alerts (every 10 min)
  Weekly Report (Saturday 6 PM)

Start with `/stock RELIANCE` to see the analysis!"""
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

_Not financial advice_"""

    send_message(chat_id, msg)

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

    msg += "\n*WATCHLIST PERFORMANCE:*\n"

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

    msg += "\nSee you tomorrow! 👋"
    send_message(ADMIN_CHAT_ID, msg)
    logger.info("Market close alert sent")


def check_breaking_news():
    """Check for important breaking news and send alerts"""
    global sent_news_ids

    try:
        # Important keywords for Indian market
        keywords = ["RBI", "SEBI", "Sensex", "Nifty", "crash", "surge", "breaking",
                   "Fed", "interest rate", "inflation", "GDP", "recession",
                   "Reliance", "TCS", "HDFC", "Infosys", "budget", "tax"]

        articles = get_market_news(limit=10)

        for article in articles:
            title = article.get('title', '')
            description = article.get('description', '')
            url = article.get('url', '')

            # Create unique ID
            news_id = hash(title)

            # Skip if already sent
            if news_id in sent_news_ids:
                continue

            # Check if important
            content = (title + " " + description).lower()
            is_important = any(kw.lower() in content for kw in keywords)

            if is_important:
                msg = f"""🚨 *BREAKING NEWS ALERT*

*{title}*

{description[:200] if description else ''}...

_Source: {article.get('source', {}).get('name', 'Unknown')}_"""

                send_message(ADMIN_CHAT_ID, msg)
                sent_news_ids.add(news_id)
                logger.info(f"Breaking news sent: {title[:50]}")

                # Keep only last 100 news IDs
                if len(sent_news_ids) > 100:
                    sent_news_ids = set(list(sent_news_ids)[-50:])

                # Only send one news at a time
                break

    except Exception as e:
        logger.error(f"Breaking news check error: {e}")


def check_strong_signals():
    """Check watchlist for strong buy/sell signals based on confidence score"""
    try:
        strong_alerts = []

        for sym in DEFAULT_WATCHLIST:
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

            msg = f"""🎯 *HIGH CONFIDENCE OPPORTUNITY ALERT*
_{now.strftime('%Y-%m-%d %H:%M')} IST_

Found {len(high_conf_alerts)} stock(s) with confidence score >= 70!

"""
            for alert in high_conf_alerts[:5]:  # Max 5 alerts
                emoji = "🟢🟢🟢" if alert['confidence'] >= 85 else "🟢🟢"
                rsi_val = alert.get('rsi')
                rsi_str = f"{rsi_val:.1f}" if rsi_val and rsi_val > 0 else "N/A"
                msg += f"""*{alert['symbol']}* {emoji}
  Signal: {alert['signal']}
  Confidence: *{alert['confidence']}/100*
  Price: ₹{alert['price']:,.2f} ({alert['change_pct']:+.2f}%)
  RSI: {rsi_str}
  Top Factors:
"""
                for factor, detail in list(alert['score_factors'].items())[:2]:
                    msg += f"    • {factor}: {detail}\n"
                msg += "\n"

            msg += "_Do your own research before investing._"
            send_message(ADMIN_CHAT_ID, msg)
            logger.info(f"High confidence alerts sent: {len(high_conf_alerts)} stocks")

    except Exception as e:
        logger.error(f"High confidence check error: {e}")


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

        for sym in DEFAULT_WATCHLIST:
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
