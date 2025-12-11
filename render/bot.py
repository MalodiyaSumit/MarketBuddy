#!/usr/bin/env python3
"""
MarketBuddy Telegram Bot - Render.com Version
"""

import os
import requests
import logging
import time
from datetime import datetime
from flask import Flask
import threading

import yfinance as yf
import pandas as pd
from ta.momentum import RSIIndicator
from ta.trend import MACD, SMAIndicator
from ta.volatility import BollingerBands

# Flask app to keep alive
app = Flask(__name__)

@app.route('/')
def home():
    return "MarketBuddy Bot is running!"

@app.route('/health')
def health():
    return "OK"

# Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ===== CONFIG =====
BOT_TOKEN = os.environ.get("BOT_TOKEN", "8241472299:AAGhewPu-VZFXpuLyVBlhRZYrGPgxpCb7mI")
ADMIN_CHAT_ID = os.environ.get("ADMIN_CHAT_ID", "2110121880")
API_URL = f"https://api.telegram.org/bot{BOT_TOKEN}"

INDIAN_INDICES = {"NIFTY": "^NSEI", "SENSEX": "^BSESN", "BANKNIFTY": "^NSEBANK"}
WATCHLIST = ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS"]


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


def get_updates(offset=None):
    try:
        url = f"{API_URL}/getUpdates"
        params = {"timeout": 30, "offset": offset}
        response = requests.get(url, params=params, timeout=35)
        return response.json()
    except Exception as e:
        logger.error(f"Update error: {e}")
        return None


# ===== DATA FETCHER =====
def get_stock_data(symbol, period="3mo"):
    try:
        if symbol.upper() in INDIAN_INDICES:
            symbol = INDIAN_INDICES[symbol.upper()]
        elif '.' not in symbol and '^' not in symbol:
            symbol = f"{symbol}.NS"

        ticker = yf.Ticker(symbol)
        df = ticker.history(period=period)
        if df.empty:
            return pd.DataFrame()
        df.columns = [c.lower() for c in df.columns]
        df = df.reset_index()
        return df
    except Exception as e:
        logger.error(f"Fetch error {symbol}: {e}")
        return pd.DataFrame()


def get_stock_info(symbol):
    try:
        if symbol.upper() in INDIAN_INDICES:
            symbol = INDIAN_INDICES[symbol.upper()]
        elif '.' not in symbol and '^' not in symbol:
            symbol = f"{symbol}.NS"

        ticker = yf.Ticker(symbol)
        info = ticker.info
        return {
            "name": info.get("longName", symbol),
            "previous_close": info.get("previousClose", 0),
        }
    except:
        return {}


def get_index_data(index_name):
    symbol = INDIAN_INDICES.get(index_name.upper(), "^NSEI")
    try:
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period="2d")
        if len(hist) >= 2:
            prev = hist['Close'].iloc[-2]
            curr = hist['Close'].iloc[-1]
            change = curr - prev
            pct = (change / prev) * 100
            return {"name": index_name, "value": curr, "change": change, "pct": pct}
    except:
        pass
    return {}


# ===== ANALYZER =====
def analyze_stock(df):
    if df.empty or len(df) < 20:
        return None

    df = df.copy()
    df.columns = [c.lower() for c in df.columns]

    try:
        df['rsi'] = RSIIndicator(close=df['close'], window=14).rsi()
        macd = MACD(close=df['close'])
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        df['sma_20'] = SMAIndicator(close=df['close'], window=20).sma_indicator()
        df['sma_50'] = SMAIndicator(close=df['close'], window=50).sma_indicator()
        bb = BollingerBands(close=df['close'])
        df['bb_upper'] = bb.bollinger_hband()
        df['bb_lower'] = bb.bollinger_lband()
    except:
        pass

    latest = df.iloc[-1]
    indicators = {}
    buy = sell = 0

    rsi = latest.get('rsi')
    if rsi and not pd.isna(rsi):
        if rsi < 30:
            indicators['RSI'] = f"OVERSOLD ({rsi:.1f})"
            buy += 2
        elif rsi > 70:
            indicators['RSI'] = f"OVERBOUGHT ({rsi:.1f})"
            sell += 2
        else:
            indicators['RSI'] = f"NEUTRAL ({rsi:.1f})"

    macd_val = latest.get('macd')
    macd_sig = latest.get('macd_signal')
    if macd_val and macd_sig and not (pd.isna(macd_val) or pd.isna(macd_sig)):
        if macd_val > macd_sig:
            indicators['MACD'] = "BULLISH"
            buy += 1
        else:
            indicators['MACD'] = "BEARISH"
            sell += 1

    sma20 = latest.get('sma_20')
    sma50 = latest.get('sma_50')
    if sma20 and sma50 and not (pd.isna(sma20) or pd.isna(sma50)):
        if sma20 > sma50:
            indicators['SMA'] = "BULLISH"
            buy += 1
        else:
            indicators['SMA'] = "BEARISH"
            sell += 1

    close = latest.get('close')
    bb_low = latest.get('bb_lower')
    bb_up = latest.get('bb_upper')
    if close and bb_low and bb_up and not (pd.isna(bb_low) or pd.isna(bb_up)):
        if close <= bb_low:
            indicators['BB'] = "LOWER BAND"
            buy += 1
        elif close >= bb_up:
            indicators['BB'] = "UPPER BAND"
            sell += 1
        else:
            indicators['BB'] = "WITHIN BANDS"

    net = buy - sell
    if net >= 3:
        signal = "STRONG BUY"
    elif net >= 1:
        signal = "BUY"
    elif net <= -3:
        signal = "STRONG SELL"
    elif net <= -1:
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
        "indicators": indicators
    }


# ===== HANDLERS =====
def handle_start(chat_id, user_name):
    msg = f"""*Welcome to MarketBuddy!*

Hello {user_name}!

*Commands:*
/stock SYMBOL - Analysis
/summary - Market indices
/help - Help

_Try /stock RELIANCE_"""
    send_message(chat_id, msg)


def handle_stock(chat_id, symbol):
    send_message(chat_id, f"Analyzing *{symbol}*...")

    df = get_stock_data(symbol)
    if df.empty:
        send_message(chat_id, f"No data for {symbol}")
        return

    info = get_stock_info(symbol)
    analysis = analyze_stock(df)

    if not analysis:
        send_message(chat_id, "Insufficient data")
        return

    signal = analysis['signal']
    if 'BUY' in signal:
        emoji = "🟢"
    elif 'SELL' in signal:
        emoji = "🔴"
    else:
        emoji = "🟡"

    msg = f"*{info.get('name', symbol)}*\n{'='*25}\n\n"
    msg += f"*Price:* {analysis['price']:,.2f}\n"

    prev = info.get('previous_close')
    if prev:
        chg = analysis['price'] - prev
        pct = (chg / prev) * 100
        e = "📈" if chg >= 0 else "📉"
        msg += f"*Change:* {e} {chg:,.2f} ({pct:.2f}%)\n"

    msg += f"\n*SIGNAL: {emoji} {signal}*\n\n"

    msg += "*Indicators:*\n"
    for k, v in analysis['indicators'].items():
        msg += f"  {k}: {v}\n"

    if analysis.get('rsi'):
        msg += f"\nRSI: {analysis['rsi']:.1f}\n"
    if analysis.get('sma_20') and analysis.get('sma_50'):
        msg += f"SMA20: {analysis['sma_20']:,.2f}\n"
        msg += f"SMA50: {analysis['sma_50']:,.2f}\n"

    msg += "\n_Not financial advice_"
    send_message(chat_id, msg)


def handle_summary(chat_id):
    send_message(chat_id, "Fetching...")

    msg = f"*MARKET SUMMARY*\n_{datetime.now().strftime('%Y-%m-%d %H:%M')}_\n\n"

    for idx in ["NIFTY", "SENSEX", "BANKNIFTY"]:
        data = get_index_data(idx)
        if data:
            e = "📈" if data.get('change', 0) >= 0 else "📉"
            msg += f"*{idx}*\n"
            msg += f"  {data.get('value', 0):,.2f}\n"
            msg += f"  {e} {data.get('change', 0):,.2f} ({data.get('pct', 0):.2f}%)\n\n"

    send_message(chat_id, msg)


# ===== BOT LOOP =====
def run_bot():
    logger.info("Bot starting...")
    send_message(ADMIN_CHAT_ID, "*MarketBuddy Started!*\n\nBot running 24/7.\nUse /help")

    offset = None
    last_check = time.time()

    while True:
        try:
            updates = get_updates(offset)

            if updates and updates.get('ok'):
                for update in updates.get('result', []):
                    offset = update['update_id'] + 1

                    message = update.get('message', {})
                    text = message.get('text', '')
                    chat_id = message.get('chat', {}).get('id')
                    user = message.get('from', {})
                    user_name = user.get('first_name', 'User')

                    if not text or not chat_id:
                        continue

                    logger.info(f"Message from {user_name}: {text}")

                    if text.startswith('/start') or text.startswith('/help'):
                        handle_start(chat_id, user_name)
                    elif text.startswith('/stock'):
                        parts = text.split()
                        if len(parts) >= 2:
                            handle_stock(chat_id, parts[1].upper())
                        else:
                            send_message(chat_id, "Usage: `/stock RELIANCE`")
                    elif text.startswith('/summary'):
                        handle_summary(chat_id)
                    else:
                        send_message(chat_id, "Unknown command. Try /help")

            # Check opportunities every 5 minutes
            if time.time() - last_check > 300:
                last_check = time.time()
                logger.info("Checking opportunities...")

                opps = []
                for sym in WATCHLIST:
                    df = get_stock_data(sym)
                    if not df.empty:
                        a = analyze_stock(df)
                        if a and a.get('strength', 0) >= 2:
                            opps.append(f"*{sym.replace('.NS', '')}*: {a['price']:,.2f} - {a['signal']}")

                if opps:
                    msg = "*BUY ALERTS*\n\n" + "\n".join(opps[:3]) + "\n\n_Not financial advice_"
                    send_message(ADMIN_CHAT_ID, msg)

                logger.info(f"Found {len(opps)} opportunities")

        except Exception as e:
            logger.error(f"Loop error: {e}")
            time.sleep(5)


# ===== MAIN =====
if __name__ == "__main__":
    # Start bot in background thread
    bot_thread = threading.Thread(target=run_bot, daemon=True)
    bot_thread.start()

    # Run Flask server
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)
