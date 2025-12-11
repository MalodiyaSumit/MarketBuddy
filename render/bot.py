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

# ===== ANALYSIS =====
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

        latest = df.iloc[-1]
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
    except Exception as e:
        logger.error(f"Analysis error: {e}")
        return None

# ===== CHART =====
def generate_chart(symbol):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from ta.momentum import RSIIndicator

        df = get_stock_data(symbol)
        if df is None or len(df) < 20:
            return None

        fig, axes = plt.subplots(2, 1, figsize=(10, 8))

        ax1 = axes[0]
        ax1.set_title(f"{symbol.upper()} - Price Chart")
        ax1.plot(df.index, df['close'], label='Close', color='blue')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        ax2 = axes[1]
        rsi = RSIIndicator(close=df['close'], window=14).rsi()
        ax2.plot(df.index, rsi, color='purple')
        ax2.axhline(y=70, color='red', linestyle='--')
        ax2.axhline(y=30, color='green', linestyle='--')
        ax2.set_ylabel('RSI')
        ax2.set_ylim(0, 100)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100)
        buf.seek(0)
        plt.close()
        return buf
    except Exception as e:
        logger.error(f"Chart error: {e}")
        return None

# ===== HANDLERS =====
def handle_start(chat_id, user_name):
    msg = f"""*Welcome to MarketBuddy Pro!*

Hello {user_name}!

*COMMANDS:*
/stock SYMBOL - Technical analysis
/price SYMBOL - Quick price
/chart SYMBOL - Price chart
/summary - Market indices
/news - Latest news
/watchlist - Your watchlist
/watchlist add SYMBOL
/watchlist remove SYMBOL
/portfolio - Your holdings
/portfolio add SYMBOL QTY PRICE
/alert SYMBOL PRICE - Set alert
/alerts - View alerts

Try: /stock RELIANCE"""
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

    msg = f"""*{info.get('name', symbol)}*

*Price:* Rs {price:,.2f}
*Change:* {change:+,.2f} ({pct:+.2f}%)

*SIGNAL: {analysis['signal']}*

*Indicators:*
"""
    for ind, val in analysis['indicators'].items():
        msg += f"- {ind}: {val}\n"

    msg += f"""
*More Info:*
- 52W High: Rs {info.get('52w_high', 0):,.2f}
- 52W Low: Rs {info.get('52w_low', 0):,.2f}
- P/E: {info.get('pe_ratio', 0):.2f}
- Sector: {info.get('sector', 'N/A')}"""

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

# ===== BOT LOOP =====
def run_bot():
    global bot_started
    if bot_started:
        logger.info("Bot already running, skipping...")
        return

    bot_started = True
    logger.info("MarketBuddy Pro starting...")

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
