#!/usr/bin/env python3
"""
MarketBuddy Telegram Bot
Stock analysis bot with TA, anomaly detection, and forecasting
"""

import os
import sys
import socket
import logging
from datetime import datetime
from dotenv import load_dotenv

# Force IPv4 to fix Telegram connection issues
original_getaddrinfo = socket.getaddrinfo
def forced_ipv4_getaddrinfo(host, port, family=0, type=0, proto=0, flags=0):
    return original_getaddrinfo(host, port, socket.AF_INET, type, proto, flags)
socket.getaddrinfo = forced_ipv4_getaddrinfo

from telegram import Update, ParseMode
from telegram.ext import (
    Updater,
    CommandHandler,
    CallbackContext,
    MessageHandler,
    Filters,
)

from data_fetcher import get_stock_data, get_stock_info, get_index_data, get_market_news
from analyzer import full_analysis, calculate_technical_indicators, generate_signals
from scheduler_runner import market_scheduler

# Load environment variables
load_dotenv()

# Configure logging
LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs')
os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, 'bot.log')),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Bot Token
BOT_TOKEN = os.getenv("BOT_TOKEN")
ADMIN_CHAT_ID = os.getenv("ADMIN_CHAT_ID")

if not BOT_TOKEN:
    logger.error("BOT_TOKEN not found in environment variables!")
    sys.exit(1)


def start(update: Update, context: CallbackContext) -> None:
    """Handle /start command"""
    user = update.effective_user
    welcome_msg = f"""
*Welcome to MarketBuddy!*

Hello {user.first_name}! I'm your personal stock analysis assistant.

*Available Commands:*

/stock SYMBOL - Get detailed analysis
  Example: `/stock RELIANCE` or `/stock TCS`

/summary - Get NIFTY & SENSEX summary

/news - Get latest market news

/help - Show this help message

*Features:*
 Technical Analysis (RSI, MACD, SMA, Bollinger)
 Anomaly Detection
 7-Day Price Forecast
 Automatic Buy Alerts (every 5 min)
 Daily Summary (8 AM)

_Start by typing `/stock RELIANCE` to see it in action!_
"""
    update.message.reply_text(welcome_msg, parse_mode=ParseMode.MARKDOWN)
    logger.info(f"User {user.id} ({user.first_name}) started the bot")


def help_command(update: Update, context: CallbackContext) -> None:
    """Handle /help command"""
    start(update, context)


def stock_command(update: Update, context: CallbackContext) -> None:
    """Handle /stock SYMBOL command"""
    if not context.args:
        update.message.reply_text(
            "Please provide a stock symbol.\n"
            "Example: `/stock RELIANCE` or `/stock TCS`",
            parse_mode=ParseMode.MARKDOWN
        )
        return

    symbol = context.args[0].upper()
    update.message.reply_text(f"Analyzing *{symbol}*... Please wait.", parse_mode=ParseMode.MARKDOWN)

    try:
        # Get stock data
        df = get_stock_data(symbol, period="6mo")

        if df.empty:
            update.message.reply_text(
                f"Could not fetch data for *{symbol}*.\n"
                "Please check the symbol and try again.\n\n"
                "For Indian stocks, use symbols like:\n"
                "RELIANCE, TCS, INFY, HDFCBANK, etc.",
                parse_mode=ParseMode.MARKDOWN
            )
            return

        # Get stock info
        info = get_stock_info(symbol)

        # Perform full analysis
        analysis = full_analysis(df, include_forecast=True)

        if 'error' in analysis:
            update.message.reply_text(f"Error analyzing {symbol}: {analysis['error']}")
            return

        # Format the response
        msg = format_stock_analysis(symbol, info, analysis)
        update.message.reply_text(msg, parse_mode=ParseMode.MARKDOWN)

        logger.info(f"Stock analysis sent for {symbol}")

    except Exception as e:
        logger.error(f"Error in stock_command for {symbol}: {e}")
        update.message.reply_text(f"Error analyzing {symbol}. Please try again.")


def format_stock_analysis(symbol: str, info: dict, analysis: dict) -> str:
    """Format stock analysis as Telegram message"""
    name = info.get('name', symbol)
    current_price = analysis.get('current_price', 0)
    signals = analysis.get('signals', {})
    indicators = analysis.get('indicators', {})
    forecast = analysis.get('forecast', {})
    anomaly = analysis.get('anomaly', {})

    # Signal emoji
    overall = signals.get('overall', 'NEUTRAL')
    if 'BUY' in overall:
        signal_emoji = ""
    elif 'SELL' in overall:
        signal_emoji = ""
    else:
        signal_emoji = ""

    msg = f"*{name}* ({symbol})\n"
    msg += f"{'='*30}\n\n"

    # Price Info
    msg += f"*Current Price:* {current_price:,.2f}\n"
    if info.get('previous_close'):
        change = current_price - info['previous_close']
        change_pct = (change / info['previous_close']) * 100
        change_emoji = "" if change >= 0 else ""
        msg += f"*Change:* {change_emoji} {change:,.2f} ({change_pct:.2f}%)\n"

    msg += f"*Open:* {analysis.get('open', 0):,.2f}\n"
    msg += f"*High:* {analysis.get('high', 0):,.2f}\n"
    msg += f"*Low:* {analysis.get('low', 0):,.2f}\n"
    msg += f"*Volume:* {analysis.get('volume', 0):,.0f}\n\n"

    # Overall Signal
    msg += f"*SIGNAL: {signal_emoji} {overall}*\n"
    msg += f"Signal Strength: {signals.get('strength', 0)}\n\n"

    # Technical Indicators
    msg += "*TECHNICAL INDICATORS*\n"

    rsi = indicators.get('rsi')
    if rsi:
        rsi_status = "Oversold" if rsi < 30 else ("Overbought" if rsi > 70 else "Neutral")
        msg += f"  RSI(14): {rsi:.1f} ({rsi_status})\n"

    macd = indicators.get('macd')
    macd_signal = indicators.get('macd_signal')
    if macd is not None and macd_signal is not None:
        macd_status = "Bullish" if macd > macd_signal else "Bearish"
        msg += f"  MACD: {macd:.2f} ({macd_status})\n"

    sma_20 = indicators.get('sma_20')
    sma_50 = indicators.get('sma_50')
    if sma_20 and sma_50:
        msg += f"  SMA 20: {sma_20:,.2f}\n"
        msg += f"  SMA 50: {sma_50:,.2f}\n"
        sma_status = "Bullish" if sma_20 > sma_50 else "Bearish"
        msg += f"  Trend: {sma_status}\n"

    bb_upper = indicators.get('bb_upper')
    bb_lower = indicators.get('bb_lower')
    if bb_upper and bb_lower:
        msg += f"  BB Upper: {bb_upper:,.2f}\n"
        msg += f"  BB Lower: {bb_lower:,.2f}\n"

    msg += "\n"

    # Indicator Signals
    indicator_signals = signals.get('indicators', {})
    if indicator_signals:
        msg += "*INDICATOR SIGNALS*\n"
        for ind, sig in indicator_signals.items():
            msg += f"  {ind}: {sig}\n"
        msg += "\n"

    # Buy/Sell Signals
    buy_signals = signals.get('buy_signals', [])
    sell_signals = signals.get('sell_signals', [])

    if buy_signals:
        msg += "*BUY SIGNALS*\n"
        for s in buy_signals[:3]:
            msg += f"   {s}\n"
        msg += "\n"

    if sell_signals:
        msg += "*SELL SIGNALS*\n"
        for s in sell_signals[:3]:
            msg += f"   {s}\n"
        msg += "\n"

    # Anomaly Detection
    if anomaly.get('is_anomaly'):
        msg += "*ANOMALY DETECTED*\n"
        msg += f"  Score: {anomaly.get('score', 0):.2f}\n"
        msg += "  (Unusual price/volume activity)\n\n"

    # Forecast
    if forecast and 'error' not in forecast:
        msg += "*7-DAY FORECAST*\n"
        trend = forecast.get('trend', 'N/A')
        trend_emoji = "" if trend == "BULLISH" else ("" if trend == "BEARISH" else "")
        msg += f"  Trend: {trend_emoji} {trend}\n"
        msg += f"  Current: {forecast.get('current_price', 0):,.2f}\n"
        msg += f"  Forecast: {forecast.get('forecast_7d', 0):,.2f}\n"
        msg += f"  Expected Change: {forecast.get('expected_change_pct', 0):.2f}%\n"
        msg += f"  Range: {forecast.get('lower_bound', 0):,.2f} - {forecast.get('upper_bound', 0):,.2f}\n\n"

    # Additional Info
    if info.get('sector'):
        msg += f"*Sector:* {info.get('sector')}\n"
    if info.get('pe_ratio') and info.get('pe_ratio') > 0:
        msg += f"*P/E Ratio:* {info.get('pe_ratio'):.2f}\n"
    if info.get('52w_high'):
        msg += f"*52W High:* {info.get('52w_high'):,.2f}\n"
    if info.get('52w_low'):
        msg += f"*52W Low:* {info.get('52w_low'):,.2f}\n"

    msg += "\n_This is not financial advice. Do your own research._"

    return msg


def summary_command(update: Update, context: CallbackContext) -> None:
    """Handle /summary command"""
    update.message.reply_text("Fetching market summary... Please wait.")

    try:
        msg = "*MARKET SUMMARY*\n"
        msg += f"_{datetime.now().strftime('%Y-%m-%d %H:%M')}_\n\n"

        # NIFTY
        nifty = get_index_data("NIFTY")
        if nifty:
            change_emoji = "" if nifty.get('change', 0) >= 0 else ""
            msg += f"*NIFTY 50*\n"
            msg += f"  Value: {nifty.get('value', 0):,.2f}\n"
            msg += f"  Change: {change_emoji} {nifty.get('change', 0):,.2f} ({nifty.get('change_pct', 0):.2f}%)\n"
            msg += f"  Open: {nifty.get('open', 0):,.2f}\n"
            msg += f"  High: {nifty.get('high', 0):,.2f}\n"
            msg += f"  Low: {nifty.get('low', 0):,.2f}\n\n"

        # SENSEX
        sensex = get_index_data("SENSEX")
        if sensex:
            change_emoji = "" if sensex.get('change', 0) >= 0 else ""
            msg += f"*SENSEX*\n"
            msg += f"  Value: {sensex.get('value', 0):,.2f}\n"
            msg += f"  Change: {change_emoji} {sensex.get('change', 0):,.2f} ({sensex.get('change_pct', 0):.2f}%)\n"
            msg += f"  Open: {sensex.get('open', 0):,.2f}\n"
            msg += f"  High: {sensex.get('high', 0):,.2f}\n"
            msg += f"  Low: {sensex.get('low', 0):,.2f}\n\n"

        # BANK NIFTY
        banknifty = get_index_data("BANKNIFTY")
        if banknifty:
            change_emoji = "" if banknifty.get('change', 0) >= 0 else ""
            msg += f"*BANK NIFTY*\n"
            msg += f"  Value: {banknifty.get('value', 0):,.2f}\n"
            msg += f"  Change: {change_emoji} {banknifty.get('change', 0):,.2f} ({banknifty.get('change_pct', 0):.2f}%)\n\n"

        if not nifty and not sensex:
            msg += "Unable to fetch market data. Markets may be closed.\n"

        msg += "_Updated just now_"
        update.message.reply_text(msg, parse_mode=ParseMode.MARKDOWN)

        logger.info("Market summary sent")

    except Exception as e:
        logger.error(f"Error in summary_command: {e}")
        update.message.reply_text("Error fetching market summary. Please try again.")


def news_command(update: Update, context: CallbackContext) -> None:
    """Handle /news command"""
    update.message.reply_text("Fetching latest market news... Please wait.")

    try:
        news = get_market_news(limit=5)

        if not news:
            update.message.reply_text(
                "No news available at the moment.\n"
                "This might be due to API limits or no recent news."
            )
            return

        msg = "*LATEST MARKET NEWS*\n"
        msg += f"_{datetime.now().strftime('%Y-%m-%d %H:%M')}_\n\n"

        for i, article in enumerate(news, 1):
            title = article.get('title', 'No title')
            source = article.get('source', 'Unknown')
            url = article.get('url', '')

            # Escape markdown special characters in title
            title = title.replace('*', '').replace('_', '').replace('[', '').replace(']', '')

            msg += f"*{i}. {title[:100]}*\n"
            msg += f"   Source: {source}\n"
            if url:
                msg += f"   [Read more]({url})\n"
            msg += "\n"

        update.message.reply_text(msg, parse_mode=ParseMode.MARKDOWN, disable_web_page_preview=True)

        logger.info("News sent")

    except Exception as e:
        logger.error(f"Error in news_command: {e}")
        update.message.reply_text("Error fetching news. Please try again.")


def unknown_command(update: Update, context: CallbackContext) -> None:
    """Handle unknown commands"""
    update.message.reply_text(
        "Unknown command. Use /help to see available commands."
    )


def error_handler(update: Update, context: CallbackContext) -> None:
    """Handle errors"""
    logger.error(f"Update {update} caused error {context.error}")


def main():
    """Main function to run the bot"""
    logger.info("Starting MarketBuddy Bot...")

    # Create updater and dispatcher
    updater = Updater(token=BOT_TOKEN, use_context=True)
    dispatcher = updater.dispatcher

    # Add command handlers
    dispatcher.add_handler(CommandHandler("start", start))
    dispatcher.add_handler(CommandHandler("help", help_command))
    dispatcher.add_handler(CommandHandler("stock", stock_command))
    dispatcher.add_handler(CommandHandler("summary", summary_command))
    dispatcher.add_handler(CommandHandler("news", news_command))

    # Handle unknown commands
    dispatcher.add_handler(MessageHandler(Filters.command, unknown_command))

    # Add error handler
    dispatcher.add_error_handler(error_handler)

    # Set bot for scheduler and start it
    market_scheduler.set_bot(updater.bot)
    market_scheduler.start()

    # Send startup notification
    if ADMIN_CHAT_ID:
        try:
            updater.bot.send_message(
                chat_id=ADMIN_CHAT_ID,
                text="*MarketBuddy Bot Started!*\n\n"
                     "Bot is now running and monitoring markets.\n"
                     " Buy alerts: Every 5 minutes\n"
                     " Daily summary: 8:00 AM\n\n"
                     "Use /help to see all commands.",
                parse_mode=ParseMode.MARKDOWN
            )
        except Exception as e:
            logger.error(f"Could not send startup message: {e}")

    logger.info("Bot started successfully!")

    # Start polling
    updater.start_polling(drop_pending_updates=True)
    updater.idle()

    # Stop scheduler on exit
    market_scheduler.stop()
    logger.info("Bot stopped.")


if __name__ == "__main__":
    main()
