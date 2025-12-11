"""
Scheduler Runner Module
Handles scheduled tasks for market alerts and daily summaries
Uses threading-based scheduler for compatibility
"""

import os
import logging
import threading
import time
from datetime import datetime
from dotenv import load_dotenv

from data_fetcher import get_bulk_stock_data, get_index_data, WATCHLIST
from analyzer import find_buy_opportunities, full_analysis

load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

ADMIN_CHAT_ID = os.getenv("ADMIN_CHAT_ID")


class MarketScheduler:
    """Scheduler for market alerts and summaries using threading"""

    def __init__(self, bot=None):
        """
        Initialize scheduler

        Args:
            bot: Telegram bot instance for sending messages
        """
        self.bot = bot
        self.last_opportunities = []
        self.running = False
        self.buy_check_thread = None
        self.daily_summary_thread = None

    def set_bot(self, bot):
        """Set bot instance after initialization"""
        self.bot = bot

    def send_message(self, chat_id: str, text: str):
        """Send message via bot"""
        if self.bot and chat_id:
            try:
                self.bot.send_message(chat_id=chat_id, text=text, parse_mode='Markdown')
                logger.info(f"Message sent to {chat_id}")
            except Exception as e:
                logger.error(f"Failed to send message: {e}")
        else:
            logger.warning("Bot not set or chat_id missing")

    def check_buy_opportunities(self):
        """Check for buy opportunities and send alerts"""
        logger.info("Checking buy opportunities...")

        try:
            # Get data for watchlist
            watchlist_data = get_bulk_stock_data(WATCHLIST[:10], period="3mo")

            if not watchlist_data:
                logger.warning("No data fetched for watchlist")
                return

            # Find opportunities
            opportunities = find_buy_opportunities(watchlist_data, min_strength=2)

            if opportunities:
                # Filter new opportunities
                current_symbols = [o['symbol'] for o in opportunities]
                last_symbols = [o['symbol'] for o in self.last_opportunities]
                new_opps = [o for o in opportunities if o['symbol'] not in last_symbols]

                if new_opps:
                    message = self._format_opportunities_message(new_opps)
                    self.send_message(ADMIN_CHAT_ID, message)

                self.last_opportunities = opportunities

            logger.info(f"Found {len(opportunities)} opportunities")

        except Exception as e:
            logger.error(f"Error checking opportunities: {e}")

    def send_daily_summary(self):
        """Send daily market summary"""
        logger.info("Sending daily summary...")

        try:
            message = self._generate_daily_summary()
            self.send_message(ADMIN_CHAT_ID, message)
            logger.info("Daily summary sent")

        except Exception as e:
            logger.error(f"Error sending daily summary: {e}")

    def _format_opportunities_message(self, opportunities: list) -> str:
        """Format buy opportunities as message"""
        msg = "*BUY ALERT*\n"
        msg += f"_{datetime.now().strftime('%Y-%m-%d %H:%M')}_\n\n"

        for opp in opportunities[:5]:  # Top 5 opportunities
            symbol = opp['symbol'].replace('.NS', '')
            price = opp.get('current_price', 0)
            signal = opp.get('signal', 'N/A')
            strength = opp.get('strength', 0)
            rsi = opp.get('rsi')

            msg += f"*{symbol}*\n"
            msg += f"  Price: {price:.2f}\n"
            msg += f"  Signal: {signal} (Strength: {strength})\n"

            if rsi:
                msg += f"  RSI: {rsi:.1f}\n"

            reasons = opp.get('buy_signals', [])
            if reasons:
                msg += f"  Reasons: {', '.join(reasons[:2])}\n"

            msg += "\n"

        msg += "_This is not financial advice_"
        return msg

    def _generate_daily_summary(self) -> str:
        """Generate daily market summary"""
        msg = "*DAILY MARKET SUMMARY*\n"
        msg += f"_{datetime.now().strftime('%Y-%m-%d %H:%M')}_\n\n"

        # Get index data
        nifty = get_index_data("NIFTY")
        sensex = get_index_data("SENSEX")

        if nifty:
            change_emoji = "" if nifty.get('change', 0) >= 0 else ""
            msg += f"*NIFTY 50*\n"
            msg += f"  Value: {nifty.get('value', 0):,.2f}\n"
            msg += f"  Change: {change_emoji} {nifty.get('change', 0):,.2f} ({nifty.get('change_pct', 0):.2f}%)\n"
            msg += f"  Open: {nifty.get('open', 0):,.2f}\n"
            msg += f"  High: {nifty.get('high', 0):,.2f}\n"
            msg += f"  Low: {nifty.get('low', 0):,.2f}\n\n"

        if sensex:
            change_emoji = "" if sensex.get('change', 0) >= 0 else ""
            msg += f"*SENSEX*\n"
            msg += f"  Value: {sensex.get('value', 0):,.2f}\n"
            msg += f"  Change: {change_emoji} {sensex.get('change', 0):,.2f} ({sensex.get('change_pct', 0):.2f}%)\n"
            msg += f"  Open: {sensex.get('open', 0):,.2f}\n"
            msg += f"  High: {sensex.get('high', 0):,.2f}\n"
            msg += f"  Low: {sensex.get('low', 0):,.2f}\n\n"

        # Get top movers
        try:
            watchlist_data = get_bulk_stock_data(WATCHLIST[:10], period="5d")
            movers = []

            for symbol, df in watchlist_data.items():
                if len(df) >= 2:
                    prev_close = df['close'].iloc[-2]
                    current = df['close'].iloc[-1]
                    change_pct = ((current - prev_close) / prev_close) * 100
                    movers.append({
                        'symbol': symbol.replace('.NS', ''),
                        'price': current,
                        'change_pct': change_pct
                    })

            # Sort by absolute change
            movers.sort(key=lambda x: abs(x['change_pct']), reverse=True)

            if movers:
                msg += "*TOP MOVERS*\n"
                for m in movers[:5]:
                    emoji = "" if m['change_pct'] >= 0 else ""
                    msg += f"  {m['symbol']}: {m['price']:.2f} ({emoji}{m['change_pct']:.2f}%)\n"

        except Exception as e:
            logger.error(f"Error getting movers: {e}")

        msg += "\n_Have a great trading day!_"
        return msg

    def _buy_check_loop(self):
        """Background thread for buy opportunity checks every 5 minutes"""
        while self.running:
            try:
                self.check_buy_opportunities()
            except Exception as e:
                logger.error(f"Error in buy check loop: {e}")

            # Sleep for 5 minutes (300 seconds)
            for _ in range(300):
                if not self.running:
                    break
                time.sleep(1)

    def _daily_summary_loop(self):
        """Background thread for daily summary at 8 AM"""
        while self.running:
            try:
                now = datetime.now()
                # Check if it's 8:00 AM (within a 1 minute window)
                if now.hour == 8 and now.minute == 0:
                    self.send_daily_summary()
                    # Sleep for 61 seconds to avoid duplicate sends
                    time.sleep(61)
            except Exception as e:
                logger.error(f"Error in daily summary loop: {e}")

            # Check every 30 seconds
            time.sleep(30)

    def start(self):
        """Start the scheduler threads"""
        self.running = True

        # Start buy check thread
        self.buy_check_thread = threading.Thread(target=self._buy_check_loop, daemon=True)
        self.buy_check_thread.start()

        # Start daily summary thread
        self.daily_summary_thread = threading.Thread(target=self._daily_summary_loop, daemon=True)
        self.daily_summary_thread.start()

        logger.info("Scheduler started")

    def stop(self):
        """Stop the scheduler"""
        self.running = False
        logger.info("Scheduler stopped")

    def trigger_daily_summary(self):
        """Manually trigger daily summary"""
        self.send_daily_summary()

    def trigger_opportunity_check(self):
        """Manually trigger opportunity check"""
        self.check_buy_opportunities()


# Global scheduler instance
market_scheduler = MarketScheduler()


if __name__ == "__main__":
    # Test the scheduler functions
    print("Testing Scheduler...")

    # Test daily summary generation
    scheduler = MarketScheduler()
    summary = scheduler._generate_daily_summary()
    print("\nDaily Summary:")
    print(summary)

    # Test opportunity check
    print("\n\nChecking opportunities...")
    watchlist_data = get_bulk_stock_data(WATCHLIST[:5], period="3mo")
    opportunities = find_buy_opportunities(watchlist_data, min_strength=1)

    if opportunities:
        msg = scheduler._format_opportunities_message(opportunities)
        print("\nOpportunities Alert:")
        print(msg)
    else:
        print("No opportunities found")
