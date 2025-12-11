# MarketBuddy Setup Guide

## Folder Structure
```
MarketBuddy/
├── bot.py                 # Main bot file
├── data_fetcher.py        # Stock data fetching
├── analyzer.py            # Technical analysis
├── scheduler_runner.py    # Scheduled tasks
├── requirements.txt       # Python dependencies
├── .env                   # Environment variables
├── com.marketbuddy.service.plist  # macOS service file
└── logs/                  # Log files
```

## Step-by-Step Setup

### 1. Install Python 3.9 (if not installed)
```bash
# Using Homebrew
brew install python@3.9
```

### 2. Create Virtual Environment
```bash
cd /Users/sumitmalodiya/Desktop/MarketBuddy

# Create venv with Python 3.9
python3.9 -m venv venv

# Activate venv
source venv/bin/activate

# Verify Python version
python --version
# Should show: Python 3.9.x
```

### 3. Install Dependencies
```bash
# Upgrade pip first
python -m pip install --upgrade pip

# Install all packages
python -m pip install -r requirements.txt
```

### 4. Test the Bot
```bash
# Make sure venv is activated
source venv/bin/activate

# Run the bot
python bot.py
```

### 5. Setup 24/7 Service (launchd)

```bash
# Copy plist to LaunchAgents
cp /Users/sumitmalodiya/Desktop/MarketBuddy/com.marketbuddy.service.plist ~/Library/LaunchAgents/

# Load the service
launchctl load ~/Library/LaunchAgents/com.marketbuddy.service.plist

# Start the service
launchctl start com.marketbuddy.service
```

## Service Management Commands

### Check Status
```bash
launchctl list | grep marketbuddy
```

### View Logs
```bash
# Standard output
tail -f /Users/sumitmalodiya/Desktop/MarketBuddy/logs/stdout.log

# Error output
tail -f /Users/sumitmalodiya/Desktop/MarketBuddy/logs/stderr.log

# Bot logs
tail -f /Users/sumitmalodiya/Desktop/MarketBuddy/logs/bot.log
```

### Stop Service
```bash
launchctl stop com.marketbuddy.service
```

### Restart Service
```bash
launchctl stop com.marketbuddy.service
launchctl start com.marketbuddy.service
```

### Unload Service (disable)
```bash
launchctl unload ~/Library/LaunchAgents/com.marketbuddy.service.plist
```

### Reload Service (after changes)
```bash
launchctl unload ~/Library/LaunchAgents/com.marketbuddy.service.plist
launchctl load ~/Library/LaunchAgents/com.marketbuddy.service.plist
launchctl start com.marketbuddy.service
```

## Bot Commands

| Command | Description |
|---------|-------------|
| `/start` | Welcome message |
| `/stock SYMBOL` | Detailed stock analysis |
| `/summary` | NIFTY + SENSEX summary |
| `/news` | Latest market news |
| `/help` | Show help |

### Example Usage
```
/stock RELIANCE
/stock TCS
/stock INFY
/stock HDFCBANK
```

## Automatic Features

- **Buy Alerts**: Every 5 minutes, checks watchlist for buy opportunities
- **Daily Summary**: Sent at 8:00 AM IST

## Troubleshooting

### Bot not responding
1. Check if service is running: `launchctl list | grep marketbuddy`
2. Check logs: `tail -50 ~/Desktop/MarketBuddy/logs/stderr.log`
3. Restart service

### Module not found errors
```bash
source venv/bin/activate
pip install -r requirements.txt
```

### Permission issues
```bash
chmod +x bot.py
```

## Environment Variables (.env)
```
BOT_TOKEN=your_telegram_bot_token
ADMIN_CHAT_ID=your_telegram_chat_id
NEWSAPI_KEY=your_newsapi_key (optional)
```
