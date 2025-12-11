"""
Analyzer Module
Technical analysis, anomaly detection, and forecasting
"""

import pandas as pd
import numpy as np
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.trend import MACD, SMAIndicator, EMAIndicator
from ta.volatility import BollingerBands, AverageTrueRange
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from prophet import Prophet
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings('ignore')


def calculate_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate technical indicators using ta library

    Args:
        df: DataFrame with OHLCV data

    Returns:
        DataFrame with added technical indicators
    """
    if df.empty or len(df) < 20:
        return df

    df = df.copy()

    # Ensure column names are lowercase
    df.columns = [col.lower() for col in df.columns]

    try:
        # RSI (Relative Strength Index)
        rsi_indicator = RSIIndicator(close=df['close'], window=14)
        df['rsi'] = rsi_indicator.rsi()

        # MACD (Moving Average Convergence Divergence)
        macd_indicator = MACD(close=df['close'], window_fast=12, window_slow=26, window_sign=9)
        df['macd'] = macd_indicator.macd()
        df['macd_signal'] = macd_indicator.macd_signal()
        df['macd_hist'] = macd_indicator.macd_diff()

        # Simple Moving Averages
        sma_20 = SMAIndicator(close=df['close'], window=20)
        sma_50 = SMAIndicator(close=df['close'], window=50)
        df['sma_20'] = sma_20.sma_indicator()
        df['sma_50'] = sma_50.sma_indicator()

        # Exponential Moving Averages
        ema_12 = EMAIndicator(close=df['close'], window=12)
        ema_26 = EMAIndicator(close=df['close'], window=26)
        df['ema_12'] = ema_12.ema_indicator()
        df['ema_26'] = ema_26.ema_indicator()

        # Bollinger Bands
        bb_indicator = BollingerBands(close=df['close'], window=20, window_dev=2)
        df['bb_upper'] = bb_indicator.bollinger_hband()
        df['bb_middle'] = bb_indicator.bollinger_mavg()
        df['bb_lower'] = bb_indicator.bollinger_lband()

        # ATR (Average True Range)
        atr_indicator = AverageTrueRange(high=df['high'], low=df['low'], close=df['close'], window=14)
        df['atr'] = atr_indicator.average_true_range()

        # Volume SMA
        vol_sma = SMAIndicator(close=df['volume'].astype(float), window=20)
        df['volume_sma'] = vol_sma.sma_indicator()

        # Stochastic Oscillator
        stoch_indicator = StochasticOscillator(high=df['high'], low=df['low'], close=df['close'])
        df['stoch_k'] = stoch_indicator.stoch()
        df['stoch_d'] = stoch_indicator.stoch_signal()

    except Exception as e:
        print(f"Error calculating indicators: {e}")

    return df


def detect_anomalies(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detect price and volume anomalies using Isolation Forest

    Args:
        df: DataFrame with OHLCV and technical indicators

    Returns:
        DataFrame with anomaly flags
    """
    if df.empty or len(df) < 30:
        return df

    df = df.copy()

    try:
        # Features for anomaly detection
        features = ['close', 'volume', 'rsi', 'macd'] if 'rsi' in df.columns else ['close', 'volume']
        features = [f for f in features if f in df.columns]

        # Prepare data
        feature_df = df[features].dropna()

        if len(feature_df) < 20:
            df['anomaly'] = 0
            return df

        # Scale features
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(feature_df)

        # Isolation Forest
        iso_forest = IsolationForest(contamination=0.05, random_state=42, n_estimators=100)
        anomalies = iso_forest.fit_predict(scaled_features)

        # Map back to original dataframe
        df['anomaly'] = 0
        df.loc[feature_df.index, 'anomaly'] = anomalies
        df['anomaly'] = df['anomaly'].apply(lambda x: 1 if x == -1 else 0)

        # Calculate anomaly score
        df['anomaly_score'] = 0.0
        scores = iso_forest.decision_function(scaled_features)
        df.loc[feature_df.index, 'anomaly_score'] = -scores  # Higher score = more anomalous

    except Exception as e:
        print(f"Error detecting anomalies: {e}")
        df['anomaly'] = 0
        df['anomaly_score'] = 0.0

    return df


def generate_forecast(df: pd.DataFrame, periods: int = 7) -> dict:
    """
    Generate price forecast using Prophet

    Args:
        df: DataFrame with date and close price
        periods: Number of days to forecast

    Returns:
        Dictionary with forecast data
    """
    if df.empty or len(df) < 30:
        return {"error": "Insufficient data for forecasting"}

    try:
        # Prepare data for Prophet
        df_prophet = df[['date', 'close']].copy() if 'date' in df.columns else df[['Date', 'close']].copy()
        df_prophet.columns = ['ds', 'y']
        df_prophet['ds'] = pd.to_datetime(df_prophet['ds'])

        # Remove timezone if present
        if df_prophet['ds'].dt.tz is not None:
            df_prophet['ds'] = df_prophet['ds'].dt.tz_localize(None)

        # Train Prophet model
        model = Prophet(
            daily_seasonality=False,
            weekly_seasonality=True,
            yearly_seasonality=True,
            changepoint_prior_scale=0.05,
        )
        model.fit(df_prophet)

        # Make future dataframe
        future = model.make_future_dataframe(periods=periods)
        forecast = model.predict(future)

        # Get last known price and forecast
        last_price = df_prophet['y'].iloc[-1]
        last_date = df_prophet['ds'].iloc[-1]

        # Future predictions only
        future_forecast = forecast[forecast['ds'] > last_date][['ds', 'yhat', 'yhat_lower', 'yhat_upper']]

        if future_forecast.empty:
            return {"error": "No future dates in forecast"}

        # Calculate expected change
        final_forecast = future_forecast['yhat'].iloc[-1]
        expected_change = ((final_forecast - last_price) / last_price) * 100

        return {
            "current_price": last_price,
            "forecast_7d": final_forecast,
            "expected_change_pct": expected_change,
            "lower_bound": future_forecast['yhat_lower'].iloc[-1],
            "upper_bound": future_forecast['yhat_upper'].iloc[-1],
            "trend": "BULLISH" if expected_change > 1 else ("BEARISH" if expected_change < -1 else "NEUTRAL"),
            "forecast_dates": future_forecast['ds'].tolist(),
            "forecast_values": future_forecast['yhat'].tolist(),
        }

    except Exception as e:
        print(f"Error generating forecast: {e}")
        return {"error": str(e)}


def generate_signals(df: pd.DataFrame) -> dict:
    """
    Generate trading signals based on technical indicators

    Args:
        df: DataFrame with technical indicators

    Returns:
        Dictionary with signals and analysis
    """
    if df.empty or len(df) < 5:
        return {"error": "Insufficient data"}

    signals = {
        "overall": "NEUTRAL",
        "strength": 0,
        "indicators": {},
        "buy_signals": [],
        "sell_signals": [],
    }

    try:
        latest = df.iloc[-1]
        prev = df.iloc[-2] if len(df) > 1 else latest

        buy_count = 0
        sell_count = 0

        # RSI Signal
        if 'rsi' in df.columns and not pd.isna(latest.get('rsi')):
            rsi = latest['rsi']
            if rsi < 30:
                signals['indicators']['RSI'] = f"OVERSOLD ({rsi:.1f}) - BUY Signal"
                signals['buy_signals'].append(f"RSI oversold at {rsi:.1f}")
                buy_count += 2
            elif rsi > 70:
                signals['indicators']['RSI'] = f"OVERBOUGHT ({rsi:.1f}) - SELL Signal"
                signals['sell_signals'].append(f"RSI overbought at {rsi:.1f}")
                sell_count += 2
            else:
                signals['indicators']['RSI'] = f"NEUTRAL ({rsi:.1f})"

        # MACD Signal
        if 'macd' in df.columns and 'macd_signal' in df.columns:
            macd = latest.get('macd', 0)
            macd_signal = latest.get('macd_signal', 0)
            prev_macd = prev.get('macd', 0)
            prev_signal = prev.get('macd_signal', 0)

            if not (pd.isna(macd) or pd.isna(macd_signal)):
                # Bullish crossover
                if prev_macd <= prev_signal and macd > macd_signal:
                    signals['indicators']['MACD'] = "BULLISH CROSSOVER - BUY Signal"
                    signals['buy_signals'].append("MACD bullish crossover")
                    buy_count += 2
                # Bearish crossover
                elif prev_macd >= prev_signal and macd < macd_signal:
                    signals['indicators']['MACD'] = "BEARISH CROSSOVER - SELL Signal"
                    signals['sell_signals'].append("MACD bearish crossover")
                    sell_count += 2
                elif macd > macd_signal:
                    signals['indicators']['MACD'] = f"BULLISH (MACD > Signal)"
                    buy_count += 1
                else:
                    signals['indicators']['MACD'] = f"BEARISH (MACD < Signal)"
                    sell_count += 1

        # SMA Crossover
        if 'sma_20' in df.columns and 'sma_50' in df.columns:
            sma_20 = latest.get('sma_20')
            sma_50 = latest.get('sma_50')
            close = latest.get('close')

            if not (pd.isna(sma_20) or pd.isna(sma_50)):
                if sma_20 > sma_50:
                    signals['indicators']['SMA'] = f"BULLISH (SMA20 > SMA50)"
                    buy_count += 1
                else:
                    signals['indicators']['SMA'] = f"BEARISH (SMA20 < SMA50)"
                    sell_count += 1

                # Price vs SMA
                if not pd.isna(close):
                    if close > sma_20 and close > sma_50:
                        signals['buy_signals'].append("Price above both SMAs")
                        buy_count += 1
                    elif close < sma_20 and close < sma_50:
                        signals['sell_signals'].append("Price below both SMAs")
                        sell_count += 1

        # Bollinger Bands
        if 'bb_lower' in df.columns and 'bb_upper' in df.columns:
            close = latest.get('close')
            bb_lower = latest.get('bb_lower')
            bb_upper = latest.get('bb_upper')

            if not (pd.isna(bb_lower) or pd.isna(bb_upper) or pd.isna(close)):
                if close <= bb_lower:
                    signals['indicators']['BB'] = "AT LOWER BAND - Potential BUY"
                    signals['buy_signals'].append("Price at Bollinger lower band")
                    buy_count += 1
                elif close >= bb_upper:
                    signals['indicators']['BB'] = "AT UPPER BAND - Potential SELL"
                    signals['sell_signals'].append("Price at Bollinger upper band")
                    sell_count += 1
                else:
                    signals['indicators']['BB'] = "WITHIN BANDS - Neutral"

        # Volume Analysis
        if 'volume' in df.columns and 'volume_sma' in df.columns:
            volume = latest.get('volume')
            vol_sma = latest.get('volume_sma')

            if not (pd.isna(volume) or pd.isna(vol_sma)) and vol_sma > 0:
                vol_ratio = volume / vol_sma
                if vol_ratio > 1.5:
                    signals['indicators']['Volume'] = f"HIGH VOLUME ({vol_ratio:.1f}x avg)"
                elif vol_ratio < 0.5:
                    signals['indicators']['Volume'] = f"LOW VOLUME ({vol_ratio:.1f}x avg)"
                else:
                    signals['indicators']['Volume'] = f"NORMAL VOLUME ({vol_ratio:.1f}x avg)"

        # Calculate overall signal
        net_signal = buy_count - sell_count
        signals['strength'] = net_signal

        if net_signal >= 3:
            signals['overall'] = "STRONG BUY"
        elif net_signal >= 1:
            signals['overall'] = "BUY"
        elif net_signal <= -3:
            signals['overall'] = "STRONG SELL"
        elif net_signal <= -1:
            signals['overall'] = "SELL"
        else:
            signals['overall'] = "NEUTRAL"

    except Exception as e:
        print(f"Error generating signals: {e}")
        signals['error'] = str(e)

    return signals


def full_analysis(df: pd.DataFrame, include_forecast: bool = True) -> dict:
    """
    Perform full technical analysis

    Args:
        df: DataFrame with OHLCV data
        include_forecast: Whether to include Prophet forecast

    Returns:
        Complete analysis dictionary
    """
    if df.empty:
        return {"error": "No data available"}

    # Calculate indicators
    df = calculate_technical_indicators(df)

    # Detect anomalies
    df = detect_anomalies(df)

    # Generate signals
    signals = generate_signals(df)

    # Get latest values
    latest = df.iloc[-1]

    analysis = {
        "current_price": latest.get('close', 0),
        "open": latest.get('open', 0),
        "high": latest.get('high', 0),
        "low": latest.get('low', 0),
        "volume": latest.get('volume', 0),
        "date": str(latest.get('date', latest.name)),
        "indicators": {
            "rsi": latest.get('rsi', None),
            "macd": latest.get('macd', None),
            "macd_signal": latest.get('macd_signal', None),
            "sma_20": latest.get('sma_20', None),
            "sma_50": latest.get('sma_50', None),
            "bb_upper": latest.get('bb_upper', None),
            "bb_lower": latest.get('bb_lower', None),
            "atr": latest.get('atr', None),
        },
        "signals": signals,
        "anomaly": {
            "is_anomaly": bool(latest.get('anomaly', 0)),
            "score": latest.get('anomaly_score', 0),
        }
    }

    # Add forecast if requested
    if include_forecast:
        analysis['forecast'] = generate_forecast(df)

    return analysis


def find_buy_opportunities(watchlist_data: dict, min_strength: int = 2) -> list:
    """
    Find buy opportunities from watchlist

    Args:
        watchlist_data: Dictionary with symbol -> DataFrame
        min_strength: Minimum signal strength to consider

    Returns:
        List of buy opportunities
    """
    opportunities = []

    for symbol, df in watchlist_data.items():
        try:
            if df.empty or len(df) < 30:
                continue

            analysis = full_analysis(df, include_forecast=False)

            if 'error' not in analysis:
                signals = analysis.get('signals', {})
                strength = signals.get('strength', 0)

                if strength >= min_strength:
                    opportunities.append({
                        "symbol": symbol,
                        "current_price": analysis.get('current_price'),
                        "signal": signals.get('overall'),
                        "strength": strength,
                        "buy_signals": signals.get('buy_signals', []),
                        "rsi": analysis['indicators'].get('rsi'),
                    })

        except Exception as e:
            print(f"Error analyzing {symbol}: {e}")

    # Sort by strength
    opportunities.sort(key=lambda x: x['strength'], reverse=True)

    return opportunities


if __name__ == "__main__":
    # Test the analyzer
    from data_fetcher import get_stock_data, WATCHLIST

    print("Testing Analyzer...")

    # Get sample data
    df = get_stock_data("RELIANCE", period="6mo")

    if not df.empty:
        print(f"\n1. Data shape: {df.shape}")

        # Calculate indicators
        df_with_indicators = calculate_technical_indicators(df)
        print(f"\n2. Columns after indicators: {df_with_indicators.columns.tolist()}")

        # Detect anomalies
        df_with_anomalies = detect_anomalies(df_with_indicators)
        anomaly_count = df_with_anomalies['anomaly'].sum()
        print(f"\n3. Anomalies detected: {anomaly_count}")

        # Generate signals
        signals = generate_signals(df_with_anomalies)
        print(f"\n4. Signals: {signals}")

        # Full analysis
        analysis = full_analysis(df)
        print(f"\n5. Full Analysis:")
        print(f"   Current Price: {analysis.get('current_price')}")
        print(f"   Overall Signal: {analysis['signals']['overall']}")
        print(f"   Forecast: {analysis.get('forecast', {}).get('trend', 'N/A')}")
    else:
        print("Failed to fetch data")
