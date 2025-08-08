import os
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.express as px
import time
from dotenv import load_dotenv
import asyncio
import aiohttp
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Load environment variables from .env file
load_dotenv()

COINGECKO_API_URL = "https://api.coingecko.com/api/v3"

# API Key configuration
COINGECKO_API_KEY = os.environ.get('COINGECKO_API_KEY', '')

# Debug API key status
if COINGECKO_API_KEY:
    print(f"[CoinGecko] API key found: {COINGECKO_API_KEY[:10]}...")
else:
    print("[CoinGecko] No API key found - using free tier limits")

# Increased limits with API key
COINS_PER_REQUEST = 100 if COINGECKO_API_KEY else 25

# Technical Analysis Functions
def calculate_rsi(prices, period=14):
    """Calculate Relative Strength Index"""
    df = pd.DataFrame(prices, columns=['timestamp', 'price'])
    delta = df['price'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    rsi_value = rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else 50.0
    return float(rsi_value)

def calculate_macd(prices, fast=12, slow=26, signal=9):
    """Calculate MACD indicator"""
    df = pd.DataFrame(prices, columns=['timestamp', 'price'])
    ema_fast = df['price'].ewm(span=fast).mean()
    ema_slow = df['price'].ewm(span=slow).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal).mean()
    histogram = macd_line - signal_line
    return {
        'macd': float(macd_line.iloc[-1]) if not pd.isna(macd_line.iloc[-1]) else 0.0,
        'signal': float(signal_line.iloc[-1]) if not pd.isna(signal_line.iloc[-1]) else 0.0,
        'histogram': float(histogram.iloc[-1]) if not pd.isna(histogram.iloc[-1]) else 0.0,
        'trend': 'bullish' if macd_line.iloc[-1] > signal_line.iloc[-1] else 'bearish'
    }

def calculate_bollinger_bands(prices, period=20, std_dev=2):
    """Calculate Bollinger Bands"""
    df = pd.DataFrame(prices, columns=['timestamp', 'price'])
    sma = df['price'].rolling(window=period).mean()
    std = df['price'].rolling(window=period).std()
    upper_band = sma + (std * std_dev)
    lower_band = sma - (std * std_dev)
    current_price = df['price'].iloc[-1]
    
    # Calculate position within bands
    if not pd.isna(upper_band.iloc[-1]) and not pd.isna(lower_band.iloc[-1]):
        band_width = upper_band.iloc[-1] - lower_band.iloc[-1]
        position = (current_price - lower_band.iloc[-1]) / band_width if band_width > 0 else 0.5
    else:
        position = 0.5
    
    return {
        'upper': float(upper_band.iloc[-1]) if not pd.isna(upper_band.iloc[-1]) else 0.0,
        'middle': float(sma.iloc[-1]) if not pd.isna(sma.iloc[-1]) else 0.0,
        'lower': float(lower_band.iloc[-1]) if not pd.isna(lower_band.iloc[-1]) else 0.0,
        'position': float(position),
        'squeeze': float((upper_band.iloc[-1] - lower_band.iloc[-1]) / sma.iloc[-1]) if not pd.isna(sma.iloc[-1]) else 0.0
    }

def calculate_moving_averages(prices):
    """Calculate various moving averages"""
    df = pd.DataFrame(prices, columns=['timestamp', 'price'])
    current_price = df['price'].iloc[-1]
    
    ma_7 = df['price'].rolling(7).mean().iloc[-1]
    ma_14 = df['price'].rolling(14).mean().iloc[-1]
    ma_21 = df['price'].rolling(21).mean().iloc[-1]
    ma_50 = df['price'].rolling(50).mean().iloc[-1] if len(df) >= 50 else ma_21
    
    return {
        'ma_7': float(ma_7) if not pd.isna(ma_7) else 0.0,
        'ma_14': float(ma_14) if not pd.isna(ma_14) else 0.0,
        'ma_21': float(ma_21) if not pd.isna(ma_21) else 0.0,
        'ma_50': float(ma_50) if not pd.isna(ma_50) else 0.0,
        'golden_cross': ma_7 > ma_21,  # Short-term above long-term
        'death_cross': ma_7 < ma_21,   # Short-term below long-term
        'above_all_ma': current_price > max(ma_7, ma_14, ma_21, ma_50),
        'below_all_ma': current_price < min(ma_7, ma_14, ma_21, ma_50)
    }

# Volume Analysis
def get_volume_data(coin_id, days=7):
    """Get volume data for analysis"""
    url = f"{COINGECKO_API_URL}/coins/{coin_id}/market_chart"
    params = {
        'vs_currency': 'usd',
        'days': days
    }
    headers = {}
    if COINGECKO_API_KEY:
        headers['X-CG-API-Key'] = COINGECKO_API_KEY
    
    try:
        response = requests.get(url, params=params, headers=headers, timeout=30)
        if response.status_code == 200:
            data = response.json()
            return data.get('total_volumes', [])
    except:
        return []

def analyze_volume(volumes):
    """Analyze volume patterns"""
    if not volumes or len(volumes) < 2:
        return {}
    
    df = pd.DataFrame(volumes, columns=['timestamp', 'volume'])
    current_volume = df['volume'].iloc[-1]
    avg_volume = df['volume'].mean()
    volume_change = (current_volume - df['volume'].iloc[-2]) / df['volume'].iloc[-2] if len(df) > 1 else 0
    
    return {
        'current_volume': current_volume,
        'avg_volume': avg_volume,
        'volume_ratio': current_volume / avg_volume if avg_volume > 0 else 1,
        'volume_change': volume_change,
        'volume_trend': 'high' if current_volume > avg_volume * 1.5 else 'normal' if current_volume > avg_volume * 0.5 else 'low'
    }

# Market Sentiment Analysis
def analyze_market_sentiment(coin_id):
    """Analyze market sentiment using social metrics"""
    url = f"{COINGECKO_API_URL}/coins/{coin_id}"
    headers = {}
    if COINGECKO_API_KEY:
        headers['X-CG-API-Key'] = COINGECKO_API_KEY
    
    try:
        response = requests.get(url, headers=headers, timeout=30)
        if response.status_code == 200:
            data = response.json()
            community_data = data.get('community_data', {})
            return {
                'reddit_subscribers': community_data.get('reddit_subscribers', 0),
                'twitter_followers': community_data.get('twitter_followers', 0),
                'telegram_channel_user_count': community_data.get('telegram_channel_user_count', 0),
                'sentiment_score': calculate_sentiment_score(community_data),
                'community_score': data.get('community_score', 0),
                'developer_score': data.get('developer_score', 0),
                'public_interest_score': data.get('public_interest_score', 0)
            }
    except:
        return {}

def calculate_sentiment_score(community_data):
    """Calculate sentiment score based on community metrics"""
    score = 0
    reddit = community_data.get('reddit_subscribers', 0)
    twitter = community_data.get('twitter_followers', 0)
    telegram = community_data.get('telegram_channel_user_count', 0)
    
    # Normalize and weight different metrics
    if reddit > 0:
        score += min(reddit / 100000, 1) * 0.4  # Max 40% from Reddit
    if twitter > 0:
        score += min(twitter / 1000000, 1) * 0.4  # Max 40% from Twitter
    if telegram > 0:
        score += min(telegram / 50000, 1) * 0.2  # Max 20% from Telegram
    
    return min(score * 100, 100)  # Scale to 0-100

# Pattern Recognition
def detect_candlestick_patterns(prices):
    """Detect common candlestick patterns"""
    patterns = []
    if len(prices) < 3:
        return patterns
    
    # Convert to OHLC-like data (simplified)
    df = pd.DataFrame(prices, columns=['timestamp', 'price'])
    
    # Detect patterns
    if detect_doji(df):
        patterns.append('Doji')
    if detect_hammer(df):
        patterns.append('Hammer')
    if detect_engulfing(df):
        patterns.append('Engulfing')
    if detect_trend_reversal(df):
        patterns.append('Trend Reversal')
    
    return patterns

def detect_doji(df):
    """Detect Doji pattern"""
    if len(df) < 2:
        return False
    
    current = df['price'].iloc[-1]
    previous = df['price'].iloc[-2]
    change = abs(current - previous) / previous if previous > 0 else 0
    
    return change < 0.01  # Less than 1% change

def detect_hammer(df):
    """Detect Hammer pattern"""
    if len(df) < 3:
        return False
    
    current = df['price'].iloc[-1]
    avg_price = df['price'].rolling(3).mean().iloc[-1]
    
    return current > avg_price * 1.02  # Price significantly above average

def detect_engulfing(df):
    """Detect Engulfing pattern"""
    if len(df) < 2:
        return False
    
    current = df['price'].iloc[-1]
    previous = df['price'].iloc[-2]
    
    return abs(current - previous) / previous > 0.05  # More than 5% change

def detect_trend_reversal(df):
    """Detect potential trend reversal"""
    if len(df) < 5:
        return False
    
    recent_prices = df['price'].tail(5)
    first_half = recent_prices.head(3).mean()
    second_half = recent_prices.tail(2).mean()
    
    return abs(second_half - first_half) / first_half > 0.03

def detect_support_resistance(prices):
    """Detect support and resistance levels"""
    df = pd.DataFrame(prices, columns=['timestamp', 'price'])
    
    if len(df) < 10:
        return {'support': [], 'resistance': []}
    
    # Find local minima and maxima
    peaks = df['price'].rolling(window=5, center=True).max()
    troughs = df['price'].rolling(window=5, center=True).min()
    
    # Identify significant levels
    support_levels = []
    resistance_levels = []
    
    for i in range(2, len(df) - 2):
        price = df['price'].iloc[i]
        if price == peaks.iloc[i]:
            resistance_levels.append(price)
        elif price == troughs.iloc[i]:
            support_levels.append(price)
    
    # Get unique levels and sort
    support_levels = sorted(list(set(support_levels)))[:3]
    resistance_levels = sorted(list(set(resistance_levels)), reverse=True)[:3]
    
    return {
        'support': support_levels,
        'resistance': resistance_levels
    }

# Machine Learning Prediction
def predict_price_ml(df):
    """Predict future price using simplified ML approach"""
    try:
        if len(df) < 20:  # Reduced minimum data requirement
            return None, 0
        
        # Simple features to avoid memory issues
        df['returns'] = df['price'].pct_change()
        df['ma_7'] = df['price'].rolling(window=7).mean()
        
        # Remove NaN values
        df = df.dropna()
        
        if len(df) < 10:  # Further reduced requirement
            return None, 0
        
        # Use only 2 features to reduce complexity
        features = ['returns', 'ma_7']
        X = df[features].values[:-1]  # All but last row
        y = df['price'].values[1:]    # All but first row
        
        # Ensure X and y have same length
        min_len = min(len(X), len(y))
        X = X[:min_len]
        y = y[:min_len]
        
        if len(X) < 5:  # Very minimal requirement
            return None, 0
        
        # Use simpler model with fewer estimators
        model = RandomForestRegressor(n_estimators=10, random_state=42, max_depth=5)
        model.fit(X, y)
        
        # Predict next price
        last_features = df[features].iloc[-1].values.reshape(1, -1)
        prediction = model.predict(last_features)[0]
        
        # Calculate confidence based on model performance
        confidence = min(0.8, model.score(X, y) + 0.3)
        
        return prediction, confidence
        
    except Exception as e:
        print(f"ML prediction error: {e}")
        return None, 0

# Risk Management
def calculate_risk_metrics(prices, current_price):
    """Calculate comprehensive risk metrics"""
    df = pd.DataFrame(prices, columns=['timestamp', 'price'])
    returns = df['price'].pct_change().dropna()
    
    if len(returns) < 2:
        return {}
    
    # Value at Risk (VaR)
    var_95 = np.percentile(returns, 5)  # 95% VaR
    
    # Maximum Drawdown
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    max_drawdown = drawdown.min()
    
    # Sharpe Ratio (assuming risk-free rate of 2%)
    risk_free_rate = 0.02 / 365  # Daily risk-free rate
    excess_returns = returns - risk_free_rate
    sharpe_ratio = excess_returns.mean() / returns.std() if returns.std() > 0 else 0
    
    # Risk score (0-100, higher = riskier)
    risk_score = calculate_risk_score(var_95, max_drawdown, sharpe_ratio)
    
    return {
        'var_95': var_95,
        'max_drawdown': max_drawdown,
        'sharpe_ratio': sharpe_ratio,
        'volatility': returns.std(),
        'risk_score': risk_score,
        'risk_level': 'high' if risk_score > 70 else 'medium' if risk_score > 30 else 'low'
    }

def calculate_risk_score(var_95, max_drawdown, sharpe_ratio):
    """Calculate risk score (0-100)"""
    # Normalize metrics to 0-100 scale
    var_score = min(abs(var_95) * 1000, 100)  # VaR contribution
    drawdown_score = min(abs(max_drawdown) * 200, 100)  # Drawdown contribution
    sharpe_score = max(0, (1 - sharpe_ratio) * 50)  # Sharpe contribution (inverse)
    
    # Weighted average
    risk_score = (var_score * 0.4 + drawdown_score * 0.4 + sharpe_score * 0.2)
    return min(risk_score, 100)

# Enhanced Data Collection
async def fetch_coin_data_async(coin_ids):
    """Fetch multiple coin data concurrently"""
    async with aiohttp.ClientSession() as session:
        tasks = []
        for coin_id in coin_ids:
            task = fetch_single_coin_data(session, coin_id)
            tasks.append(task)
        return await asyncio.gather(*tasks)

async def fetch_single_coin_data(session, coin_id):
    """Fetch data for a single coin"""
    url = f"{COINGECKO_API_URL}/coins/{coin_id}/market_chart"
    params = {'vs_currency': 'usd', 'days': 7}
    headers = {}
    if COINGECKO_API_KEY:
        headers['X-CG-API-Key'] = COINGECKO_API_KEY
    
    try:
        async with session.get(url, params=params, headers=headers) as response:
            if response.status == 200:
                data = await response.json()
                return {'id': coin_id, 'data': data}
            else:
                return {'id': coin_id, 'data': None}
    except:
        return {'id': coin_id, 'data': None}

def get_top_coins():
    """Get top coins by market cap (aim for 100 items, paginate if needed)"""
    url = f"{COINGECKO_API_URL}/coins/markets"
    target_count = 100
    params = {
        'vs_currency': 'usd',
        'order': 'market_cap_desc',
        'per_page': COINS_PER_REQUEST,
        'page': 1,
        'sparkline': False
    }

    # Add API key to headers if available
    headers = {}
    if COINGECKO_API_KEY:
        # Try both header names for compatibility
        headers['X-CG-API-Key'] = COINGECKO_API_KEY
        headers['X-CG-Demo-API-Key'] = COINGECKO_API_KEY
        print(f"[CoinGecko] Using API key for enhanced rate limits")

    print(f"[CoinGecko] Fetching top coins (target={target_count}): {url} params={params}")
    print(f"[CoinGecko] Headers: {headers}")
    try:
        collected = []
        # Determine how many pages we need to hit to reach target_count
        per_page = params['per_page']
        max_pages = (target_count + per_page - 1) // per_page
        for page in range(1, max_pages + 1):
            params['page'] = page
            response = requests.get(url, params=params, headers=headers, timeout=30)
            print(f"[CoinGecko] Page {page} status: {response.status_code}")
            if response.status_code != 200:
                print(f"[CoinGecko] Error: {response.text}")
                break
            data = response.json() or []
            if not data:
                print("[CoinGecko] Warning: Empty page received; stopping pagination.")
                break
            collected.extend(data)
            if len(collected) >= target_count:
                break
            # Gentle delay to be polite to the API when paginating without a key
            if not COINGECKO_API_KEY:
                time.sleep(1)

        if not collected:
            print(f"[CoinGecko] Warning: Empty data returned. Possible rate limit or API error.")
            return get_fallback_coins()

        return collected[:target_count]
    except requests.exceptions.RequestException as e:
        print(f"[CoinGecko] Request error: {str(e)}")
        return get_fallback_coins()
    except Exception as e:
        print(f"[CoinGecko] Unexpected error: {str(e)}")
        return get_fallback_coins()

def get_fallback_coins():
    """Return fallback coin data when API is unavailable"""
    print("[CoinGecko] Using fallback coin data")
    return [
        {
            'id': 'bitcoin',
            'name': 'Bitcoin',
            'symbol': 'btc',
            'current_price': 45000.0
        },
        {
            'id': 'ethereum',
            'name': 'Ethereum',
            'symbol': 'eth',
            'current_price': 2800.0
        },
        {
            'id': 'binancecoin',
            'name': 'BNB',
            'symbol': 'bnb',
            'current_price': 320.0
        },
        {
            'id': 'cardano',
            'name': 'Cardano',
            'symbol': 'ada',
            'current_price': 0.45
        },
        {
            'id': 'solana',
            'name': 'Solana',
            'symbol': 'sol',
            'current_price': 95.0
        }
    ]

def get_price_history(coin_id, days=7):
    """Get price history for a specific coin, with API key if available"""
    url = f"{COINGECKO_API_URL}/coins/{coin_id}/market_chart"
    params = {
        'vs_currency': 'usd',
        'days': days
    }
    
    # Add API key to headers if available
    headers = {}
    if COINGECKO_API_KEY:
        # Try both header names for compatibility
        headers['X-CG-API-Key'] = COINGECKO_API_KEY
        headers['X-CG-Demo-API-Key'] = COINGECKO_API_KEY
    
    print(f"[CoinGecko] Fetching price history for {coin_id}: {url} params={params}")
    try:
        response = requests.get(url, params=params, headers=headers, timeout=10)  # Further reduced timeout
        if response.status_code == 200:
            data = response.json()
            if data and 'prices' in data and len(data['prices']) > 0:
                return data
            else:
                print(f"[CoinGecko] Invalid data returned for {coin_id}")
                return get_fallback_price_history(coin_id, days)
        else:
            print(f"[CoinGecko] Error status {response.status_code} for {coin_id}: {response.text}")
            return get_fallback_price_history(coin_id, days)
    except requests.exceptions.Timeout:
        print(f"[CoinGecko] Timeout for {coin_id} - using fallback")
        return get_fallback_price_history(coin_id, days)
    except requests.exceptions.RequestException as e:
        print(f"[CoinGecko] Request error for {coin_id}: {str(e)} - using fallback")
        return get_fallback_price_history(coin_id, days)
    except Exception as e:
        print(f"[CoinGecko] Unexpected error for {coin_id}: {str(e)} - using fallback")
        return get_fallback_price_history(coin_id, days)

def get_fallback_price_history(coin_id, days=7):
    """Return fallback price history when API is unavailable"""
    print(f"[CoinGecko] Using fallback price history for {coin_id}")
    current_time = int(time.time() * 1000)
    prices = []
    
    # Generate mock price data for the last 'days' days
    for i in range(days * 24):  # 24 data points per day
        timestamp = current_time - (days * 24 - i) * 3600000  # 1 hour intervals
        # Generate realistic price movement
        base_price = 100.0
        if coin_id == 'bitcoin':
            base_price = 45000.0
        elif coin_id == 'ethereum':
            base_price = 2800.0
        elif coin_id == 'binancecoin':
            base_price = 320.0
        elif coin_id == 'cardano':
            base_price = 0.45
        elif coin_id == 'solana':
            base_price = 95.0
        
        # Add some random variation
        import random
        variation = random.uniform(-0.05, 0.05)  # ±5% variation
        price = base_price * (1 + variation)
        prices.append([timestamp, price])
    
    return {'prices': prices}

def calculate_volatility(prices):
    """Calculate volatility (standard deviation) of price changes"""
    df = pd.DataFrame(prices, columns=['timestamp', 'price'])
    df['price_change'] = df['price'].pct_change()
    volatility = df['price_change'].std()
    return float(volatility) if not pd.isna(volatility) else 0.0

def analyze():
    import json
    from flask import jsonify
    start_time = time.time()
    try:
        # Get top coins (reduced to prevent timeouts)
        coins = get_top_coins()
        if not coins:
            return jsonify({
                'status': 'error',
                'message': 'Failed to fetch coin data from CoinGecko API. Please try again later.'
            })
        
        # Check if we're using fallback data
        using_fallback = len(coins) <= 5  # Fallback has 5 coins
        
        results = []
        raw_scores = []
        coin_results = []
        
        # Prepare concurrent fetch of price data when API key is present
        price_data_by_id = {}
        coin_batch = coins[:100]
        if COINGECKO_API_KEY:
            try:
                coin_ids = [c.get('id', '') for c in coin_batch if c.get('id')]
                if coin_ids:
                    fetched = asyncio.run(fetch_coin_data_async(coin_ids))
                    for item in fetched:
                        cid = item.get('id') if isinstance(item, dict) else None
                        cdata = item.get('data') if isinstance(item, dict) else None
                        if cid and cdata and 'prices' in cdata and len(cdata['prices']) > 0:
                            price_data_by_id[cid] = cdata
            except Exception as e:
                print(f"Concurrent fetch error, falling back to sequential: {e}")

        # Analyze up to 100 coins
        for coin in coin_batch:
            try:
                # Get price history
                price_history_payload = None
                if COINGECKO_API_KEY and price_data_by_id.get(coin['id']):
                    price_history_payload = price_data_by_id[coin['id']]
                else:
                    ph = get_price_history(coin['id'])
                    if ph and isinstance(ph, dict):
                        price_history_payload = ph

                if not price_history_payload or 'prices' not in price_history_payload:
                    print(f"Error analyzing {coin.get('name','?')}: Invalid price data")
                    continue

                price_data = price_history_payload['prices']
                df = pd.DataFrame(price_data, columns=['timestamp', 'price'])
                
                if len(df) < 10:  # Need sufficient data
                    print(f"Error analyzing {coin.get('name','?')}: Insufficient price data")
                    continue
                    
                current_price = df['price'].iloc[-1]

                # Basic metrics
                volatility = calculate_volatility(price_data)
                price_change = (df['price'].iloc[-1] - df['price'].iloc[0]) / df['price'].iloc[0]

                # Enhanced Technical Analysis
                rsi = calculate_rsi(price_data)
                macd_data = calculate_macd(price_data)
                bollinger_data = calculate_bollinger_bands(price_data)
                ma_data = calculate_moving_averages(price_data)
                
                # Volume Analysis (use real volumes when available via API key)
                volume_analysis = {'volume_trend': 'normal'}
                try:
                    volumes = None
                    if COINGECKO_API_KEY and isinstance(price_history_payload, dict) and 'total_volumes' in price_history_payload:
                        volumes = price_history_payload.get('total_volumes')
                    elif COINGECKO_API_KEY:
                        volumes = get_volume_data(coin['id'])
                    if volumes:
                        volume_analysis = analyze_volume(volumes)
                except Exception as e:
                    print(f"Volume analysis error for {coin.get('name','?')}: {e}")
                
                # Market Sentiment (use real sentiment when API key is present)
                if COINGECKO_API_KEY:
                    try:
                        sentiment_data = analyze_market_sentiment(coin['id']) or {'sentiment_score': 50}
                    except Exception as e:
                        print(f"Sentiment error for {coin.get('name','?')}: {e}")
                        sentiment_data = {'sentiment_score': 50}
                else:
                    sentiment_data = {'sentiment_score': 50}
                
                # Pattern Recognition
                patterns = detect_candlestick_patterns(price_data)
                support_resistance = detect_support_resistance(price_data)
                
                # Machine Learning Prediction (simplified)
                try:
                    ml_prediction, confidence = predict_price_ml(df)
                except Exception as e:
                    print(f"ML prediction error for {coin['name']}: {e}")
                    ml_prediction, confidence = None, 0
                
                # Risk Management (simplified)
                try:
                    risk_metrics = calculate_risk_metrics(price_data, current_price)
                    risk_score = calculate_risk_score(
                        risk_metrics['var_95'],
                        risk_metrics['max_drawdown'],
                        risk_metrics['sharpe_ratio']
                    )
                except Exception as e:
                    print(f"Risk calculation error for {coin['name']}: {e}")
                    risk_metrics = {'var_95': 0, 'max_drawdown': 0, 'sharpe_ratio': 0}
                    risk_score = 0.5

                # Enhanced Trend Detection
                if ma_data['golden_cross'] and bollinger_data['position'] > 0.7:
                    trend = 'Strong Uptrend'
                elif ma_data['death_cross'] and bollinger_data['position'] < 0.3:
                    trend = 'Strong Downtrend'
                elif ma_data['above_all_ma']:
                    trend = 'Uptrend'
                elif ma_data['below_all_ma']:
                    trend = 'Downtrend'
                else:
                    trend = 'Sideways'

                # Enhanced Breakout Detection
                breakout = False
                if len(df) > 3:
                    recent_high = df['price'].tail(10).max()
                    breakout = current_price > recent_high * 1.02  # 2% above recent high

                # Advanced Opportunity Score Calculation
                explanation = []
                score_components = []
                
                # Risk-adjusted return
                if volatility > 0:
                    risk_adjusted = price_change / volatility
                    score_components.append(risk_adjusted)
                    explanation.append(f'Risk-adjusted return: {risk_adjusted:.2f}')
                else:
                    risk_adjusted = 0
                    explanation.append('Low volatility - stable price')
                
                # Technical indicator bonuses
                if rsi < 30:
                    score_components.append(0.5)
                    explanation.append('Oversold (RSI < 30)')
                elif rsi > 70:
                    score_components.append(-0.3)
                    explanation.append('Overbought (RSI > 70)')
                
                if macd_data['trend'] == 'bullish':
                    score_components.append(0.3)
                    explanation.append('Bullish MACD')
                
                if bollinger_data['position'] < 0.2:
                    score_components.append(0.4)
                    explanation.append('Near support (Bollinger)')
                elif bollinger_data['position'] > 0.8:
                    score_components.append(-0.2)
                    explanation.append('Near resistance (Bollinger)')
                
                # Volume analysis bonus
                if volume_analysis.get('volume_trend') == 'high':
                    score_components.append(0.3)
                    explanation.append('High volume activity')
                
                # Sentiment bonus
                if sentiment_data.get('sentiment_score', 0) > 70:
                    score_components.append(0.2)
                    explanation.append('Strong community sentiment')
                
                # Pattern recognition bonus
                if patterns:
                    score_components.append(0.4)
                    explanation.append(f'Pattern detected: {", ".join(patterns)}')
                
                # ML prediction bonus
                if ml_prediction is not None:
                    score_components.append(0.3)
                    explanation.append(f'ML predicts {ml_prediction:.2f} with confidence {confidence:.2f}')
                
                # Risk adjustment
                if risk_metrics.get('risk_level') == 'high':
                    score_components.append(-0.3)
                    explanation.append('High risk - proceed with caution')
                
                # Calculate final score
                base_score = sum(score_components)
                
                # Trend and breakout bonuses
                if trend == 'Strong Uptrend':
                    base_score += 0.8
                    explanation.append('Strong uptrend detected')
                elif trend == 'Uptrend':
                    base_score += 0.4
                    explanation.append('Uptrend detected')
                elif trend == 'Strong Downtrend':
                    base_score -= 0.5
                    explanation.append('Strong downtrend - avoid')
                
                if breakout:
                    base_score += 0.6
                    explanation.append('Breakout pattern detected')
                
                # Price momentum analysis
                if price_change > 0.15:
                    explanation.append('Strong positive momentum')
                elif price_change < -0.15:
                    explanation.append('Significant decline - potential reversal')
                
                if not explanation:
                    explanation = ['No strong signals detected']

                coin_results.append({
                    'id': coin.get('id', ''),
                    'name': coin['name'],
                    'symbol': coin['symbol'],
                    'current_price': coin['current_price'],
                    'price_change': price_change,
                    'volatility': volatility,
                    'trend': trend,
                    'breakout': int(breakout),  # Convert boolean to int
                    'rsi': rsi,
                    'macd_trend': macd_data['trend'],
                    'bollinger_position': bollinger_data['position'],
                    'volume_trend': volume_analysis.get('volume_trend', 'normal'),
                    'sentiment_score': sentiment_data.get('sentiment_score', 0),
                    'patterns': patterns,
                    'support_levels': support_resistance.get('support', []),
                    'resistance_levels': support_resistance.get('resistance', []),
                    'ml_prediction': ml_prediction,
                    'risk_level': risk_metrics.get('risk_level', 'medium'),
                    'risk_score': risk_metrics.get('risk_score', 50),
                    'raw_score': base_score,
                    'explanation': '; '.join(explanation)
                })
                raw_scores.append(base_score)
                
            except Exception as e:
                print(f"Error analyzing {coin['name']}: {str(e)}")
                continue

        # Normalize scores
        results = []
        if coin_results:
            min_score = min(raw_scores)
            max_score = max(raw_scores)
            for c in coin_results:
                if max_score == min_score:
                    norm_score = 100
                else:
                    norm_score = int(round((c['raw_score'] - min_score) / (max_score - min_score) * 100))
                c['opportunity_score'] = norm_score
                del c['raw_score']
                results.append(c)
        
        # Enhanced sorting: opportunity score, risk level, volume trend, sentiment
        def sort_key(x):
            risk_priority = {'low': 3, 'medium': 2, 'high': 1}
            volume_priority = {'high': 3, 'normal': 2, 'low': 1}
            return (
                -x['opportunity_score'],
                risk_priority.get(x.get('risk_level', 'medium'), 2),
                volume_priority.get(x.get('volume_trend', 'normal'), 2),
                -x.get('sentiment_score', 0)
            )
        
        results.sort(key=sort_key)
        
        # Check if we have any results
        if not results:
            return jsonify({
                'status': 'error',
                'message': 'No coins could be analyzed. This might be due to API rate limits or temporary issues.'
            })
        
        # Calculate analysis time
        elapsed = time.time() - start_time
        if COINGECKO_API_KEY:
            message = f'Advanced Analysis Complete: {len(results)} coins analyzed with enhanced technical indicators, ML predictions, and risk assessment. Completed in {elapsed:.1f} seconds.'
        else:
            message = f'Advanced Analysis Complete: {len(results)} coins analyzed with enhanced features. Consider upgrading to API key for faster analysis. Completed in {elapsed:.1f} seconds.'
        
        if using_fallback:
            message = 'Using fallback data due to CoinGecko API issues. Real-time data will be restored when API is available.'
        
        return jsonify({
            'status': 'success',
            'data': results,
            'message': message
        })
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }) 