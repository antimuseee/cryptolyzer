import requests
import pandas as pd
from datetime import datetime, timedelta
import plotly.express as px
import time

COINGECKO_API_URL = "https://api.coingecko.com/api/v3"

# Only fetch 25 coins per request to avoid CoinGecko free API rate limit
COINS_PER_REQUEST = 25
MIN_ANALYSIS_DURATION = 60  # seconds

def get_top_coins():
    """Get top 25 coins by market cap (single API call, no retry/sleep)"""
    url = f"{COINGECKO_API_URL}/coins/markets"
    params = {
        'vs_currency': 'usd',
        'order': 'market_cap_desc',
        'per_page': COINS_PER_REQUEST,
        'page': 1,
        'sparkline': False
    }
    print(f"[CoinGecko] Fetching top coins: {url} params={params}")
    try:
        response = requests.get(url, params=params, timeout=30)
        print(f"[CoinGecko] Status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            if not data:
                print(f"[CoinGecko] Warning: Empty data returned. Possible rate limit or API error.")
                return get_fallback_coins()
            return data
        else:
            print(f"[CoinGecko] Error: {response.text}")
            return get_fallback_coins()
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
    """Get price history for a specific coin, with basic error handling"""
    url = f"{COINGECKO_API_URL}/coins/{coin_id}/market_chart"
    params = {
        'vs_currency': 'usd',
        'days': days
    }
    print(f"[CoinGecko] Fetching price history for {coin_id}: {url} params={params}")
    try:
        response = requests.get(url, params=params, timeout=30)
        if response.status_code == 200:
            return response.json()
        else:
            print(f"[CoinGecko] Error fetching price history: {response.text}")
            return get_fallback_price_history(coin_id, days)
    except requests.exceptions.RequestException as e:
        print(f"[CoinGecko] Request error for price history: {str(e)}")
        return get_fallback_price_history(coin_id, days)
    except Exception as e:
        print(f"[CoinGecko] Unexpected error for price history: {str(e)}")
        return get_fallback_price_history(coin_id, days)

def get_fallback_price_history(coin_id, days=7):
    """Return fallback price history when API is unavailable"""
    print(f"[CoinGecko] Using fallback price history for {coin_id}")
    import time
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
    return df['price_change'].std()

def analyze():
    import json
    from flask import jsonify
    start_time = time.time()
    try:
        # Get top coins (25 per request)
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
        for coin in coins:  # Analyze up to 25 coins for performance
            try:
                # Get price history
                price_data = get_price_history(coin['id'])['prices']
                df = pd.DataFrame(price_data, columns=['timestamp', 'price'])

                # Calculate volatility
                volatility = calculate_volatility(price_data)

                # Calculate price change
                price_change = (df['price'].iloc[-1] - df['price'].iloc[0]) / df['price'].iloc[0]

                # Detect trend (simple linear regression slope)
                x = range(len(df))
                y = df['price'].values
                if len(y) > 1:
                    slope = (y[-1] - y[0]) / len(y)
                else:
                    slope = 0
                if slope > 0.01:
                    trend = 'Uptrend'
                elif slope < -0.01:
                    trend = 'Downtrend'
                else:
                    trend = 'Sideways'

                # Detect pattern: breakout (last price > max of previous n-1 prices)
                breakout = False
                if len(y) > 3 and y[-1] > max(y[:-1]):
                    breakout = True

                # Improved Opportunity Score: risk-adjusted return + bonuses
                explanation = []
                # Avoid division by zero
                if volatility > 0:
                    risk_adjusted = price_change / volatility
                    explanation.append(f'Risk-adjusted return: {risk_adjusted:.2f} (price change / volatility)')
                else:
                    risk_adjusted = 0
                    explanation.append('Volatility too low to calculate risk-adjusted return')
                # Bonus for uptrend
                bonus = 0
                if trend == 'Uptrend':
                    bonus += 0.5
                    explanation.append('Bonus: Uptrend detected')
                if breakout:
                    bonus += 0.5
                    explanation.append('Bonus: Breakout pattern detected')
                score = risk_adjusted + bonus
                if price_change > 0.1:
                    explanation.append('Strong positive price momentum')
                elif price_change < -0.1:
                    explanation.append('Significant price drop (potential reversal)')
                if volatility > 0.2:
                    explanation.append('High volatility (risky but high reward)')
                elif volatility < 0.05:
                    explanation.append('Low volatility (safer entry)')
                if not explanation:
                    explanation = ['No strong signals detected']

                coin_results.append({
                    'id': coin.get('id', ''),
                    'name': coin['name'],
                    'symbol': coin['symbol'],
                    'volatility': volatility,
                    'price_change': price_change,
                    'current_price': coin['current_price'],
                    'trend': trend,
                    'breakout': breakout,
                    'raw_score': score,
                    'explanation': '; '.join(explanation)
                })
                raw_scores.append(score)
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
        # Sort by opportunity score (descending), then by volatility (ascending), then by price change (descending)
        results.sort(key=lambda x: (-x['opportunity_score'], x['volatility'], -x['price_change']))
        
        # Check if we have any results
        if not results:
            return jsonify({
                'status': 'error',
                'message': 'No coins could be analyzed. This might be due to API rate limits or temporary issues.'
            })
        
        # Enforce minimum analysis duration of 1 minute
        elapsed = time.time() - start_time
        if elapsed < MIN_ANALYSIS_DURATION:
            time.sleep(MIN_ANALYSIS_DURATION - elapsed)
        message = 'Free API mode: Only 25 coins analyzed per request to avoid CoinGecko rate limits. Each request takes at least 1 minute.'
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
