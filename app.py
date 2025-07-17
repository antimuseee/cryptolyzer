from flask import Flask, render_template
import requests
import pandas as pd
from datetime import datetime, timedelta
import plotly.graph_objects as go
from dotenv import load_dotenv
import os

app = Flask(__name__)

# CoinGecko API endpoints
COINGECKO_URL = "https://api.coingecko.com/api/v3"

from flask import Flask, render_template, jsonify
# ... rest of imports ...

@app.route('/')
def index():
    coins = get_top_coins()
    fluctuation_data = analyze_fluctuations(coins)
    return render_template('index.html', coins=fluctuation_data)

import time

# Simple in-memory cache for analyze endpoint
analyze_cache = {
    'data': None,
    'timestamp': 0
}
CACHE_DURATION = 900  # seconds (15 minutes)

from flask import request

@app.route('/analyze')
def analyze():
    try:
        now = time.time()
        # Get thresholds from query params (with defaults)
        max_swing = float(request.args.get('max_swing', 20))
        fluct_threshold = float(request.args.get('fluct_threshold', 2))
        # Return cached data if it's still valid and params haven't changed
        cache_key = f"{max_swing}-{fluct_threshold}"
        if analyze_cache.get('data') and analyze_cache.get('params') == cache_key and now - analyze_cache['timestamp'] < CACHE_DURATION:
            print("Serving /analyze from cache.")
            return jsonify(analyze_cache['data'])
        coins = get_top_coins()
        if not coins:
            result = {"status": "error", "message": "No coins returned from CoinGecko. You may be rate limited or the API is down."}
            analyze_cache['data'] = result
            analyze_cache['timestamp'] = now
            analyze_cache['params'] = cache_key
            return jsonify(result)
        fluctuation_data = analyze_fluctuations(coins, max_swing=max_swing, fluct_threshold=fluct_threshold)
        result = {"status": "success", "data": fluctuation_data}
        analyze_cache['data'] = result
        analyze_cache['timestamp'] = now
        analyze_cache['params'] = cache_key
        return jsonify(result)
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        result = {"status": "error", "message": str(e)}
        analyze_cache['data'] = result
        analyze_cache['timestamp'] = time.time()
        return jsonify(result)

def analyze_fluctuations(coins, max_swing=20, fluct_threshold=2):
    fluctuation_data = []
    if not isinstance(coins, list):
        print("analyze_fluctuations: coins is not a list!", type(coins))
        return []
    print(f"Fetched {len(coins)} coins from API.")
    filtered = 0
    skipped = 0
    # WARNING: This will take 50+ seconds for 100 coins due to API rate limits
    for coin in coins:
        try:
            price_data = get_historical_prices(coin['id'])
            if len(price_data) < 2:
                skipped += 1
                continue
            prices = [p['price'] for p in price_data]
            min_price = min(prices)
            max_price = max(prices)
            swing = (max_price - min_price) / min_price * 100
            if swing > max_swing:
                skipped += 1
                continue
            fluct_count = 0
            for i in range(1, len(prices)):
                pct_change = abs(prices[i] - prices[i-1]) / prices[i-1] * 100
                if pct_change >= fluct_threshold:
                    fluct_count += 1
            fluctuation_data.append({
                'name': coin['name'],
                'symbol': coin['symbol'],
                'current_price': coin['current_price'],
                'fluctuations': fluct_count,
                'max_swing': max_swing,
                'high': max_price,
                'low': min_price,
                'market_cap': coin['market_cap']
            })
            time.sleep(0)  # Reduced delay to minimum (0 seconds)
        except Exception as e:
            print(f"Error processing {coin['name']}: {str(e)}")
            continue
    fluctuation_data.sort(key=lambda x: x['fluctuations'], reverse=True)
    print(f"Filtered {len(fluctuation_data)} coins, skipped {skipped}.")
    return fluctuation_data

def get_top_coins():
    # Fetch top 100 coins by market cap from CoinGecko
    try:
        url = f"{COINGECKO_URL}/coins/markets"
        params = {
            'vs_currency': 'usd',
            'order': 'market_cap_desc',
            'per_page': 100,
            'page': 1,
            'price_change_percentage': '24h'
        }
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        print(f"Error fetching top coins: {e}")
        return []

def get_historical_prices(coin_id):
    # Fetch 7 days of daily price data from CoinGecko
    try:
        url = f"{COINGECKO_URL}/coins/{coin_id}/market_chart"
        params = {
            'vs_currency': 'usd',
            'days': 7,
            'interval': 'daily'
        }
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        if 'prices' not in data or not data['prices']:
            print(f"No prices found for {coin_id}")
            return []
        # CoinGecko returns [[timestamp, price], ...]
        prices = []
        for entry in data['prices']:
            prices.append({'date': entry[0], 'price': entry[1]})
        return prices
    except Exception as e:
        print(f"Error fetching historical prices for {coin_id}: {e}")
        return []

def calculate_volatility(price_data):
    volatility_data = []
    
    for coin in price_data:
        prices = coin['prices']
        if len(prices) < 2:  # Need at least 2 data points
            continue
            
        # Calculate daily returns
        returns = np.diff(prices) / prices[:-1]
        
        # Calculate volatility metrics
        volatility = np.std(returns) * np.sqrt(252)  # Annualized volatility
        avg_return = np.mean(returns)
        
        volatility_data.append({
            'name': coin['name'],
            'volatility': volatility,
            'avg_return': avg_return,
            'current_price': prices[-1],
            'price_change_24h': (prices[-1] - prices[0]) / prices[0] * 100
        })
    
    # Sort by volatility (ascending)
    volatility_data.sort(key=lambda x: x['volatility'])
    
    return volatility_data

if __name__ == '__main__':
    app.run(debug=True)
