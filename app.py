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
    print(">>> /analyze route called")
    try:
        now = time.time()
        # Get thresholds from query params (None if not provided)
        max_swing = request.args.get('max_swing')
        fluct_threshold = request.args.get('fluct_threshold')
        coin_type = request.args.get('coin_type', 'all')
        # Convert to float if provided
        max_swing = float(max_swing) if max_swing is not None else None
        fluct_threshold = float(fluct_threshold) if fluct_threshold is not None else None
        # Return cached data if it's still valid and params haven't changed
        cache_key = f"{max_swing}-{fluct_threshold}-{coin_type}"
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

        # Use enhanced analysis from Cryptolyzer
        from Cryptolyzer import analyze as enhanced_analyze
        with app.app_context():
            response = enhanced_analyze()
        return response
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        result = {"status": "error", "message": str(e)}
        analyze_cache['data'] = result
        analyze_cache['timestamp'] = time.time()
        return jsonify(result)

progress_cache = {'messages': []}

def analyze_fluctuations(coins, max_swing=None, fluct_threshold=None):
    import math
    fluctuation_data = []
    if not isinstance(coins, list):
        print("analyze_fluctuations: coins is not a list!", type(coins))
        return []
    progress_cache['messages'] = []
    progress_cache['messages'].append(f"Fetched {len(coins)} coins from API.")
    filtered = 0
    skipped = 0
    RATE_LIMIT = 29  # CoinGecko free tier: 30/min, use 29 to be safe
    total = len(coins)
    num_batches = math.ceil(total / RATE_LIMIT)
    for batch_idx in range(num_batches):
        start = batch_idx * RATE_LIMIT
        end = min((batch_idx + 1) * RATE_LIMIT, total)
        batch = coins[start:end]
        msg = f"Processing batch {batch_idx+1}/{num_batches}: coins {start+1}-{end} of {total}"
        print(msg)
        progress_cache['messages'].append(msg)
        for coin in batch:
            try:
                price_data = get_historical_prices(coin['id'])
                if len(price_data) < 2:
                    skipped += 1
                    continue
                prices = [p['price'] for p in price_data]
                min_price = min(prices)
                max_price = max(prices)
                swing = (max_price - min_price) / min_price * 100
                if max_swing is not None and swing > max_swing:
                    skipped += 1
                    continue
                fluct_count = 0
                for i in range(1, len(prices)):
                    pct_change = abs(prices[i] - prices[i-1]) / prices[i-1] * 100
                    if fluct_threshold is not None:
                        if pct_change >= fluct_threshold:
                            fluct_count += 1
                    else:
                        if pct_change:
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
            except Exception as e:
                print(f"Error processing {coin['name']}: {str(e)}")
                progress_cache['messages'].append(f"Error processing {coin['name']}: {str(e)}")
                continue
        if batch_idx < num_batches - 1:
            msg = f"Sleeping for 1 second to respect CoinGecko rate limits..."
            print(msg)
            progress_cache['messages'].append(msg)
            time.sleep(1)
    fluctuation_data.sort(key=lambda x: x['fluctuations'], reverse=True)
    msg = f"Filtered {len(fluctuation_data)} coins, skipped {skipped}."
    print(msg)
    progress_cache['messages'].append(msg)
    return fluctuation_data

from flask import jsonify

@app.route('/analyze_progress')
def analyze_progress():
    return jsonify({'messages': progress_cache.get('messages', [])})


# In-memory cache for CoinGecko results
coingecko_cache = {
    'data': None,
    'timestamp': 0
}
COINGECKO_CACHE_DURATION = 300  # 5 minutes

def get_top_coins():
    # Fetch top 100 coins by market cap from CoinGecko, with caching and robust error handling
    global coingecko_cache
    try:
        now = time.time()
        # Serve from cache if recent
        if coingecko_cache['data'] and now - coingecko_cache['timestamp'] < COINGECKO_CACHE_DURATION:
            print("[CoinGecko] Serving top coins from cache.")
            return coingecko_cache['data']
        url = f"{COINGECKO_URL}/coins/markets"
        params = {
            'vs_currency': 'usd',
            'order': 'market_cap_desc',
            'per_page': 100,
            'page': 1,
            'price_change_percentage': '24h'
        }
        print(f"[CoinGecko] Fetching top coins: {url} params={params}")
        resp = requests.get(url, params=params, timeout=10)
        print(f"[CoinGecko] Status: {resp.status_code}")
        if resp.status_code == 429:
            print(f"[CoinGecko] Rate limited! Serving cached data if available.")
            if coingecko_cache['data']:
                return coingecko_cache['data']
            return []
        if resp.status_code != 200:
            print(f"[CoinGecko] Error: {resp.text}")
            if coingecko_cache['data']:
                return coingecko_cache['data']
            return []
        data = resp.json()
        if not data:
            print(f"[CoinGecko] Warning: Empty data returned. Possible rate limit or API error.")
            if coingecko_cache['data']:
                return coingecko_cache['data']
            return []
        # Update cache
        coingecko_cache['data'] = data
        coingecko_cache['timestamp'] = now
        return data
    except Exception as e:
        print(f"Error fetching top coins: {e}")
        if coingecko_cache['data']:
            print("[CoinGecko] Exception occurred, serving cached data.")
            return coingecko_cache['data']
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

# Placeholder for fetching moonshot coins from the Moonshot Crypto app
# Replace this with your actual API call or data source

def fetch_moonshot_coins():
    """
    Fetch moonshot coins using the Dexscreener public API.
    We define 'moonshot' as new/low market cap coins (<$1M) on popular chains.
    """
    import requests
    import time
    chains = ['solana', 'bsc', 'eth', 'base']
    moonshots = []
    for chain in chains:
        url = f'https://api.dexscreener.com/latest/dex/pairs/{chain}'
        try:
            resp = requests.get(url, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            pairs = data.get('pairs', [])
            for p in pairs:
                # Filter for low market cap (< $1M) and recent pairs (created in last ~7 days)
                market_cap = p.get('fdv', 0) or 0
                if market_cap is None or market_cap > 1_000_000:
                    continue
                # Dexscreener provides 'pairCreatedAt' as a timestamp (ms)
                created_ts = p.get('pairCreatedAt', 0)
                if created_ts:
                    age_days = (time.time() - int(created_ts)/1000) / 86400
                    if age_days > 7:
                        continue
                moonshots.append({
                    'name': p.get('baseToken', {}).get('name', ''),
                    'symbol': p.get('baseToken', {}).get('symbol', ''),
                    'current_price': float(p.get('priceUsd', 0) or 0),
                    'fluctuations': 0,  # Not available
                    'max_swing': 0,     # Not available
                    'high': 0,          # Not available
                    'low': 0,           # Not available
                    'market_cap': market_cap,
                })
        except Exception as e:
            print(f'Error fetching Dexscreener {chain}: {e}')
            continue
    # Sort by most recent, then by lowest market cap
    moonshots.sort(key=lambda x: (x['market_cap'],), reverse=False)
    # Limit to top 50
    return moonshots[:50]


@app.route('/price_history')
def price_history():
    coin_id = request.args.get('coin_id')
    days = request.args.get('days', 7)
    if not coin_id:
        return jsonify({'status': 'error', 'message': 'Missing coin_id'}), 400
    import time
    max_retries = 5
    backoff_times = [10, 30, 60, 120, 240]  # seconds
    url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"
    params = {'vs_currency': 'usd', 'days': days}
    for attempt in range(max_retries):
        try:
            resp = requests.get(url, params=params, timeout=10)
            if resp.status_code == 429:
                wait_time = backoff_times[min(attempt, len(backoff_times)-1)]
                print(f"[ERROR] CoinGecko rate limit hit for coin_id={coin_id}, days={days}, attempt {attempt+1}/{max_retries}. Sleeping {wait_time}s.")
                time.sleep(wait_time)
                continue
            resp.raise_for_status()
            data = resp.json()
            prices = data.get('prices', [])
            return jsonify({'status': 'success', 'prices': prices})
        except requests.HTTPError as e:
            if resp.status_code == 429:
                wait_time = backoff_times[min(attempt, len(backoff_times)-1)]
                print(f"[ERROR] CoinGecko HTTPError 429 for coin_id={coin_id}, days={days}, attempt {attempt+1}/{max_retries}. Sleeping {wait_time}s.")
                time.sleep(wait_time)
                continue
            print(f"[ERROR] CoinGecko API error for coin_id={coin_id}, days={days}: {e}")
            return jsonify({'status': 'error', 'message': f'CoinGecko API error: {str(e)}'}), resp.status_code
        except Exception as e:
            print(f"[ERROR] Backend exception in /price_history for coin_id={coin_id}, days={days}: {e}")
            return jsonify({'status': 'error', 'message': str(e)}), 500
    return jsonify({'status': 'error', 'message': 'CoinGecko API rate limit reached after multiple retries. Please try again later.'}), 429

    coin_id = request.args.get('coin_id')
    days = request.args.get('days', 7)
    if not coin_id:
        return jsonify({'status': 'error', 'message': 'Missing coin_id'}), 400
    import time
    max_retries = 5
    backoff_times = [10, 30, 60, 120, 240]  # seconds
    url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"
    params = {'vs_currency': 'usd', 'days': days}
    for attempt in range(max_retries):
        try:
            resp = requests.get(url, params=params, timeout=10)
            if resp.status_code == 429:
                wait_time = backoff_times[min(attempt, len(backoff_times)-1)]
                print(f"[ERROR] CoinGecko rate limit hit for coin_id={coin_id}, days={days}, attempt {attempt+1}/{max_retries}. Sleeping {wait_time}s.")
                time.sleep(wait_time)
                continue
            resp.raise_for_status()
            data = resp.json()
            prices = data.get('prices', [])
            return jsonify({'status': 'success', 'prices': prices})
        except requests.HTTPError as e:
            if resp.status_code == 429:
                wait_time = backoff_times[min(attempt, len(backoff_times)-1)]
                print(f"[ERROR] CoinGecko HTTPError 429 for coin_id={coin_id}, days={days}, attempt {attempt+1}/{max_retries}. Sleeping {wait_time}s.")
                time.sleep(wait_time)
                continue
            print(f"[ERROR] CoinGecko API error for coin_id={coin_id}, days={days}: {e}")
            return jsonify({'status': 'error', 'message': f'CoinGecko API error: {str(e)}'}), resp.status_code
        except Exception as e:
            print(f"[ERROR] Backend exception in /price_history for coin_id={coin_id}, days={days}: {e}")
            return jsonify({'status': 'error', 'message': str(e)}), 500
    return jsonify({'status': 'error', 'message': 'CoinGecko API rate limit reached after multiple retries. Please try again later.'}), 429

if __name__ == '__main__':
    app.run(debug=True)
