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
CACHE_DURATION = 300  # seconds (5 minutes)

@app.route('/analyze')
def analyze():
    try:
        now = time.time()
        # Return cached data if it's still valid
        if analyze_cache['data'] is not None and now - analyze_cache['timestamp'] < CACHE_DURATION:
            print("Serving /analyze from cache.")
            return jsonify(analyze_cache['data'])
        coins = get_top_coins()
        if not coins:
            result = {"status": "error", "message": "No coins returned from CoinGecko. You may be rate limited or the API is down."}
            analyze_cache['data'] = result
            analyze_cache['timestamp'] = now
            return jsonify(result)
        fluctuation_data = analyze_fluctuations(coins)
        result = {"status": "success", "data": fluctuation_data}
        analyze_cache['data'] = result
        analyze_cache['timestamp'] = now
        return jsonify(result)
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        result = {"status": "error", "message": str(e)}
        analyze_cache['data'] = result
        analyze_cache['timestamp'] = time.time()
        return jsonify(result)

def analyze_fluctuations(coins):
    fluctuation_data = []
    if not isinstance(coins, list):
        print("analyze_fluctuations: coins is not a list!", type(coins))
        return []
    print(f"Fetched {len(coins)} coins from API.")
    filtered = 0
    skipped = 0
    for coin in coins[:5]:
        try:
            price_data = get_historical_prices(coin['id'])
            if len(price_data) < 2:
                skipped += 1
                continue
            prices = [p['price'] for p in price_data]
            min_price = min(prices)
            max_price = max(prices)
            max_swing = (max_price - min_price) / min_price * 100
            if max_swing > 20:
                skipped += 1
                continue
            fluct_count = 0
            for i in range(1, len(prices)):
                pct_change = abs(prices[i] - prices[i-1]) / prices[i-1] * 100
                if pct_change >= 2:
                    fluct_count += 1
            fluctuation_data.append({
                'name': coin['name'],
                'symbol': coin['symbol'],
                'current_price': coin['current_price'],
                'fluctuations': fluct_count,
                'max_swing': max_swing,
                'market_cap': coin['market_cap']
            })
        except Exception as e:
            print(f"Error processing {coin['name']}: {str(e)}")
            continue
    fluctuation_data.sort(key=lambda x: x['fluctuations'], reverse=True)
    print(f"Filtered {len(fluctuation_data)} coins, skipped {skipped}.")
    return fluctuation_data

def get_top_coins():
    # Hardcoded sample data for testing (from user)
    return [
        {"id":"bitcoin","symbol":"btc","name":"Bitcoin","image":"https://coin-images.coingecko.com/coins/images/1/large/bitcoin.png?1696501400","current_price":117678,"market_cap":2341160774191,"market_cap_rank":1,"fully_diluted_valuation":2341163363327,"total_volume":78650357399,"high_24h":119342,"low_24h":116094,"price_change_24h":-1664.4037566889747,"price_change_percentage_24h":-1.39465,"market_cap_change_24h":-33279398950.836426,"market_cap_change_percentage_24h":-1.40157,"circulating_supply":19892943.0,"total_supply":19892965.0,"max_supply":21000000.0,"ath":122838,"ath_change_percentage":-4.1597,"ath_date":"2025-07-14T07:56:01.937Z","atl":67.81,"atl_change_percentage":173517.50864,"atl_date":"2013-07-06T00:00:00.000Z","roi":None,"last_updated":"2025-07-16T01:46:45.649Z"},
        {"id":"ethereum","symbol":"eth","name":"Ethereum","image":"https://coin-images.coingecko.com/coins/images/279/large/ethereum.png?1696501628","current_price":3138.31,"market_cap":378909719781,"market_cap_rank":2,"fully_diluted_valuation":378909719781,"total_volume":44030141861,"high_24h":3143.77,"low_24h":2942.21,"price_change_24h":146.07,"price_change_percentage_24h":4.88177,"market_cap_change_24h":18428576499,"market_cap_change_percentage_24h":5.11222,"circulating_supply":120714500.839526,"total_supply":120714500.839526,"max_supply":None,"ath":4878.26,"ath_change_percentage":-35.58986,"ath_date":"2021-11-10T14:24:19.604Z","atl":0.432979,"atl_change_percentage":725592.38431,"atl_date":"2015-10-20T00:00:00.000Z","roi":{"times":34.65802710173756,"currency":"btc","percentage":3465.8027101737557},"last_updated":"2025-07-16T01:46:46.881Z"},
        {"id":"ripple","symbol":"xrp","name":"XRP","image":"https://coin-images.coingecko.com/coins/images/44/large/xrp-symbol-white-128.png?1696501442","current_price":2.92,"market_cap":172754831098,"market_cap_rank":3,"fully_diluted_valuation":292111964575,"total_volume":7178124646,"high_24h":2.94,"low_24h":2.82,"price_change_24h":0.0091558,"price_change_percentage_24h":0.31441,"market_cap_change_24h":1144721000,"market_cap_change_percentage_24h":0.66705,"circulating_supply":59131625363.0,"total_supply":99985946231.0,"max_supply":100000000000.0,"ath":3.4,"ath_change_percentage":-14.00213,"ath_date":"2018-01-07T00:00:00.000Z","atl":0.00268621,"atl_change_percentage":108700.0652,"atl_date":"2014-05-22T00:00:00.000Z","roi":None,"last_updated":"2025-07-16T01:46:49.847Z"},
        {"id":"tether","symbol":"usdt","name":"Tether","image":"https://coin-images.coingecko.com/coins/images/325/large/Tether.png?1696501661","current_price":1.0,"market_cap":159892063000,"market_cap_rank":4,"fully_diluted_valuation":159892063000,"total_volume":148272960346,"high_24h":1.0,"low_24h":0.999659,"price_change_24h":-0.000121094688638745,"price_change_percentage_24h":-0.01211,"market_cap_change_24h":341006474,"market_cap_change_percentage_24h":0.21373,"circulating_supply":159886922749.0362,"total_supply":159886922749.0362,"max_supply":None,"ath":1.32,"ath_change_percentage":-24.41618,"ath_date":"2018-07-24T00:00:00.000Z","atl":0.572521,"atl_change_percentage":74.6741,"atl_date":"2015-03-02T00:00:00.000Z","roi":None,"last_updated":"2025-07-16T01:46:51.192Z"},
        {"id":"binancecoin","symbol":"bnb","name":"BNB","image":"https://coin-images.coingecko.com/coins/images/825/large/bnb-icon2_2x.png?1696501970","current_price":690.77,"market_cap":100789238909,"market_cap_rank":5,"fully_diluted_valuation":100789238909,"total_volume":1472045603,"high_24h":693.57,"low_24h":676.25,"price_change_24h":3.28,"price_change_percentage_24h":0.477,"market_cap_change_24h":564373854,"market_cap_change_percentage_24h":0.56311,"circulating_supply":145887575.79,"total_supply":145887575.79,"max_supply":200000000.0,"ath":788.84,"ath_change_percentage":-12.38043,"ath_date":"2024-12-04T10:35:25.220Z","atl":0.0398177,"atl_change_percentage":1735764.87717,"atl_date":"2017-10-19T00:00:00.000Z","roi":None,"last_updated":"2025-07-16T01:46:52.673Z"}
    ]

def get_historical_prices(coin_id):
    # Mock price data for 7 days for each coin
    import random
    from datetime import datetime, timedelta
    base_prices = {
        'bitcoin': 117678,
        'ethereum': 3138.31,
        'ripple': 2.92,
        'tether': 1.0,
        'binancecoin': 690.77
    }
    base = base_prices.get(coin_id, 100)
    today = datetime.now()
    prices = []
    # Simulate 7 days of price data with random daily swings within ~2-5%
    for i in range(7, 0, -1):
        date = today - timedelta(days=i)
        # Swing price up/down by up to 5% randomly
        swing = random.uniform(-0.05, 0.05)
        price = base * (1 + swing)
        prices.append({'date': date, 'price': price})
        base = price  # next day starts from today's price
    return prices

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
