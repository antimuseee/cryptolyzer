import requests
import pandas as pd
from datetime import datetime, timedelta
import plotly.express as px
from flask import Flask, render_template, jsonify

app = Flask(__name__)

COINGECKO_API_URL = "https://api.coingecko.com/api/v3"

@app.route('/')
def index():
    return render_template('index.html')

def get_top_coins():
    """Get top 100 coins by market cap"""
    url = f"{COINGECKO_API_URL}/coins/markets"
    params = {
        'vs_currency': 'usd',
        'order': 'market_cap_desc',
        'per_page': 100,
        'page': 1,
        'sparkline': False
    }
    response = requests.get(url, params=params)
    return response.json()

def get_price_history(coin_id, days=7):
    """Get price history for a specific coin"""
    url = f"{COINGECKO_API_URL}/coins/{coin_id}/market_chart"
    params = {
        'vs_currency': 'usd',
        'days': days
    }
    response = requests.get(url, params=params)
    return response.json()

def calculate_volatility(prices):
    """Calculate volatility (standard deviation) of price changes"""
    df = pd.DataFrame(prices, columns=['timestamp', 'price'])
    df['price_change'] = df['price'].pct_change()
    return df['price_change'].std()

@app.route('/analyze')
def analyze():
    try:
        # Get top coins
        coins = get_top_coins()
        
        # Analyze each coin
        results = []
        for coin in coins[:50]:  # Analyze top 50 coins for performance
            try:
                # Get price history
                price_data = get_price_history(coin['id'])['prices']
                
                # Calculate volatility
                volatility = calculate_volatility(price_data)
                
                # Calculate price change
                df = pd.DataFrame(price_data, columns=['timestamp', 'price'])
                price_change = (df['price'].iloc[-1] - df['price'].iloc[0]) / df['price'].iloc[0]
                
                results.append({
                    'name': coin['name'],
                    'symbol': coin['symbol'],
                    'volatility': volatility,
                    'price_change': price_change,
                    'current_price': coin['current_price']
                })
                
            except Exception as e:
                print(f"Error analyzing {coin['name']}: {str(e)}")
                continue
        
        # Sort by volatility (ascending) and price change (descending)
        results.sort(key=lambda x: (x['volatility'], -x['price_change']))
        
        return jsonify({
            'status': 'success',
            'data': results[:10]  # Return top 10 coins
        })
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        })

if __name__ == '__main__':
    app.run(debug=True)
