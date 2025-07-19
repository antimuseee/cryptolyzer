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
    print(f"[CoinGecko] Fetching top coins: {url} params={params}")
    response = requests.get(url, params=params)
    print(f"[CoinGecko] Status: {response.status_code}")
    if response.status_code != 200:
        print(f"[CoinGecko] Error: {response.text}")
        return []
    data = response.json()
    if not data:
        print(f"[CoinGecko] Warning: Empty data returned. Possible rate limit or API error.")
    return data

def get_price_history(coin_id, days=7):
    """Get price history for a specific coin"""
    url = f"{COINGECKO_API_URL}/coins/{coin_id}/market_chart"
    params = {
        'vs_currency': 'usd',
        'days': days
    }
    print(f"[CoinGecko] Fetching price history for {coin_id}: {url} params={params}")
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
        results = []
        raw_scores = []
        coin_results = []
        for coin in coins[:50]:  # Analyze top 50 coins for performance
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
        return jsonify({
            'status': 'success',
            'data': results
        })
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        })

if __name__ == '__main__':
    app.run(debug=True)
