from flask import Flask, render_template, jsonify, request
import time
import requests
from cryptolyzer import analyze as cryptolyzer_analyze, get_top_coins, get_price_history
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = Flask(__name__)

COINGECKO_URL = "https://api.coingecko.com/api/v3"

# Global error handler to ensure JSON responses
@app.errorhandler(Exception)
def handle_exception(e):
    import traceback
    print(f"Global error handler caught: {str(e)}")
    print(traceback.format_exc())
    return jsonify({
        "status": "error",
        "message": "Internal server error",
        "details": str(e)
    }), 500

@app.errorhandler(ImportError)
def handle_import_error(e):
    import traceback
    print(f"Import error caught: {str(e)}")
    print(traceback.format_exc())
    return jsonify({
        "status": "error",
        "message": "Module import error",
        "details": str(e)
    }), 500

@app.errorhandler(404)
def not_found(e):
    return jsonify({
        "status": "error",
        "message": "Endpoint not found"
    }), 404

@app.errorhandler(500)
def internal_error(e):
    return jsonify({
        "status": "error",
        "message": "Internal server error"
    }), 500

# Simple in-memory cache for analyze endpoint
analyze_cache = {
    'data': None,
    'timestamp': 0,
    'params': None
}
CACHE_DURATION = 900  # seconds (15 minutes)

@app.route('/')
def index():
    coins = get_top_coins()
    return render_template('index.html', coins=coins)

@app.route('/health')
def health():
    """Health check endpoint for debugging"""
    return jsonify({
        "status": "healthy",
        "message": "Cryptolyzer is running",
        "timestamp": time.time()
    })

@app.route('/test')
def test():
    """Simple test endpoint"""
    try:
        return jsonify({
            "status": "success",
            "message": "Test endpoint working",
            "data": {"test": "value"}
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

@app.route('/analyze')
def analyze():
    try:
        now = time.time()
        max_swing = request.args.get('max_swing')
        fluct_threshold = request.args.get('fluct_threshold')
        coin_type = request.args.get('coin_type', 'all')
        max_swing = float(max_swing) if max_swing is not None else None
        fluct_threshold = float(fluct_threshold) if fluct_threshold is not None else None
        cache_key = f"{max_swing}-{fluct_threshold}-{coin_type}"
        if analyze_cache.get('data') and analyze_cache.get('params') == cache_key and now - analyze_cache['timestamp'] < CACHE_DURATION:
            return jsonify(analyze_cache['data'])
        # Use enhanced analysis from cryptolyzer
        response = cryptolyzer_analyze()
        analyze_cache['data'] = response.get_json()
        analyze_cache['timestamp'] = now
        analyze_cache['params'] = cache_key
        return response
    except Exception as e:
        import traceback
        print(f"Error in analyze endpoint: {str(e)}")
        print(traceback.format_exc())
        result = {"status": "error", "message": str(e)}
        analyze_cache['data'] = result
        analyze_cache['timestamp'] = time.time()
        return jsonify(result), 500

@app.route('/price_history')
def price_history():
    coin_id = request.args.get('coin_id')
    days = request.args.get('days', 7)
    if not coin_id:
        return jsonify({'status': 'error', 'message': 'Missing coin_id'}), 400
    max_retries = 5
    backoff_times = [10, 30, 60, 120, 240]  # seconds
    url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"
    params = {'vs_currency': 'usd', 'days': days}
    for attempt in range(max_retries):
        try:
            resp = requests.get(url, params=params, timeout=10)
            if resp.status_code == 429:
                wait_time = backoff_times[min(attempt, len(backoff_times)-1)]
                time.sleep(wait_time)
                continue
            resp.raise_for_status()
            data = resp.json()
            prices = data.get('prices', [])
            return jsonify({'status': 'success', 'prices': prices})
        except requests.HTTPError as e:
            if resp.status_code == 429:
                wait_time = backoff_times[min(attempt, len(backoff_times)-1)]
                time.sleep(wait_time)
                continue
            return jsonify({'status': 'error', 'message': f'CoinGecko API error: {str(e)}'}), resp.status_code
        except Exception as e:
            return jsonify({'status': 'error', 'message': str(e)}), 500
    return jsonify({'status': 'error', 'message': 'CoinGecko API rate limit reached after multiple retries. Please try again later.'}), 429

if __name__ == '__main__':
    # Production settings for Render
    app.config['JSON_AS_ASCII'] = False
    app.config['JSONIFY_PRETTYPRINT_REGULAR'] = False
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
