from flask import Flask, render_template, jsonify, request
from werkzeug.exceptions import HTTPException
import time
import requests
import os
from dotenv import load_dotenv
from cryptolyzer import analyze as cryptolyzer_analyze, get_top_coins, get_price_history
import pandas as pd

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)

COINGECKO_URL = "https://api.coingecko.com/api/v3"

# Helper: convert numpy/pandas types to JSON-serializable Python primitives
def json_safe(obj):
    try:
        import numpy as _np  # local import to avoid global dependency issues
        numpy_generic = _np.generic
    except Exception:  # numpy may not be available in all contexts
        numpy_generic = tuple()

    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [json_safe(v) for v in obj]
    # Unwrap numpy/pandas scalars
    if hasattr(obj, 'item'):
        try:
            return json_safe(obj.item())
        except Exception:
            pass
    # Explicitly coerce booleans that may be numpy.bool_
    try:
        if isinstance(obj, bool) or type(obj).__name__ == 'bool_':
            return bool(obj)
    except Exception:
        pass
    # Native JSON types
    if obj is None or isinstance(obj, (int, float, str)):
        return obj
    # Try float coercion for numpy numbers
    try:
        return float(obj)
    except Exception:
        return str(obj)

# Helper: stable JSON response that avoids provider type issues
def safe_jsonify(obj):
    from flask import Response
    import json as _json
    try:
        serialized = _json.dumps(json_safe(obj), separators=(",", ":"))
    except Exception:
        # Last-resort stringify
        serialized = _json.dumps(str(obj))
    return Response(serialized + "\n", mimetype="application/json")

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

# Ensure JSON for common gateway/timeouts
@app.errorhandler(502)
def bad_gateway(e):
    return jsonify({
        "status": "error",
        "message": "Bad gateway",
        "details": str(e)
    }), 502

@app.errorhandler(503)
def service_unavailable(e):
    return jsonify({
        "status": "error",
        "message": "Service unavailable",
        "details": str(e)
    }), 503

@app.errorhandler(504)
def gateway_timeout(e):
    return jsonify({
        "status": "error",
        "message": "Gateway timeout",
        "details": str(e)
    }), 504

# Fallback to JSON for any HTTPException not explicitly handled
@app.errorhandler(HTTPException)
def handle_http_exception(e: HTTPException):
    response = e.get_response()
    return jsonify({
        "status": "error",
        "message": e.name,
        "code": e.code,
        "details": e.description,
    }), e.code

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
        
        # Check cache first
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
        result = {"status": "error", "message": f"Analysis failed: {str(e)}"}
        analyze_cache['data'] = result
        analyze_cache['timestamp'] = time.time()
        return jsonify(result), 500

@app.route('/price_history')
def price_history():
    coin_id = request.args.get('coin_id')
    days = request.args.get('days', 7)
    if not coin_id:
        return jsonify({'status': 'error', 'message': 'Missing coin_id'}), 400
    
    url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"
    params = {'vs_currency': 'usd', 'days': days}
    
    try:
        # Add API key headers if available
        headers = {}
        api_key = os.environ.get('COINGECKO_API_KEY', '')
        if api_key:
            headers['X-CG-API-Key'] = api_key
            headers['X-CG-Demo-API-Key'] = api_key
            print(f"[CoinGecko] Using API key for price history: {api_key[:10]}...")
        
        resp = requests.get(url, params=params, headers=headers, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        prices = data.get('prices', [])
        return jsonify({'status': 'success', 'prices': prices})
    except requests.HTTPError as e:
        print(f"[CoinGecko] HTTP error for price history: {resp.status_code} - {resp.text}")
        return jsonify({'status': 'error', 'message': f'CoinGecko API error: {str(e)}'}), resp.status_code
    except Exception as e:
        print(f"[CoinGecko] Error fetching price history: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/detailed_analysis')
def detailed_analysis():
    """Get detailed technical analysis for a specific coin"""
    coin_id = request.args.get('coin_id')
    if not coin_id:
        return jsonify({'status': 'error', 'message': 'Missing coin_id'}), 400
    
    try:
        from cryptolyzer import (
            get_price_history, calculate_rsi, calculate_macd, 
            calculate_bollinger_bands, calculate_moving_averages,
            get_volume_data, analyze_volume, analyze_market_sentiment,
            detect_candlestick_patterns, detect_support_resistance,
            predict_price_ml, calculate_risk_metrics
        )
        
        # Get price data
        price_data = get_price_history(coin_id)['prices']
        df = pd.DataFrame(price_data, columns=['timestamp', 'price'])
        current_price = df['price'].iloc[-1]
        
        # Calculate all technical indicators
        rsi = calculate_rsi(price_data)
        macd_data = calculate_macd(price_data)
        bollinger_data = calculate_bollinger_bands(price_data)
        ma_data = calculate_moving_averages(price_data)
        
        # Volume analysis
        volume_data = get_volume_data(coin_id)
        volume_analysis = analyze_volume(volume_data)
        
        # Sentiment analysis
        sentiment_data = analyze_market_sentiment(coin_id)
        
        # Pattern recognition
        patterns = detect_candlestick_patterns(price_data)
        support_resistance = detect_support_resistance(price_data)
        
        # ML prediction (robust: construct object only if model returns valid result)
        ml_prediction_obj = None
        try:
            ml_prediction, confidence = predict_price_ml(df)
            if ml_prediction is not None:
                predicted_price = float(ml_prediction)
                change_percent = (predicted_price - current_price) / current_price if current_price else 0.0
                ml_prediction_obj = {
                    'predicted_price': predicted_price,
                    'direction': 'up' if change_percent >= 0 else 'down',
                    'change_percent': change_percent * 100.0,
                    'confidence': float(confidence or 0.0)
                }
        except Exception as _:
            ml_prediction_obj = None
        
        # Risk metrics
        risk_metrics = calculate_risk_metrics(price_data, current_price)
        
        # Price change
        price_change = (df['price'].iloc[-1] - df['price'].iloc[0]) / df['price'].iloc[0]
        
        # Volatility
        volatility = df['price'].pct_change().std()
        
        payload = {
            'status': 'success',
            'data': {
                'current_price': current_price,
                'price_change': price_change,
                'volatility': volatility,
                'rsi': rsi,
                'macd': macd_data,
                'bollinger_bands': bollinger_data,
                'moving_averages': ma_data,
                'volume_analysis': volume_analysis,
                'sentiment': sentiment_data,
                'patterns': patterns,
                'support_resistance': support_resistance,
                'ml_prediction': ml_prediction_obj,
                'risk_metrics': risk_metrics
            }
        }
        return safe_jsonify(payload)
        
    except Exception as e:
        import traceback
        print(f"Error in detailed analysis: {str(e)}")
        print(traceback.format_exc())
        return jsonify({'status': 'error', 'message': str(e)}), 500

if __name__ == '__main__':
    # Production settings for Render
    app.config['JSON_AS_ASCII'] = False
    app.config['JSONIFY_PRETTYPRINT_REGULAR'] = False
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
