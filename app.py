import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Force CPU usage
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow warnings

from flask import Flask, request, jsonify, render_template, Response, stream_with_context
from stock_prediction import StockPredictor
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import time
from functools import lru_cache
import json
import random
from threading import Lock
import queue
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Initialize predictor with error handling
try:
    predictor = StockPredictor()
    logger.info("StockPredictor initialized successfully")
except Exception as e:
    logger.error(f"Error initializing StockPredictor: {str(e)}")
    predictor = None

# Global rate limiting
request_queue = queue.Queue()
last_request_time = 0
min_request_interval = 2  # Minimum seconds between requests
rate_limit_lock = Lock()

def wait_for_rate_limit():
    global last_request_time
    with rate_limit_lock:
        current_time = time.time()
        time_since_last = current_time - last_request_time
        if time_since_last < min_request_interval:
            sleep_time = min_request_interval - time_since_last
            time.sleep(sleep_time)
        last_request_time = time.time()

# Cache for stock data to reduce API calls
@lru_cache(maxsize=100)
def get_stock_data(symbol, start_date, end_date):
    max_retries = 5
    base_delay = 3  # Increased base delay
    
    for attempt in range(max_retries):
        try:
            # Wait for rate limit
            wait_for_rate_limit()
            
            # Add jitter to avoid synchronized retries
            jitter = random.uniform(0.5, 1.5)
            delay = base_delay * (2 ** attempt) * jitter  # Exponential backoff with jitter
            time.sleep(delay)
            
            # Use Ticker object instead of download for better rate limit handling
            ticker = yf.Ticker(symbol)
            stock_data = ticker.history(start=start_date, end=end_date)
            
            if not stock_data.empty:
                return stock_data
            elif attempt < max_retries - 1:
                continue
            else:
                raise ValueError(f"No data found for {symbol}")
                
        except Exception as e:
            if "rate limit" in str(e).lower() and attempt < max_retries - 1:
                # More aggressive backoff for rate limits
                time.sleep(base_delay * (3 ** attempt) * jitter)
                continue
            else:
                raise e
                
    return pd.DataFrame()

@app.route('/')
def landing():
    return render_template('landing.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'GET':
        return render_template('index.html')
    
    if predictor is None:
        return jsonify({'error': 'Prediction service is not available. Please try again later.'}), 503
        
    try:
        if not request.is_json:
            return jsonify({'error': 'Request must be JSON'}), 400
            
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400
            
        symbol = data.get('symbol', 'AAPL')
        if not isinstance(symbol, str) or not symbol.strip():
            return jsonify({'error': 'Invalid symbol format'}), 400
            
        try:
            days = int(data.get('days', 7))
            if days <= 0 or days > 30:
                return jsonify({'error': 'Days must be between 1 and 30'}), 400
        except (ValueError, TypeError):
            return jsonify({'error': 'Invalid days format'}), 400
        
        # Get historical data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=90)
        
        print("ABOUT TO FETCH STOCK DATA...")  # Debug log
        try:
            stock_data = get_stock_data(symbol, start_date, end_date)
            print("STOCK DATA FETCHED!")  # Debug log
        except Exception as e:
            error_msg = str(e)
            if "delisted" in error_msg.lower():
                return jsonify({'error': f'Stock {symbol} might be delisted or not available on Yahoo Finance. Please try a different symbol.'}), 400
            elif "no data found" in error_msg.lower():
                return jsonify({'error': f'No data found for {symbol}. Please check if the symbol is correct.'}), 400
            elif "rate limit" in error_msg.lower():
                return jsonify({'error': 'Yahoo Finance rate limit reached. Please try again in a few minutes.'}), 429
            else:
                return jsonify({'error': f'Error fetching data for {symbol}: {error_msg}'}), 400
        
        if stock_data.empty:
            return jsonify({'error': f'No data found for {symbol}. Please check if the symbol is correct.'}), 400
            
        # Prepare data for prediction
        df = pd.DataFrame(stock_data)
        df = df[['Close']]
        df.columns = ['Close']
        
        def generate():
            try:
                # Create a function to log training progress
                def log_training_progress(message):
                    print(f"TRAINING LOG: {message}")  # Debug log
                    return message

                print("STARTING TRAINING...")  # Debug log
                for message in predictor.train(df, epochs=50, batch_size=32, log_function=log_training_progress):
                    print(f"YIELDING TRAINING MSG: {message}")  # Debug log
                    yield json.dumps({'status': 'training', 'message': message}) + '\n'

                print("STARTING PREDICTION...")  # Debug log
                prediction = predictor.predict(df, days)
                current_price = df['Close'].iloc[-1]
                print("YIELDING FINAL RESULT...")  # Debug log
                yield json.dumps({
                    'symbol': symbol,
                    'prediction': prediction.tolist(),
                    'last_price': current_price,
                    'training_data': df['Close'].tolist(),
                    'training_dates': df.index.strftime('%Y-%m-%d').tolist()
                }) + '\n'
            except Exception as e:
                print(f"ERROR: {e}")  # Debug log
                yield json.dumps({'error': str(e)}) + '\n'
        
        return Response(stream_with_context(generate()), mimetype='application/json')
        
    except Exception as e:
        logger.error(f"Error in predict endpoint: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'healthy',
        'predictor_initialized': predictor is not None
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True) 