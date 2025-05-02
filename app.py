from flask import Flask, request, jsonify, render_template, Response, stream_with_context
from stock_prediction import StockPredictor
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import time
from functools import lru_cache
import json

app = Flask(__name__)
predictor = StockPredictor()

# Cache for stock data to reduce API calls
@lru_cache(maxsize=100)
def get_stock_data(symbol, start_date, end_date):
    max_retries = 3
    retry_delay = 5  # seconds
    
    for attempt in range(max_retries):
        try:
            stock_data = yf.download(symbol, start=start_date, end=end_date)
            if not stock_data.empty:
                return stock_data
            time.sleep(retry_delay)
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
            else:
                raise e
    return pd.DataFrame()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        symbol = data.get('symbol', 'AAPL')
        days = int(data.get('days', 7))
        
        # Get historical data - using more recent data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=90)  # Using last 90 days to ensure we have enough data
        
        # Use cached function to get stock data
        stock_data = get_stock_data(symbol, start_date, end_date)
        
        if stock_data.empty:
            return jsonify({'error': 'No data found for the given symbol'}), 400
            
        # Prepare data for prediction
        df = pd.DataFrame(stock_data)
        df = df[['Close']]
        df.columns = ['Close']
        
        def generate():
            try:
                # Create a function to log training progress
                def log_training_progress(message):
                    yield json.dumps({'status': 'training', 'message': message}) + '\n'
                
                # Train the model on the recent data
                predictor.train(df, epochs=50, batch_size=32, log_function=log_training_progress)
                
                # Make prediction
                prediction = predictor.predict(df, days)
                
                # Get the actual current price from the most recent data point
                current_price = df['Close'].iloc[-1]
                
                # Send final response
                yield json.dumps({
                    'symbol': symbol,
                    'prediction': prediction.tolist(),
                    'last_price': current_price,
                    'training_data': df['Close'].tolist(),
                    'training_dates': df.index.strftime('%Y-%m-%d').tolist()
                }) + '\n'
                
            except ValueError as e:
                yield json.dumps({'error': str(e)}) + '\n'
            except Exception as e:
                yield json.dumps({'error': str(e)}) + '\n'
        
        return Response(stream_with_context(generate()), mimetype='application/json')
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True) 