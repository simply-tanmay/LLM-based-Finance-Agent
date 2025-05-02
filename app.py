from flask import Flask, request, jsonify
from stock_prediction import StockPredictor
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

app = Flask(__name__)
predictor = StockPredictor()

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        symbol = data.get('symbol', 'AAPL')
        days = int(data.get('days', 7))
        
        # Get historical data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365)
        stock_data = yf.download(symbol, start=start_date, end=end_date)
        
        if stock_data.empty:
            return jsonify({'error': 'No data found for the given symbol'}), 400
            
        # Prepare data for prediction
        df = pd.DataFrame(stock_data)
        df = df[['Close']]
        df.columns = ['Close']
        
        # Make prediction
        prediction = predictor.predict(df, days)
        
        return jsonify({
            'symbol': symbol,
            'prediction': prediction.tolist(),
            'last_price': df['Close'].iloc[-1]
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True) 