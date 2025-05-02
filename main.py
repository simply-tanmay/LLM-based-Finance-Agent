import json
from datetime import datetime, timedelta
from utils import Agent
from stock_prediction import StockPredictor
import yfinance as yf
import pandas as pd

def run_prediction(symbol='AAPL', days=7):
    print(f"\nRunning stock prediction for {symbol} ({days} days)")
    print("=" * 50)
    
    # Initialize predictor
    predictor = StockPredictor()
    
    # Get historical data
    end_date = datetime.now()
    start_date = end_date - timedelta(days=90)
    
    print(f"Fetching historical data from {start_date} to {end_date}...")
    stock_data = yf.download(symbol, start=start_date, end=end_date)
    
    if stock_data.empty:
        print("No data found for the given symbol")
        return
    
    # Prepare data
    df = pd.DataFrame(stock_data)
    df = df[['Close']]
    df.columns = ['Close']
    
    # Train model
    print("\nTraining model...")
    predictor.train(df, epochs=50, batch_size=32, log_function=lambda msg: print(msg))
    
    # Make prediction
    print("\nMaking predictions...")
    prediction = predictor.predict(df, days)
    
    # Get current price
    current_price = df['Close'].iloc[-1]
    
    # Print results
    print("\nResults:")
    print(f"Current price: ${current_price:.2f}")
    print(f"Next day prediction: ${prediction[0]:.2f}")
    print("\nFuture predictions:")
    for i, pred in enumerate(prediction, 1):
        print(f"Day {i}: ${pred:.2f}")

def run_backtesting(config):
    try:
        agent = Agent(config)
        
        # Using a different date range that's more likely to have data
        start_date = datetime(2022, 1, 1)
        end_date = datetime(2022, 1, 31)
        
        print(f"Starting backtesting for {config['stock_symbol']}")
        print(f"Date range: {start_date} to {end_date}")
        
        agent.backtesting(start_date, end_date, verbose=True)
        
    except Exception as e:
        print(f"Error in backtesting: {str(e)}")
        raise

def main():
    try:
        # Load config
        with open('config.json', 'r', encoding="utf-8") as file:
            config = json.load(file)
        
        # Run prediction
        run_prediction(config['stock_symbol'], days=7)
        
        # Run backtesting
        run_backtesting(config)
        
    except Exception as e:
        print(f"Error in main: {str(e)}")
        raise

if __name__ == '__main__':
    main()
