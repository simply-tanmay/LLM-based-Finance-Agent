import sys
import time
from random import randint
from datetime import datetime
import requests

# Try to import yfinance-cache first, fall back to yfinance
try:
    import yfinance_cache as yf
    print("Using yfinance-cache")
except ImportError:
    import yfinance as yf
    print("Using standard yfinance")

def test_connection():
    print("Testing connection to Yahoo Finance...")
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    try:
        response = requests.get("https://finance.yahoo.com", headers=headers)
        print(f"Connection to Yahoo Finance: {'Success' if response.status_code == 200 else 'Failed'}")
        print(f"Status code: {response.status_code}")
    except Exception as e:
        print(f"Connection test failed: {str(e)}")

def test_yfinance():
    # Test with a different date range and stock
    start_date = datetime(2022, 1, 1)
    end_date = datetime(2022, 1, 31)
    
    print(f"\nTesting with AAPL from {start_date} to {end_date}")
    
    try:
        # Add a random delay to avoid rate limiting
        delay = randint(1, 3)
        print(f"Waiting {delay} seconds before making request...")
        time.sleep(delay)
        
        # First try to get the ticker info
        ticker = yf.Ticker("AAPL")
        print("\nTicker info:")
        info = ticker.info
        if info:
            print(info)
        else:
            print("No ticker info available")
        
        # Add another delay
        delay = randint(1, 3)
        print(f"\nWaiting {delay} seconds before downloading data...")
        time.sleep(delay)
        
        # Then try to download the data
        print("\nAttempting to download data...")
        data = yf.download("AAPL", start=start_date, end=end_date, progress=False)
        
        if not data.empty:
            print("\nDownloaded data:")
            print(data.head())
            print("\nData shape:", data.shape)
        else:
            print("\nNo data downloaded. Trying alternative method...")
            try:
                # Try using a different method to get the data
                data = ticker.history(start=start_date, end=end_date)
                if not data.empty:
                    print("\nData downloaded using alternative method:")
                    print(data.head())
                    print("\nData shape:", data.shape)
                else:
                    print("Alternative method also returned empty data")
            except Exception as e2:
                print(f"Alternative method failed: {str(e2)}")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        print("\nTrying one more time with a different approach...")
        try:
            # Try one more time with a different approach
            data = yf.download("AAPL", period="1mo", interval="1d")
            if not data.empty:
                print("\nData downloaded using period method:")
                print(data.head())
                print("\nData shape:", data.shape)
            else:
                print("Period method also returned empty data")
        except Exception as e3:
            print(f"Final attempt also failed: {str(e3)}")

if __name__ == "__main__":
    test_connection()
    test_yfinance() 