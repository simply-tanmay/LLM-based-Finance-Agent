import os
import time
from datetime import datetime, timedelta
import pandas as pd
from alpha_vantage.timeseries import TimeSeries
from alpha_vantage.fundamentaldata import FundamentalData
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_alpha_vantage():
    """
    Test various Alpha Vantage API endpoints and functionality
    """
    # Get API key from environment variable
    api_key = '392AP2MBRDUBL9Z3'  # Your Alpha Vantage API key
    if not api_key:
        raise ValueError("API key not set")

    # Initialize API clients
    ts = TimeSeries(key=api_key, output_format='pandas')
    fd = FundamentalData(key=api_key, output_format='pandas')

    # Test stock symbol
    symbol = 'AAPL'
    
    print(f"\nTesting Alpha Vantage API with {symbol}")
    print("=" * 50)

    try:
        # Test 1: Daily Time Series
        print("\n1. Testing Daily Time Series...")
        data, meta_data = ts.get_daily(symbol=symbol, outputsize='compact')
        print("\nDaily Time Series Data:")
        print(data.head())
        print("\nMeta Data:")
        print(meta_data)

        # Add delay to respect rate limits
        time.sleep(12)  # Alpha Vantage free tier allows 5 API calls per minute

        # Test 2: Intraday Time Series
        print("\n2. Testing Intraday Time Series...")
        data, meta_data = ts.get_intraday(symbol=symbol, interval='5min', outputsize='compact')
        print("\nIntraday Time Series Data:")
        print(data.head())
        print("\nMeta Data:")
        print(meta_data)

        time.sleep(12)

        # Test 3: Company Overview
        print("\n3. Testing Company Overview...")
        data, meta_data = fd.get_company_overview(symbol=symbol)
        print("\nCompany Overview Data:")
        print(data.head())
        print("\nMeta Data:")
        print(meta_data)

        time.sleep(12)

        # Test 4: Technical Indicators (SMA)
        print("\n4. Testing Technical Indicators (SMA)...")
        data, meta_data = ts.get_sma(symbol=symbol, interval='daily', time_period=20, series_type='close')
        print("\nSMA Data:")
        print(data.head())
        print("\nMeta Data:")
        print(meta_data)

        time.sleep(12)

        # Test 5: Search Endpoint
        print("\n5. Testing Search Endpoint...")
        data, meta_data = ts.get_search(keywords='apple')
        print("\nSearch Results:")
        print(data.head())
        print("\nMeta Data:")
        print(meta_data)

        print("\nAll tests completed successfully!")

    except Exception as e:
        logger.error(f"Error during testing: {str(e)}")
        raise

def test_rate_limits():
    """
    Test rate limiting behavior of Alpha Vantage API
    """
    api_key = '392AP2MBRDUBL9Z3'  # Your Alpha Vantage API key
    if not api_key:
        raise ValueError("API key not set")

    ts = TimeSeries(key=api_key, output_format='pandas')
    symbol = 'AAPL'

    print("\nTesting Rate Limits")
    print("=" * 50)

    try:
        # Make multiple requests in quick succession
        for i in range(6):  # Try to exceed the free tier limit
            print(f"\nRequest {i+1}/6")
            try:
                data, meta_data = ts.get_daily(symbol=symbol, outputsize='compact')
                print(f"Request {i+1} successful")
                print(data.head())
            except Exception as e:
                print(f"Request {i+1} failed: {str(e)}")
            
            # Wait 12 seconds between requests
            if i < 5:  # Don't wait after the last request
                print("Waiting 12 seconds before next request...")
                time.sleep(12)

    except Exception as e:
        logger.error(f"Error during rate limit testing: {str(e)}")
        raise

if __name__ == "__main__":
    try:
        # Run main tests
        test_alpha_vantage()
        
        # Run rate limit tests
        test_rate_limits()
        
    except Exception as e:
        logger.error(f"Test suite failed: {str(e)}")
        raise 