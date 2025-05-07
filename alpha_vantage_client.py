import os
import time
from datetime import datetime, timedelta
import pandas as pd
from alpha_vantage.timeseries import TimeSeries
import logging
from functools import wraps
import random

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AlphaVantageClient:
    def __init__(self, api_key=None):
        """
        Initialize Alpha Vantage client with API key
        """
        self.api_key = api_key or '392AP2MBRDUBL9Z3'  # Use provided key or default
        self.ts = TimeSeries(key=self.api_key, output_format='pandas')
        self.last_request_time = 0
        self.min_request_interval = 12  # Alpha Vantage free tier allows 5 calls per minute

    def _wait_for_rate_limit(self):
        """
        Implement rate limiting to respect API constraints
        """
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.min_request_interval:
            sleep_time = self.min_request_interval - time_since_last
            # Add some jitter to avoid synchronized requests
            sleep_time += random.uniform(0.1, 0.5)
            time.sleep(sleep_time)
        self.last_request_time = time.time()

    def get_daily_data(self, symbol, start_date=None, end_date=None, outputsize='full'):
        """
        Get daily stock data for a given symbol
        
        Args:
            symbol (str): Stock symbol
            start_date (datetime): Start date for data
            end_date (datetime): End date for data
            outputsize (str): Size of output ('compact' or 'full')
            
        Returns:
            pandas.DataFrame: Stock data
        """
        try:
            self._wait_for_rate_limit()
            
            # Get daily data
            data, meta_data = self.ts.get_daily(symbol=symbol, outputsize=outputsize)
            
            # Convert index to datetime if it's not already
            if not isinstance(data.index, pd.DatetimeIndex):
                data.index = pd.to_datetime(data.index)
            
            # Filter by date range if provided
            if start_date and end_date:
                mask = (data.index >= start_date) & (data.index <= end_date)
                data = data.loc[mask]
            
            if data.empty:
                raise ValueError(f"No data found for {symbol} in the specified date range")
            
            return data
            
        except Exception as e:
            logger.error(f"Error fetching daily data for {symbol}: {str(e)}")
            raise

    def get_intraday_data(self, symbol, interval='5min', outputsize='compact'):
        """
        Get intraday stock data for a given symbol
        
        Args:
            symbol (str): Stock symbol
            interval (str): Time interval ('1min', '5min', '15min', '30min', '60min')
            outputsize (str): Size of output ('compact' or 'full')
            
        Returns:
            pandas.DataFrame: Stock data
        """
        try:
            self._wait_for_rate_limit()
            
            data, meta_data = self.ts.get_intraday(symbol=symbol, interval=interval, outputsize=outputsize)
            
            if data.empty:
                raise ValueError(f"No data found for {symbol}")
            
            return data
            
        except Exception as e:
            logger.error(f"Error fetching intraday data for {symbol}: {str(e)}")
            raise

    def get_company_overview(self, symbol):
        """
        Get company overview data
        
        Args:
            symbol (str): Stock symbol
            
        Returns:
            pandas.DataFrame: Company overview data
        """
        try:
            self._wait_for_rate_limit()
            
            from alpha_vantage.fundamentaldata import FundamentalData
            fd = FundamentalData(key=self.api_key, output_format='pandas')
            data, meta_data = fd.get_company_overview(symbol=symbol)
            
            if data.empty:
                raise ValueError(f"No company overview data found for {symbol}")
            
            return data
            
        except Exception as e:
            logger.error(f"Error fetching company overview for {symbol}: {str(e)}")
            raise

def get_stock_data(symbol, start_date=None, end_date=None, interval='daily'):
    """
    Main function to get stock data with proper error handling and rate limiting
    
    Args:
        symbol (str): Stock symbol
        start_date (datetime): Start date for data
        end_date (datetime): End date for data
        interval (str): Data interval ('daily' or 'intraday')
        
    Returns:
        pandas.DataFrame: Stock data
    """
    client = AlphaVantageClient()
    
    try:
        if interval == 'daily':
            data = client.get_daily_data(symbol, start_date, end_date, outputsize='full')
        else:
            data = client.get_intraday_data(symbol, interval=interval)
            
        if data.empty:
            raise ValueError(f"No data found for {symbol}")
            
        return data
        
    except Exception as e:
        logger.error(f"Error in get_stock_data: {str(e)}")
        raise 