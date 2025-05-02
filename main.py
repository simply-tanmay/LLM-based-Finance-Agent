import json
from datetime import datetime
from utils import Agent

def main():
    try:
        with open('config.json', 'r', encoding="utf-8") as file:
            config = json.load(file)
        agent = Agent(config)
        
        # Using a different date range that's more likely to have data
        start_date = datetime(2022, 1, 1)
        end_date = datetime(2022, 1, 31)
        
        print(f"Starting backtesting for {config['stock_symbol']}")
        print(f"Date range: {start_date} to {end_date}")
        
        agent.backtesting(start_date, end_date, verbose=True)
        
    except Exception as e:
        print(f"Error in main: {str(e)}")
        raise

if __name__ == '__main__':
    main()
