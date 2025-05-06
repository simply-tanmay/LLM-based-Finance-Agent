import yfinance as yf

ticker = yf.Ticker("MSFT")
data = ticker.history(period="1mo")
print("DataFrame shape:", data.shape)
print(data)