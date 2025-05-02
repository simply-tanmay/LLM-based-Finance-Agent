import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

class StockPredictor:
    def __init__(self):
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.model = self._build_model()
        
    def _build_model(self):
        model = Sequential([
            LSTM(units=50, return_sequences=True, input_shape=(60, 1)),
            Dropout(0.2),
            LSTM(units=50, return_sequences=True),
            Dropout(0.2),
            LSTM(units=50),
            Dropout(0.2),
            Dense(units=1)
        ])
        model.compile(optimizer='adam', loss='mean_squared_error')
        return model
        
    def _prepare_data(self, data, lookback=60):
        # Scale the data
        scaled_data = self.scaler.fit_transform(data)
        
        X, y = [], []
        for i in range(lookback, len(scaled_data)):
            X.append(scaled_data[i-lookback:i, 0])
            y.append(scaled_data[i, 0])
            
        return np.array(X), np.array(y)
        
    def train(self, data, epochs=50, batch_size=32):
        X, y = self._prepare_data(data)
        X = np.reshape(X, (X.shape[0], X.shape[1], 1))
        
        self.model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=0)
        
    def predict(self, data, days=7):
        # Prepare the data
        scaled_data = self.scaler.transform(data)
        
        # Get the last 60 days of data
        last_60_days = scaled_data[-60:]
        
        predictions = []
        for _ in range(days):
            # Reshape the data for prediction
            X = np.reshape(last_60_days, (1, 60, 1))
            
            # Make prediction
            pred = self.model.predict(X, verbose=0)
            
            # Inverse transform the prediction
            pred_price = self.scaler.inverse_transform(pred)[0][0]
            predictions.append(pred_price)
            
            # Update the last_60_days array
            last_60_days = np.append(last_60_days[1:], pred)
            
        return np.array(predictions) 