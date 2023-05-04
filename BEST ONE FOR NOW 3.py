### PART 1

import sys
import yfinance as yf
import numpy as np
import pandas as pd
import json
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, Bidirectional
from keras.callbacks import EarlyStopping
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit, RandomizedSearchCV
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest
from scikeras.wrappers import KerasRegressor
from ta import add_all_ta_features
from ta.volatility import BollingerBands
import sqlite3
from sqlite3 import Error
import pickle
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

def initialize_database(db_name):
    conn = None
    try:
        conn = sqlite3.connect(db_name)
    except Error as e:
        print(e)
    return conn

def engineer_features(stock_data):
    # Add Moving Average Convergence Divergence (MACD) indicator
    macd = MACD(stock_data['Close']).macd()
    stock_data['MACD'] = macd
    
    # Add Relative Strength Index (RSI) indicator
    rsi = RSIIndicator(stock_data['Close']).rsi()
    stock_data['RSI'] = rsi
    
    # Drop rows with missing values
    stock_data = stock_data.dropna()
    
    # Scale the data
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(stock_data)
    
    return scaled_data, scaler

def train_test_split(scaled_data):
    # Split data into train and test sets
    train_size = int(len(scaled_data) * 0.8)
    train_data = scaled_data[:train_size, :]
    test_data = scaled_data[train_size:, :]
    
    # Split data into input and output
    X_train, y_train = train_data[:, :-1], train_data[:, -1]
    X_test, y_test = test_data[:, :-1], test_data[:, -1]
    
    # Reshape input to be 3D [samples, timesteps, features]
    X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
    X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))
    
    
    return X_train, y_train, X_test, y_test

def get_historical_data(ticker, start_date, end_date):
    """
    Fetches historical data for a given stock ticker and time range.
    
    Args:
    ticker (str): Stock ticker symbol (e.g., 'AAPL' for Apple)
    start_date (str): Start date in 'YYYY-MM-DD' format
    end_date (str): End date in 'YYYY-MM-DD' format
    
    Returns:
    pandas.DataFrame: DataFrame containing historical data for the specified stock and time range.
    """
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    return stock_data

### PART 2

# Get user inputs for stock ticker and date range
ticker = input("Enter the stock ticker (e.g., AAPL): ")

import datetime as dt

# Get the current date and subtract 5 years
end_datea = pd.Timestamp(dt.datetime.now())
start_datea = end_datea - pd.DateOffset(years=5)

# Convert the dates to strings in the required format
start_date = start_datea.strftime('%Y-%m-%d')
end_date = end_datea.strftime('%Y-%m-%d')

# Initialize the database connection and create the stocks table if it doesn't exist
db_name = 'stock_data.db'
conn = initialize_database(db_name)

def create_stock_table(conn):
    cursor = conn.cursor()
    cursor.execute("""CREATE TABLE IF NOT EXISTS stocks (
                        id INTEGER PRIMARY KEY,
                        ticker TEXT NOT NULL,
                        date TEXT NOT NULL,
                        open REAL,
                        high REAL,
                        low REAL,
                        close REAL,
                        volume INTEGER,
                        UNIQUE(ticker, date)
                      )""")
    conn.commit()

create_stock_table(conn)

# Define a function to create the best_params table if it doesn't exist
def create_best_params_table(conn):
    cursor = conn.cursor()
    cursor.execute("""CREATE TABLE IF NOT EXISTS best_params (
                        id INTEGER PRIMARY KEY,
                        ticker TEXT NOT NULL,
                        params TEXT NOT NULL,
                        score REAL NOT NULL,
                        UNIQUE(ticker)
                      )""")
    conn.commit()

create_best_params_table(conn)

# Define a function to save the best parameters to the database
def save_best_params_to_db(conn, ticker, params, score):
    cursor = conn.cursor()
    cursor.execute("INSERT OR REPLACE INTO best_params (ticker, params, score) VALUES (?, ?, ?)",
                   (ticker, json.dumps(params), score))
    conn.commit()

# Define a function to load the best parameters from the database
def load_best_params_from_db(conn, ticker):
    cursor = conn.cursor()
    query = f"""SELECT params FROM best_params WHERE ticker = '{ticker}'"""
    cursor.execute(query)
    row = cursor.fetchone()
    if row is not None:
        params = json.loads(row[0])
        return params
    else:
        return None

def get_stock_data(ticker, start_date, end_date):
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    return stock_data

def save_stock_data_to_db(conn, ticker, data):
    cursor = conn.cursor()
    for index, row in data.iterrows():
        cursor.execute("INSERT OR IGNORE INTO stocks (ticker, date, open, high, low, close, volume) VALUES (?, ?, ?, ?, ?, ?, ?)",
                       (ticker, index.strftime('%Y-%m-%d'), row['Open'], row['High'], row['Low'], row['Close'], row['Volume']))
    conn.commit()

def get_stock_data_from_db(conn, ticker, start_date, end_date):
    cursor = conn.cursor()
    query = f"""SELECT * FROM stocks
                WHERE Ticker = '{ticker}' AND
                Date >= '{start_date}' AND
                Date <= '{end_date}'"""
    cursor.execute(query)
    rows = cursor.fetchall()
    columns = [description[0] for description in cursor.description]
    data = pd.DataFrame(rows, columns=columns)
    if 'Date' in data.columns:
        data['Date'] = pd.to_datetime(data['Date'])
        data.set_index('Date', inplace=True)
    return data

### PART 3

def feature_engineering(data):
    data = add_all_ta_features(data, open="Open", high="High", low="Low", close="Close", volume="Volume", fillna=True)

    # Add Bollinger Bands
    indicator_bb = BollingerBands(close=data["Close"], window=20, window_dev=2)
    data['bb_bbm'] = indicator_bb.bollinger_mavg()
    data['bb_bbh'] = indicator_bb.bollinger_hband()
    data['bb_bbl'] = indicator_bb.bollinger_lband()

    # Replace NaN and Inf values with 0
    data = data.replace([np.inf, -np.inf, np.nan], 0)
    
    return data

# Get stock data and perform feature engineering
data = get_stock_data(ticker, start_date, end_date)
data = feature_engineering(data)

# Drop unnecessary columns and scale the data
close_scaler = MinMaxScaler(feature_range=(0, 1))
close_prices = data['Close'].values.reshape(-1, 1)
scaled_close_prices = close_scaler.fit_transform(close_prices)

columns_to_drop = ['Open','High', 'Low', 'Close', 'Volume']
if 'Dividends' in data.columns:
    columns_to_drop.append('Dividends')
if 'Stock Splits' in data.columns:
    columns_to_drop.append('Stock Splits')

data.drop(columns_to_drop, axis=1, inplace=True)

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

###PART 4A

# Split the data into features and target
X = scaled_data
y = scaled_close_prices.ravel()  # Reshape target data to a 1D array

# Split the data into training and test sets
train_size = int(len(X) * 0.8)
test_size = len(X) - train_size
X_train, X_test = X[0:train_size,:], X[train_size:len(X),:]
y_train, y_test = y[0:train_size], y[train_size:len(y)]

# Fill any missing values in the data
data.fillna(method='ffill', inplace=True)
data.fillna(method='bfill', inplace=True)
data.dropna(inplace=True)

# Create a pipeline with the scaler and the regressor
pipeline = Pipeline([
    ('scaler', MinMaxScaler()),
    ('regressor', RandomForestRegressor())
])

# Define the parameter distributions for the random search
param_distributions = {
    'scaler__feature_range': [(0, 1), (0, 0.5), (0, 0.1)],
    'regressor__n_estimators': [50, 100, 200],
    'regressor__max_depth': [5, 10, 20, None],
    'regressor__min_samples_split': [2, 5, 10],
    'regressor__min_samples_leaf': [1, 2, 4]
}

# Perform the random search
random_search = RandomizedSearchCV(pipeline, param_distributions=param_distributions, n_iter=10, cv=5, verbose=1)
random_search.fit(X_train, y_train)

# Print the best parameters and the best score
print("Best parameters: ", random_search.best_params_)
print("Best score: ", random_search.best_score_)

# Fit the model with the best parameters on the training data
model = random_search.best_estimator_
model.fit(X_train, y_train)

# Create a MinMaxScaler object
scaler = MinMaxScaler(feature_range=(0, 1))

# Fit the scaler on the training data
scaler.fit(X_train)

# Save the best parameters to the database
save_best_params_to_db(conn, ticker, random_search.best_params_, random_search.best_score_)

# Load the best parameters from the database (if they exist)
best_params = load_best_params_from_db(conn, ticker)
if best_params is not None:
    # Use the saved best parameters
    model = pipeline.set_params(**best_params)
else:
    # Fit the model with the best parameters on the training data
    model = random_search.best_estimator_
    model.fit(X_train, y_train)


### PART 4B

# Evaluate the model on the test data
X_test_scaled = scaler.transform(X_test)
y_pred = model.predict(X_test_scaled)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
print("RMSE: ", rmse)

# Use the model to make a prediction for tomorrow's price
last_data_point = data.tail(1)
last_scaled_data_point = scaler.transform(last_data_point)
next_day_prediction_scaled = model.predict(last_scaled_data_point)
next_day_prediction = close_scaler.inverse_transform(next_day_prediction_scaled.reshape(-1, 1))

# Print the prediction
print("Tomorrow's predicted closing price:", next_day_prediction[0][0])

# Use the model to make a prediction for the next 7 days
next_week_prediction_scaled = last_scaled_data_point
for i in range(7):
    next_week_prediction_scaled = np.append(next_week_prediction_scaled, model.predict(np.array([next_week_prediction_scaled[-1]]).reshape((1, -1))), axis=0)

# Inverse transform the predicted prices to get the actual prices
next_week_prediction = close_scaler.inverse_transform(next_week_prediction_scaled.reshape(-1, 1))

# Print the predictions for the next 7 days
print("Predicted closing prices for the next 7 days:")
for i in range(1, 8):
    print("Day {}: {}".format(i, next_week_prediction[i][0]))

### PART 5

# Inverse transform the predictions and the actual prices
y_pred = y_pred.reshape(-1, 1)
y_pred = close_scaler.inverse_transform(y_pred)
y_test = y_test.reshape(-1, 1)
y_test = close_scaler.inverse_transform(y_test)

# Create a plot of the predicted prices and the actual prices
plt.plot(y_pred, label='Predicted')
plt.plot(y_test, label='Actual')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()

# Use the model to make predictions for the next week, 30 days, 60 days, and 90 days
last_data_point = data.tail(1)
last_scaled_data_point = scaler.transform(last_data_point)
last_scaled_data_point = last_scaled_data_point.reshape((1, -1))

next_week_prediction_scaled = model.predict(last_scaled_data_point)
for i in range(5):
    next_week_prediction_scaled = np.append(next_week_prediction_scaled, model.predict(np.array([next_week_prediction_scaled[-1]]).reshape((1, -1))), axis=0)

next_week_prediction = close_scaler.inverse_transform(next_week_prediction_scaled)

next_30days_prediction_scaled = model.predict(last_scaled_data_point)
for i in range(30):
    next_30days_prediction_scaled = np.append(next_30days_prediction_scaled, model.predict(np.array([next_30days_prediction_scaled[-1]]).reshape((1, -1))), axis=0)

next_30days_prediction = close_scaler.inverse_transform(next_30days_prediction_scaled)

next_60days_prediction_scaled = model.predict(last_scaled_data_point)
for i in range(60):
    next_60days_prediction_scaled = np.append(next_60days_prediction_scaled, model.predict(np.array([next_60days_prediction_scaled[-1]]).reshape((1, -1))), axis=0)

next_60days_prediction = close_scaler.inverse_transform(next_60days_prediction_scaled)

next_90days_prediction_scaled = model.predict(last_scaled_data_point)
for i in range(90):
    next_90days_prediction_scaled = np.append(next_90days_prediction_scaled, model.predict(np.array([next_90days_prediction_scaled[-1]]).reshape((1, -1))), axis=0)

next_90days_prediction = close_scaler.inverse_transform(next_90days_prediction_scaled)

# Create a new dataframe with the predictions
last_price = data['Close'][-1]
next_week_price = next_week_prediction[6][0]
next_30days_price = next_30days_prediction[30][0]
next_60days_price = next_60days_prediction[60][0]
next_90days_price = next_90days_prediction[90][0]

predictions = pd.DataFrame({'Price': [last_price, next_week_price, next_30days_price, next_60days_price, next_90days_price]},
                           index=['Last Price', 'Next Week', 'Next 30 Days', 'Next 60 Days', 'Next 90 Days'])

# Plot the predictions
plt.figure(figsize=(12, 6))
plt.plot(data['Close'], label='Actual Price')
plt.plot(next_week_prediction, label='Next Week Prediction')
plt.plot(next_30days_prediction, label='Next 30 Days Prediction')
plt.plot(next_60days_prediction, label='Next 60 Days Prediction')
plt.plot(next_90days_prediction, label='Next 90 Days Prediction')
plt.title('Stock Price Prediction')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()

print(predictions)

### PART 6

# Create LSTM model
model = Sequential()
model.add(Bidirectional(LSTM(64, return_sequences=True), input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dropout(0.2))
model.add(Bidirectional(LSTM(32, return_sequences=True)))
model.add(Dropout(0.2))
model.add(Bidirectional(LSTM(16, return_sequences=False)))
model.add(Dropout(0.2))
model.add(Dense(1))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Fit the model on the training data
early_stop = EarlyStopping(monitor='val_loss', patience=10)
history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.1, callbacks=[early_stop], shuffle=False)

# Evaluate the model on the test data
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
print("RMSE: ", rmse)

# Use the model to make a prediction for tomorrow's price
last_data_point = data.tail(1)
last_scaled_data_point = scaler.transform(last_data_point)
last_scaled_data_point = last_scaled_data_point.reshape((1, -1, 1))
next_day_prediction_scaled = model.predict(last_scaled_data_point)
next_day_prediction = close_scaler.inverse_transform(next_day_prediction_scaled)

# Print the prediction
print("Tomorrow's predicted closing price:", next_day_prediction[0][0])

# Inverse transform the predictions and the actual prices
y_pred = y_pred.reshape(-1, 1)
y_pred = close_scaler.inverse_transform(y_pred)
y_test = y_test.reshape(-1, 1)
y_test = close_scaler.inverse_transform(y_test)

# Create a plot of the predicted prices and the actual prices
plt.plot(y_pred, label='Predicted')
plt.plot(y_test, label='Actual')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()

### PART 7

# Use the model to make predictions for the next week, 30 days, 60 days, and 90 days
last_data_point = data.tail(1)
last_scaled_data_point = scaler.transform(last_data_point)

next_week_prediction_scaled = model.predict(last_scaled_data_point)
for i in range(5):
    next_week_prediction_scaled = np.append(next_week_prediction_scaled, model.predict(np.array([next_week_prediction_scaled[-1]]).reshape((1, -1))), axis=0)

next_week_prediction = close_scaler.inverse_transform(next_week_prediction_scaled.reshape(-1, 1))

next_30days_prediction_scaled = model.predict(last_scaled_data_point)
for i in range(30):
    next_30days_prediction_scaled = np.append(next_30days_prediction_scaled, model.predict(np.array([next_30days_prediction_scaled[-1]]).reshape((1, -1))), axis=0)

next_30days_prediction = close_scaler.inverse_transform(next_30days_prediction_scaled.reshape(-1, 1))

next_60days_prediction_scaled = model.predict(last_scaled_data_point)
for i in range(60):
    next_60days_prediction_scaled = np.append(next_60days_prediction_scaled, model.predict(np.array([next_60days_prediction_scaled[-1]]).reshape((1, -1))), axis=0)

next_60days_prediction = close_scaler.inverse_transform(next_60days_prediction_scaled.reshape(-1, 1))

next_90days_prediction_scaled = model.predict(last_scaled_data_point)
for i in range(90):
    next_90days_prediction_scaled = np.append(next_90days_prediction_scaled, model.predict(np.array([next_90days_prediction_scaled[-1]]).reshape((1, -1))), axis=0)

next_90days_prediction = close_scaler.inverse_transform(next_90days_prediction_scaled.reshape(-1, 1))

# Create a new dataframe with the predictions
last_price = data['Close'][-1]
next_week_price = next_week_prediction[6][0]
next_30days_price = next_30days_prediction[29][0]
next_60days_price = next_60days_prediction[59][0]
next_90days_price = next_90days_prediction[89][0]

predictions = pd.DataFrame({'Price': [last_price, next_week_price, next_30days_price, next_60days_price, next_90days_price]},
                           index=['Last Price', 'Next Week', 'Next 30 Days', 'Next 60 Days', 'Next 90 Days'])

# Plot the predictions
plt.figure(figsize=(12, 6))
plt.plot(data['Close'], label='Actual Price')
plt.plot(next_week_prediction, label='Next Week Prediction')
plt.plot(next_30days_prediction, label='Next 30 Days Prediction')
plt.plot(next_60days_prediction, label='Next 60 Days Prediction')
plt.plot(next_90days_prediction, label='Next 90 Days Prediction')
plt.title('Stock Price Prediction')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()

print(predictions)