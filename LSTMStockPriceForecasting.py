import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import re
from pandas_datareader import data as pdr
from datetime import date
import yfinance as yf
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_squared_error, mean_absolute_error


class LSTM():

    # initialise variables
    def __init__(self, file_path, label, start_date, end_date, early_stopping, stock_variable):
      self.file_path = file_path
      match = re.search(r'/(?P<content>[^/]+).xlsx$', file_path)
      if match:
        self.stock_name = match.group('content')

      self.label = label
      self.start_date = start_date
      self.end_date = end_date
      self.early_stopping = early_stopping
      self.stock_variable = stock_variable

      try:
        self.data = pd.read_excel(self.file_path)
      except Exception as e:
        print('Error downloading file:' + (str(e)))

      self.data = self.data[['Date',self.stock_variable]]
      self.data = self.data[(self.data['Date'] >= self.start_date) & (self.data['Date'] < self.end_date)]
      self.data.index = self.data.pop('Date')


    def __str__(self):
      print(f'''
      -------------------------------------------
      -------------------------------------------
      -------------------------------------------

                 STOCK = {self.stock_name}
             START DATE = {self.start_date}
               END DATE = {self.end_date}
              EARLY STOPPING = {self.early_stopping}
           PREDICTING ON = {self.stock_variable} price

      -------------------------------------------
      -------------------------------------------
      -------------------------------------------

      ''')


    def visualise_stock(self, data):
      plt.plot(data.index, data[self.stock_variable])
      plt.title(f"Stock {self.stock_variable} Prices Between {self.start_date} and {self.end_date}")
      plt.show()
      return

    # convert the datasets into supervised learning by taking a date, it's closing price and a window of closing prices preceding it
    def create_window_dataframe(self, df, n=5):

      """
        This function takes a dataframe containing a timestamp (index) and a closing price of a stock (column).
        It generates a new dataframe where each row represents a specific date, the last n closing prices as independent variables
        and its own closing price as the target variable.
      """

      dates = []
      data = []

      for i in range(n, len(df)):
          dates.append(df.index[i])
          target = df.iloc[i, 0]
          window = df.iloc[i-n:i, 0].values
          row = np.concatenate(([dates[-1]], window, [target]))
          data.append(row)

      columns = ['Target Date'] + [f'Target-{i}' for i in range(n, 0, -1)] + ['Target']
      newdf = pd.DataFrame(data, columns=columns)

      return newdf


    def format_input_data(self, dataframe):
      np_df = dataframe.to_numpy()

      # get date values
      dates = np_df[:, 0]

      # reshape window matrix ie independent variables into a format suitable for tensorflow
      independent = np_df[:, 1:-1]
      X = independent.reshape((len(dates), independent.shape[1], 1))

      # get target/dependent variables
      Y = np_df[:, -1]

      return dates, X.astype(np.float32), Y.astype(np.float32)


    def train_valid_test_split(self, dates, X , y):
      # set the split to 75% train, 15% validation and 15% test
      train = int(len(dates) * .7)
      validation = int(len(dates) * .85)

      dates_train, X_train, y_train = dates[:train], X[:train], y[:train]
      dates_val, X_val, y_val = dates[train:validation], X[train:validation], y[train:validation]
      dates_test, X_test, y_test = dates[validation:], X[validation:], y[validation:]

      plt.plot(dates_train, y_train)
      plt.plot(dates_val, y_val)
      plt.plot(dates_test, y_test)

      plt.legend(['Train', 'Validation', 'Test'])

      plt.show()

      return dates_train, X_train, y_train, dates_val, X_val, y_val, dates_test, X_test, y_test


    # Build meta LSTM model
    def build_model(self, X_train, y_train, X_val, y_val, window_input=(5,1), LSTM=64, dense_layer=32, activation_layer='relu',
                    target=1, loss_function='mse', learning_rate=0.001, patience_val=5, metric='mean_absolute_error', iterations=100):

      model = Sequential([layers.Input(window_input),
                          layers.LSTM(LSTM),
                          layers.Dense(dense_layer, activation=activation_layer),
                          layers.Dense(dense_layer, activation=activation_layer),
                          layers.Dense(target)])

      model.compile(loss=loss_function,
                    optimizer=Adam(learning_rate=learning_rate),
                    metrics=[metric])

      if self.early_stopping == True:
        # Define the early stopping callback
        early_stopping = EarlyStopping(monitor='val_mean_absolute_error', patience=patience_val)

        # Compile and fit the model with early stopping
        model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=iterations, callbacks=[early_stopping])

        return model

      elif self.early_stopping == False:

        model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=iterations)

        return model


    def evaluation(self, model, dates_train, dates_val, dates_test, X_train, X_val, X_test, y_train, y_val, y_test):
      # Generate predictions for the training, validation, and testing data
      train_predictions = model.predict(X_train).flatten()
      val_predictions = model.predict(X_val).flatten()
      test_predictions = model.predict(X_test).flatten()

      # Calculate evaluation metrics (MSE and MAE) for the training data
      train_mse = mean_squared_error(y_train, train_predictions)
      train_mae = mean_absolute_error(y_train, train_predictions)

      # Calculate evaluation metrics (MSE and MAE) for the validation data
      val_mse = mean_squared_error(y_val, val_predictions)
      val_mae = mean_absolute_error(y_val, val_predictions)

      # Calculate evaluation metrics (MSE and MAE) for the testing data
      test_mse = mean_squared_error(y_test, test_predictions)
      test_mae = mean_absolute_error(y_test, test_predictions)

      # Print the evaluation metrics
      print("\nEvaluation Metrics:\n")
      print(f"Train MSE: {train_mse:.4f}\n")
      print(f"Train MAE: {train_mae:.4f}\n")
      print(f"Validation MSE: {val_mse:.4f}\n")
      print(f"Validation MAE: {val_mae:.4f}\n")
      print(f"Test MSE: {test_mse:.4f}\n")
      print(f"Test MAE: {test_mae:.4f}\n")

      # Assess the model's performance
      print("\nModel Performance:\n")
      if train_mse < val_mse and train_mse < test_mse:
          print("The model has performed well on the training data.\n")
          if val_mse > test_mse:
              print("The model's performance on the validation data is worse than on the testing data.\n")
          else:
              print("The model's performance on the validation data is comparable to its performance on the testing data.\n")
      elif val_mse < train_mse and val_mse < test_mse:
          print("The model has performed well on the validation data.\n")
          if train_mse > test_mse:
              print("The model's performance on the training data is worse than on the testing data.\n")
          else:
              print("The model's performance on the training data is comparable to its performance on the testing data.\n")
      elif test_mse < train_mse and test_mse < val_mse:
          print("The model has performed well on the testing data.\n")
          if train_mse > val_mse:
              print("The model's performance on the training data is worse than on the validation data.\n")
          else:
              print("The model's performance on the training data is comparable to its performance on the validation data.\n")
      else:
          print("The model's performance is similar across different datasets.\n")

      # Determine if the model's performance is good or bad
      if train_mse < 0.1 and val_mse < 0.1 and test_mse < 0.1:
          print("The model's performance is excellent!\n")
      elif train_mse < 0.5 and val_mse < 0.5 and test_mse < 0.5:
          print("The model's performance is good.\n")
      elif train_mse < 1.0 and val_mse < 1.0 and test_mse < 1.0:
          print("The model's performance is acceptable.\n")
      else:
          print("The model's performance needs improvement.\n")

      # Plot the predictions and observations
      plt.plot(dates_train, train_predictions)
      plt.plot(dates_train, y_train)
      plt.plot(dates_val, val_predictions)
      plt.plot(dates_val, y_val)
      plt.plot(dates_test, test_predictions)
      plt.plot(dates_test, y_test)
      plt.legend(['Training Predictions', 'Training Observations',
                  'Validation Predictions', 'Validation Observations',
                  'Testing Predictions', 'Testing Observations'])
      plt.show()

      return test_mse, test_mae





def main(stock_list, start, end, window, early_termination, dependent):

  best_mse = 10000
  Stock1 = None
  best_mae = 10000
  Stock2 = None

  try:
    for stock in stock_list:
      lstm = LSTM(file_path=f'/content/{stock}.xlsx', label=stock, start_date=start, end_date=end, early_stopping=early_termination, stock_variable=dependent)
      lstm.__str__()
      if len(lstm.data) > 0:
        lstm.visualise_stock(lstm.data)
        lstm_df = lstm.create_window_dataframe(lstm.data, n=window)
        lstm_dates, lstm_X, lstm_y = lstm.format_input_data(lstm_df)
        lstm_dates_train, lstm_X_train, lstm_y_train, lstm_dates_val, lstm_X_val, lstm_y_val, lstm_dates_test, lstm_X_test, lstm_y_test = lstm.train_valid_test_split(lstm_dates, lstm_X, lstm_y)
        lstm_model = lstm.build_model(lstm_X_train, lstm_y_train, lstm_X_val, lstm_y_val, window_input=(window,1), LSTM=64, dense_layer=32, activation_layer='relu',
                            target=1, loss_function='mse', learning_rate=0.001, patience_val=15, metric='mean_absolute_error', iterations=500)
        mse, mae = lstm.evaluation(lstm_model, lstm_dates_train, lstm_dates_val, lstm_dates_test, lstm_X_train, lstm_X_val, lstm_X_test, lstm_y_train, lstm_y_val, lstm_y_test)
        if mse < best_mse:
          best_mse = mse
          Stock1 = stock
        if mae < best_mae:
          best_mae = mae
          Stock2 = stock
      else:
        print('              Stock out of time range\n')

  except Exception as e:
      print('Error running algorithm:' + (str(e)))


  print(f'''
        -------------------------------------------
        -------------------------------------------
        -------------------------------------------

            BEST ALGORITHM (acc MSE) = {Stock1}
            BEST ALGORITHM (acc MAE) = {Stock2}

        -------------------------------------------
        -------------------------------------------
        -------------------------------------------

        ''')



######## STOCKS ###########
stock_list = ['META', 'DOW', 'GOOG']
###########################

######## CONFIG ###########
start_date = "2018-01-01"
end_date = "2023-01-01"
window = 10
early_termination = True
dependent = 'Close'
today = date.today()
files = []
successful_tickers = []
data_folder = "/content/"
###########################

print(f"Downloading stock data from Yahoo Finance\n")

def getData(ticker):
    print(ticker)
    data = yf.download(ticker, start=start_date, end=end_date)
    if len(data) > 0:
        dataname = ticker
        files.append(dataname)
        SaveData(data, dataname)  # Save all columns of the DataFrame
        successful_tickers.append(ticker)
    else:
        print(f"Data for {ticker} couldn't be downloaded. Deleting corresponding file.\n")
        deleteFile(data_folder + ticker + '.xlsx')

def SaveData(df, filename):
    df.to_excel(data_folder + filename + '.xlsx')

def deleteFile(file_path):
    if os.path.exists(file_path):
        os.remove(file_path)
        print(f"Deleted file: {file_path}\n")

# This loop will iterate over the ticker list, get data, and save all columns as a file.
for ticker in stock_list:
    getData(ticker)

stock_list = successful_tickers

if __name__ == "__main__":
  main(stock_list, start_date, end_date, window, early_termination, dependent)
