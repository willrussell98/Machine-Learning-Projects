############## IMPORT LIBRARIES ###############

import pandas as pd
import numpy as np
from pandas.io.pytables import DataCol
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error

############## STATIC FUNCTIONS ###############

# convert time string into seconds
def time_string_to_seconds(time_str):
    '''Converts a time string of format 'H:MM:SS.sss' or 'MM:SS.sss' to seconds.'''
    if pd.isna(time_str):
        return None
    
    parts = time_str.split(":")
    if len(parts) == 3:  # format: H:MM:SS.sss
        hours, minutes, seconds = parts
        return int(hours) * 3600 + int(minutes) * 60 + float(seconds)
    elif len(parts) == 2:  # format: MM:SS.sss
        minutes, seconds = parts
        return int(minutes) * 60 + float(seconds)


# convert seconds to timestamp
def seconds_to_time_string(seconds):
    '''Converts seconds to a time string of format 'H:MM:SS.sss'.'''
    hours = int(seconds // 3600)
    seconds %= 3600
    minutes = int(seconds // 60)
    seconds %= 60
    return f"{hours}:{minutes:02}:{seconds:06.3f}"


# function to determine time based upon the leader grouped by race
def adjust_times(df):
  '''Takes raceId to the determine the exact timestamp of a driver in a certain race'''
  races = df['raceId'].unique()
  
  for race in races:
      race_data = df[df['raceId'] == race]
      
      # Ensure there's a winner's time for this race before proceeding
      if '1' in race_data['position'].values:
          winner_time = race_data.loc[race_data['position'] == '1', 'time'].iloc[0]
          
          # Convert the winner's time to seconds
          winner_time_seconds = time_string_to_seconds(winner_time)
          
          for index, row in race_data.iterrows():
              if row['time'] and "+" in row['time']:
                  if ":" in row['time']:
                      difference_seconds = time_string_to_seconds(row['time'].replace("+", ""))
                  else:
                      difference_seconds = float(row['time'].replace("+", ""))
                  
                  absolute_time_seconds = winner_time_seconds + difference_seconds
                  df.at[index, 'time'] = seconds_to_time_string(absolute_time_seconds)
    
  return df


# function to split data into train, validation and test
def train_valid_test_split(X , y):
  '''This function splits the dataset to 70% train, 15% validation and 15% test'''
  train = int(len(X) * .7)
  validation = int(len(X) * .85)

  X_train, y_train = X[:train], y[:train]
  X_val, y_val = X[train:validation], y[train:validation]
  X_test, y_test = X[validation:], y[validation:]

  return X_train, y_train, X_val, y_val, X_test, y_test


############## CLASS ###############

# create class to predict f1 race ranking
class F1Predictor():

  # initialise class
  def __init__(self, race, drivers, constructors, constructor_results, results, lap_times, status, driver_standings, pit_stops, qualifying, circuits):
    self.race = race
    self.drivers = drivers
    self.constructors = constructors
    self.constructor_results = constructor_results
    self.results = results
    self.lap_times = lap_times
    self.status = status
    self.driver_standings = driver_standings
    self.pit_stops = pit_stops
    self.qualifying = qualifying
    self.circuits = circuits

  # function to load data
  def load_data(self):
    '''Loads the data for independent variables'''

    try:
      # races data
      self.races_df = pd.read_csv(self.race)
      self.races_df = self.races_df[['raceId','date','year','circuitId']]

      # result data
      self.results_df = pd.read_csv(self.results)
      self.results_df = self.results_df[['resultId','raceId','driverId','constructorId','number','grid','position','positionOrder','points','laps','time',
                              'milliseconds','fastestLap','rank','fastestLapTime','fastestLapSpeed','statusId']]

      # constructor results data
      self.constructor_results_df = pd.read_csv(self.constructor_results)
      self.constructor_results_df = self.constructor_results_df[['constructorResultsId','raceId','constructorId','points']]

      # drivers data
      self.drivers_df = pd.read_csv(self.drivers)
      self.drivers_df = self.drivers_df[['driverId','driverRef','code','forename','surname']]
      self.drivers_df['full_name'] = (self.drivers_df['forename'].str.strip() + ' ' + self.drivers_df['surname'].str.strip()).str.strip()

      # constructors data
      self.constructors_df = pd.read_csv(self.constructors)
      self.constructors_df = self.constructors_df[['constructorId','constructorRef','name']]

      # lap time data
      self.lap_times_df = pd.read_csv(self.lap_times)
      self.lap_times_df = self.lap_times_df[['raceId','driverId','lap','position','time']]
      self.lap_times_df = self.lap_times_df.groupby(['raceId', 'driverId'])['time'].agg(slowestLapTime='max').reset_index()

      # car status data
      self.status_df = pd.read_csv(self.status)
      self.status_df = self.status_df[['statusId','status']]

      # driver standings data
      self.driver_standings_df = pd.read_csv(self.driver_standings)
      self.driver_standings_df = self.driver_standings_df[['driverStandingsId','raceId','driverId','points']]
      self.driver_standings_df = self.driver_standings_df.rename(columns={'points':'overall_points'})

      # pit stop data
      self.pit_stops_df = pd.read_csv(self.pit_stops)
      self.pit_stops_df = self.pit_stops_df[['raceId','driverId','stop','lap','time','duration']]
      self.pit_stops_df['average_pit_stop_duration'] = pd.to_numeric(self.pit_stops_df['duration'], errors='coerce')
      self.pit_stops_df = self.pit_stops_df.groupby(['raceId', 'driverId'])['average_pit_stop_duration'].mean().reset_index()

      # qualifying data
      self.qualifying_df = pd.read_csv(self.qualifying)
      self.qualifying_df = self.qualifying_df[['qualifyId','raceId','driverId','q1','q2','q3']]
      self.qualifying_df = self.qualifying_df.rename(columns={'q1':'q1_time','q2':'q2_time','q3':'q3_time'})

      # circuits data
      self.circuits_df = pd.read_csv(self.circuits)

      return

    except Exception as e:
      print(f'Could not fully access data due to error: {e}')

      return


  # function to preprocess data
  def preprocess_data(self):
    '''function to combine the dataframes together in a single format'''
  
    try:
      # adjust timestamp of drivers in each race to get all timestamp values and not "+ time"
      self.results_df = adjust_times(self.results_df)

      # combine the race with the circuit
      first_df = pd.merge(self.races_df,self.circuits_df,how='inner',on='circuitId')

      # combine the previous dataframe with results
      second_df = pd.merge(first_df,self.results_df,how='inner',on='raceId')

      # combine the previous dataframe with drivers
      third_df = pd.merge(second_df,self.drivers_df,how='inner',on='driverId')
      third_df = third_df[['raceId','date','year','name','circuitId','alt','resultId','driverId','driverRef','code','full_name','constructorId',	
                          'number','grid','position','positionOrder','points','laps','time','milliseconds','fastestLap','rank',
                          'fastestLapTime','fastestLapSpeed','statusId']]  

      # combine the previous dataframe with constructors                  
      self.constructors_df = self.constructors_df.rename(columns={'name':'constructor_name'})
      fourth_df = pd.merge(third_df,self.constructors_df,how='inner',on='constructorId')
      fourth_df = fourth_df[['raceId','date','year','name','circuitId','alt','resultId','driverId','driverRef','code','full_name','constructorId',	
                          'constructor_name','number','grid','position','positionOrder','points','laps','time','milliseconds','fastestLap','rank',
                          'fastestLapTime','fastestLapSpeed','statusId']]
      
      # combine the previous dataframe with lap times
      fifth_df = pd.merge(fourth_df,self.lap_times_df,how='inner',on=['raceId','driverId'])

      # combine the previous dataframe with car status
      sixth_df = pd.merge(fifth_df,self.status_df,how='inner',on='statusId')

      # combine the previous dataframe with driver standings
      seventh_df = pd.merge(sixth_df,self.driver_standings_df,how='inner',on=['raceId','driverId'])

      # combine the previous dataframe with pit stop data
      eighth_df = pd.merge(seventh_df,self.pit_stops_df,how='inner',on=['raceId','driverId'])

      # combine the previous dataframe with qualifying data
      final_df = pd.merge(eighth_df,self.qualifying_df,how='inner',on=['raceId','driverId'])

      # format final dataframe
      final_df = final_df[['raceId','date','name','full_name','constructor_name','grid','position','time',
                          'q1_time',	'q2_time', 'q3_time','fastestLapTime','slowestLapTime','fastestLapSpeed',
                          'average_pit_stop_duration','status','points']]
      final_df.replace('\\N', None, inplace=True)
      final_df['status'] = final_df['status'].apply(lambda x: 1 if x == 'Finished' else 0)
      final_df = final_df.sort_values(by=['raceId','date','position'], ascending=[True,True,True])

      return final_df

    except Exception as e:
      print(f'Could not preprocess due to error: {e}')

      return


  # feature engineering function to get independent and dependent variables
  def feature_engineering(self, final_df):
    try:
      # convert all timestamp columns into second values
      timestamp_columns = ['time', 'q1_time', 'q2_time', 'q3_time', 'fastestLapTime', 'slowestLapTime']

      for col in timestamp_columns:
          final_df[col] = final_df[col].apply(time_string_to_seconds)

      # take a copy of the features dataframe
      features = final_df.copy()

      # Calculate the rolling average time
      rolling_avg_time = (features.groupby(['full_name', 'constructor_name'])['time'].rolling(window=3, min_periods=1).mean().reset_index(level=[0,1], drop=True))
      features['rolling_average_time'] = round(rolling_avg_time,2)

      # Calculate the rolling average grid starting position 
      rolling_avg_grid = (features.groupby(['full_name', 'constructor_name'])['grid'].rolling(window=3, min_periods=1).mean().reset_index(level=[0,1], drop=True))
      features['rolling_average_grid_position'] = round(rolling_avg_grid,2)

      # Calculate the rolling average points
      rolling_avg_points = (features.groupby(['full_name', 'constructor_name'])['points'].rolling(window=3, min_periods=1).mean().reset_index(level=[0,1], drop=True))
      features['rolling_average_points'] = round(rolling_avg_points,2)

      # shift position score so that a current racers perfomance corresponds to next races position
      features['target_finish_position'] = features.groupby(['full_name','constructor_name'])['position'].shift(-1)

      # if they have a null value in position then fix their position as 21
      features['target_finish_position'].fillna(21, inplace=True)

      # Filter columns
      features = features[['raceId', 'date', 'name', 'full_name', 'constructor_name', 
                          'rolling_average_grid_position', 'rolling_average_points',
                          'rolling_average_time', 'q1_time', 'q2_time', 'q3_time', 'fastestLapTime',
                          'slowestLapTime', 'fastestLapSpeed', 'average_pit_stop_duration', 'status', 
                          'target_finish_position']]

      # convert columsn to specific data types
      features['fastestLapSpeed'] = features['fastestLapSpeed'].astype(float)
      features = features.copy()
      features['target_finish_position'] = features['target_finish_position'].astype(int)

      # identify the raceId for the last race
      last_race_id = features['raceId'].max()

      # split the data
      expected_position = features[features['raceId'] == last_race_id].copy()
      features = features[features['raceId'] != last_race_id].copy()

      return features, expected_position 

    except Exception as e:
      print(f'Could not engineer features due to error: {e}')

      return


  # train the algorithm
  def train_validate_test(self, data, param_grid):
    '''function to train the algorithm using xgboost'''
    try:
      # Filter rows and label the independent variables as X
      X = data[['rolling_average_grid_position','rolling_average_points','q1_time','q2_time','q3_time','fastestLapTime',
              'slowestLapTime','fastestLapSpeed','average_pit_stop_duration','status','target_finish_position']]

      # Shuffle values in dataframe
      X = X.sample(frac=1, random_state=42)

      # Split the data into features and target
      y = X['target_finish_position']
      X = X.drop(columns=['target_finish_position'])

      # Split the dataframe into train, validation and test and convert all to numpy for training
      X_train, y_train, X_val, y_val, X_test, y_test = train_valid_test_split(X , y)
      X_train = X_train.to_numpy()
      y_train = y_train.to_numpy()
      X_val = X_val.to_numpy()
      y_val = y_val.to_numpy()
      X_test = X_test.to_numpy()
      y_test = y_test.to_numpy()

      # Print the shapes of the splits
      print(f"Train set shapes: {X_train.shape}, {y_train.shape}\n")
      print(f"Validation set shapes: {X_val.shape}, {y_val.shape}\n")
      print(f"Test set shapes: {X_test.shape}, {y_test.shape}\n")

      # Create an XGBoost regressor
      xgb_model = xgb.XGBRegressor()

      # Create GridSearchCV
      grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid,
                                scoring='neg_mean_squared_error', cv=3, verbose=2)

      # Fit the grid search to the data
      grid_search.fit(X_train, y_train)

      # Get the best parameters from the grid search
      best_params = grid_search.best_params_
      print(f"Best parameters: {best_params}\n")

      # Create a new XGBoost model with the best parameters
      best_xgb_model = xgb.XGBRegressor(**best_params)

      # Train the best model on the entire training data
      best_xgb_model.fit(X_train, y_train)
      print("== TRAINING FINISHED ==\n")

      # Make predictions on the validation set
      y_pred = best_xgb_model.predict(X_val)

      # Calculate RMSE on the validation set
      rmse = np.sqrt(mean_squared_error(y_val, y_pred))
      print(f"Validation RMSE: {rmse}\n")

      # Make predictions on the test set
      y_pred_test = best_xgb_model.predict(X_test)

      # Calculate RMSE on the test set
      rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
      print(f"Test RMSE: {rmse_test}\n")

      return best_xgb_model

    except Exception as e:
      print(f'Could not train algorithm due to error: {e}')

      return


  # train the algorithm
  def predict_ranking(self, data, algorithm):
    '''function to predict the next ranking using the trained algorithm'''
    try:
      X_expected = data.copy()
      X_expected = X_expected[['rolling_average_grid_position','rolling_average_points','q1_time','q2_time','q3_time','fastestLapTime',
                    'slowestLapTime','fastestLapSpeed','average_pit_stop_duration','status']]

      X_expected = X_expected.to_numpy()

      # Use the trained XGBoost model to predict target_points for the expected data
      predicted_target_position = algorithm.predict(X_expected)

      next_race = data.copy()

      # Assign the predicted target_points to the DataFrame using .loc
      next_race.loc[:, 'predicted_target_position'] = predicted_target_position
      next_race = next_race.sort_values(by='predicted_target_position',ascending=True)
      next_race['predicted_position'] = next_race['predicted_target_position'].rank()
      next_race = next_race[['full_name','constructor_name','predicted_position']].reset_index(drop=True)

      return next_race

    except Exception as e:
      print(f'Could not predict ranking due to error: {e}')

      return


######### MODEL PARAMETERS ##########
param_grid = {
'n_estimators': [100, 200],
'max_depth': [3, 6],
'learning_rate': [0.1, 0.3],
'subsample': [0.8, 1.0],
'colsample_bytree': [0.8, 1.0],
'gamma': [0, 0.1, 0.2],
'min_child_weight': [1, 3, 5]
}


############# PROGRAM  ##############
def main():
  print("== RUNNING ALGORITHM ==\n")

  # initialise the class
  f1 = F1Predictor('races.csv', 'drivers.csv', 'constructors.csv', 'constructor_results.csv', 'results.csv', 
                 'lap_times.csv', 'status.csv', 'driver_standings.csv', 'pit_stops.csv', 'qualifying.csv', 'circuits.csv')
  
  # load the data
  f1.load_data()
  print("== DATA LOADED SUCCESSFULLY ==\n")

  # preprocess data
  data = f1.preprocess_data()
  print("== DATA PREPROCESSED SUCCESSFULLY ==\n")

  # return historical data for training and last race for prediction
  historical_data, last_race = f1.feature_engineering(data)
  print("== FEATURES ENGINEERED SUCCESSFULLY ==\n")

  # train the algorithm
  algorithm = f1.train_validate_test(historical_data, param_grid)
  print("\n== ALGORITHM TRAINED SUCCESSFULLY==\n")

  # return the predicted ranking
  result = f1.predict_ranking(last_race, algorithm)
  print("== PREDICTIONS MADE SUCCESSFULLY ==\n")

  print("== ALGORITHM FINISHED ==\n")

  return result


if __name__ == "__main__":
  results = main()
  print(results)
