import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
import multiprocessing
import time 

def seq_model():
    df = pd.read_csv("Cleaned_Apartments.csv")
    df = df.drop(columns=['id'])
    # Extracting sorted_prices list
    df['sorted_prices'] = df['sorted_prices'].str.strip('[]').str.split(', ')

    # Creating individual columns for prices
    for i in range(len(df['sorted_prices'].iloc[0])):
        df[f'price_{i+1}'] = df['sorted_prices'].apply(lambda x: int(x[i]))

    # Dropping the original sorted_prices column
    df.drop(columns=['sorted_prices'], inplace=True)
    # Normalize the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df)

    # Split into input (X) and output (y) variables
    X = scaled_data[:, :-1]  # All columns except the last one (price_7) as input
    y = scaled_data[:, -1]    # Last column (price_7) as output

    # Reshape input data to 3D tensor (samples, timesteps, features)
    # Assuming you want to consider previous 6 prices to predict the next one (change it as necessary)
    timesteps = 6
    X_reshaped = np.array([X[i:timesteps+i] for i in range(len(X)-timesteps)])
    y_reshaped = y[timesteps:]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X_reshaped, y_reshaped, test_size=0.2, random_state=42)

    # Build LSTM model
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(LSTM(units=50))
    model.add(Dense(units=1))  # Output layer

    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Train the model
    model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2)

    # Evaluate the model
    loss = model.evaluate(X_test, y_test)
    print("Test Loss:", loss)

    # Make predictions
    y_pred = model.predict(X_test)

    # Calculate R-squared
    r_squared = r2_score(y_test, y_pred)
    print("R-squared:", r_squared)

def price_prediction_model():
    df = pd.read_csv("apartments_pl/apartments_pl_2024_02.csv")
    df = df.dropna()
    
    df['condition'] = df['condition'].map({'low': 0, 'premium': 1})
    df['type'] = df['type'].map({'apartmentBuilding': 1, 'blockOfFlats': 2, 'tenement': 3})
    for col in ['hasParkingSpace', 'hasBalcony', 'hasElevator', 'hasSecurity', 'hasStorageRoom']:
        df[col] = df[col].map({'no': 0, 'yes': 1})

    df['city'] = df['city'].map({'szczecin':1,
                                'gdynia':2,
                                'krakow':3,
                                'poznan':4,
                                'bialystok':5,
                                'gdansk':6,
                                'wroclaw':7,
                                'radom':8,
                                'rzeszow':9,
                                'katowice':10,
                                'lublin':11,
                                'czestochowa':12,
                                'warszawa':13,
                                'bydgoszcz':14
                                })
    df['ownership'] = df['ownership'].map({'condominium': 0, 'cooperative': 1})
    df['buildingMaterial'] = df['buildingMaterial'].map({'brick': 0, 'concreteSlab': 1})

    df.drop(columns=['id'], inplace=True)
    df.dropna(inplace = True)

    # Normalize the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df)

    # Split into input (X) and output (y) variables
    X = scaled_data[:, :-1]  
    y = scaled_data[:, -1]   

    # Reshape input data to 3D tensor (samples, timesteps, features)
    # Assuming you want to consider previous 6 prices to predict the next one (change it as necessary)
    timesteps = 6
    X_reshaped = np.array([X[i:timesteps+i] for i in range(len(X)-timesteps)])
    y_reshaped = y[timesteps:]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X_reshaped, y_reshaped, test_size=0.2, random_state=42)

    # Build LSTM model
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(LSTM(units=50))
    model.add(Dense(units=1))  # Output layer

    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Train the model
    model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2)

    # Evaluate the model
    loss = model.evaluate(X_test, y_test)
    print("Test Loss:", loss)

if __name__ == '__main__':

    start_time = time.time()

    # # Create processes for each model
    # process1 = multiprocessing.Process(target=seq_model)
    # process2 = multiprocessing.Process(target=price_prediction_model)

    # # Start processes
    # process1.start()
    # process2.start()

    # # Wait for processes to finish
    # process1.join()
    # process2.join()

    seq_model()
    price_prediction_model()

    end_time = time.time()
    print("Runtime: " + str(end_time - start_time))
