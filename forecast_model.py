import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# --- 1. Data Simulation ---
# In a real-world scenario, you would load your historical production data here.
# This data typically comes from well reports, SCADA systems, or reservoir simulation outputs.
# For this example, we'll simulate data that mimics a typical production decline curve.

def simulate_production_data(days=1825):
    """
    Simulates daily oil production data with some noise and trends.
    This mimics a reservoir's production life (initial ramp-up, plateau, decline).
    """
    time = np.arange(0, days)
    
    # Simulate a more realistic decline curve (e.g., hyperbolic)
    initial_rate = 5000  # bbl/day
    decline_rate = 0.001
    b_exponent = 0.8
    
    # Production rate with noise
    noise = np.random.normal(0, 50, days)
    production = initial_rate / (1 + b_exponent * decline_rate * time)**(1/b_exponent) + noise
    production[production < 0] = 0 # Ensure production is not negative
    
    # Simulate Bottom-Hole Pressure (BHP) - generally declines with production
    initial_bhp = 3000 # psi
    bhp_decline = 0.2 * time
    bhp_noise = np.random.normal(0, 20, days)
    bhp = initial_bhp - bhp_decline + bhp_noise
    
    df = pd.DataFrame({
        'Date': pd.to_datetime(pd.to_datetime('2020-01-01') + pd.to_timedelta(time, unit='d')),
        'OilProduction_BBLD': production,
        'BottomHolePressure_PSI': bhp
    })
    df.set_index('Date', inplace=True)
    return df

print("Simulating reservoir production data...")
data = simulate_production_data()
print("Data simulation complete.")
print(data.head())

# --- 2. Data Visualization ---
print("Visualizing historical production data...")
plt.style.use('seaborn-v0_8-darkgrid')
fig, ax1 = plt.subplots(figsize=(15, 7))

ax1.set_xlabel('Date')
ax1.set_ylabel('Oil Production (BBL/D)', color='tab:blue')
ax1.plot(data.index, data['OilProduction_BBLD'], color='tab:blue', label='Oil Production')
ax1.tick_params(axis='y', labelcolor='tab:blue')

ax2 = ax1.twinx()
ax2.set_ylabel('Bottom Hole Pressure (PSI)', color='tab:red')
ax2.plot(data.index, data['BottomHolePressure_PSI'], color='tab:red', alpha=0.6, label='BHP')
ax2.tick_params(axis='y', labelcolor='tab:red')

fig.suptitle('Simulated Historical Reservoir Performance', fontsize=16)
fig.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()


# --- 3. Data Preprocessing for LSTM ---
# LSTMs are sensitive to the scale of the data. We'll use MinMaxScaler to scale data to [0, 1].
# We will focus on predicting 'OilProduction_BBLD'.
dataset = data['OilProduction_BBLD'].values.reshape(-1, 1)

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_dataset = scaler.fit_transform(dataset)

# Split data into training and testing sets
# We'll train on the first 80% of the data and test on the remaining 20%.
training_size = int(len(scaled_dataset) * 0.80)
train_data = scaled_dataset[0:training_size, :]
test_data = scaled_dataset[training_size - 60:, :] # Include 60 previous days for context

def create_dataset(dataset, time_step=1):
    """
    Converts an array of values into a dataset matrix.
    X = data at time t, t+1, ..., t+time_step-1
    Y = data at time t+time_step
    """
    dataX, dataY = [], []
    for i in range(len(dataset) - time_step - 1):
        a = dataset[i:(i + time_step), 0]
        dataX.append(a)
        dataY.append(dataset[i + time_step, 0])
    return np.array(dataX), np.array(dataY)

# We'll use the last 60 days of data to predict the next day's production.
time_step = 60
X_train, y_train = create_dataset(train_data, time_step)
X_test, y_test = create_dataset(test_data, time_step)

# Reshape input to be [samples, time_steps, features] which is required for LSTM layers
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

print(f"X_train shape: {X_train.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_test shape: {y_test.shape}")


# --- 4. Build and Train the LSTM Model ---
print("\nBuilding the LSTM model...")
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(time_step, 1)))
model.add(Dropout(0.2))
model.add(LSTM(50, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(50))
model.add(Dropout(0.2))
model.add(Dense(1)) # Output layer

model.compile(optimizer='adam', loss='mean_squared_error')
model.summary()

print("\nTraining the model...")
# In a real project, you'd use more epochs and potentially a validation split.
history = model.fit(X_train, y_train, epochs=20, batch_size=64, verbose=1)


# --- 5. Prediction and Evaluation ---
print("\nMaking predictions on test data...")
train_predict = model.predict(X_train)
test_predict = model.predict(X_test)

# Inverse transform to get actual production values
train_predict = scaler.inverse_transform(train_predict)
test_predict = scaler.inverse_transform(test_predict)
y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))

# Calculate RMSE
train_rmse = np.sqrt(mean_squared_error(y_train, scaler.inverse_transform(train_predict)))
test_rmse = np.sqrt(mean_squared_error(y_test_actual, test_predict))
print(f"\nTrain RMSE: {train_rmse:.2f} BBL/D")
print(f"Test RMSE: {test_rmse:.2f} BBL/D")


# --- 6. Visualize Predictions ---
print("Visualizing model predictions...")

# Shift train predictions for plotting
train_predict_plot = np.empty_like(scaled_dataset)
train_predict_plot[:, :] = np.nan
train_predict_plot[time_step:len(train_predict) + time_step, :] = train_predict

# Shift test predictions for plotting
test_predict_plot = np.empty_like(scaled_dataset)
test_predict_plot[:, :] = np.nan
test_predict_plot[len(train_predict) + (time_step * 2) + 1:len(scaled_dataset) - 1, :] = test_predict

# Plot baseline and predictions
plt.figure(figsize=(15, 7))
plt.plot(scaler.inverse_transform(scaled_dataset), label='Actual Production')
plt.plot(train_predict_plot, label='Train Predictions')
plt.plot(test_predict_plot, label='Test Predictions', linewidth=2)
plt.title('Production Forecast vs Actual')
plt.xlabel('Days')
plt.ylabel('Oil Production (BBL/D)')
plt.legend()
plt.show()
