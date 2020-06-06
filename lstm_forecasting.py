from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.metrics import mean_squared_error

timesteps = window_size-1
n_features = 1

model = Sequential()
model.add(LSTM(16, activation='relu', input_shape=(timesteps, n_features), return_sequences=True))
model.add(LSTM(16, activation='relu'))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

model.fit(X_train, y_train, epochs=30, batch_size=32)
y_pred = model.predict(X_test)
print("MSE:", mean_squared_error(y_test, y_pred))
