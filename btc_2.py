import yfinance as yf
import pandas as pd
from keras import Sequential, Input
from keras.src.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import matplotlib.pyplot as plt


# Скачивание данных
data = yf.download('BTC-USD', start='2023-12-28', end='2024-05-24', interval='1h')

# Добавление технических индикаторов
data['SMA_10'] = data['Close'].rolling(window=10).mean()
data['SMA_50'] = data['Close'].rolling(window=50).mean()
data = data.dropna()

# Предобработка данных
data = data[['Close', 'SMA_10', 'SMA_50']]
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

# Разделение данных
train_size = int(len(scaled_data) * 0.8)
train_data, test_data = scaled_data[:train_size], scaled_data[train_size:]

# Подготовка данных для LSTM
def create_dataset(dataset, look_back=1):
    X, Y = [], []
    for i in range(len(dataset) - look_back - 1):
        X.append(dataset[i:(i + look_back)])
        Y.append(dataset[i + look_back, 0])
    return np.array(X), np.array(Y)

look_back = 60
X_train, Y_train = create_dataset(train_data, look_back)
X_test, Y_test = create_dataset(test_data, look_back)

# Создание модели
model = Sequential()
model.add(Input(shape=(look_back, X_train.shape[2])))
model.add(LSTM(100, return_sequences=True))
model.add(LSTM(100, return_sequences=True))
model.add(LSTM(50))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')

# Обучение модели
model.fit(X_train, Y_train, epochs=50, batch_size=32, verbose=2)

# Сохранение модели
model.save('btc_price_prediction_model_improved.keras')

# Предсказание
predictions = model.predict(X_test)

# Обратное преобразование только для столбца с закрытием
scaler_close = MinMaxScaler(feature_range=(0, 1))
scaled_close = scaler_close.fit_transform(data[['Close']])
predictions = scaler_close.inverse_transform(predictions)
Y_test_scaled = scaler_close.inverse_transform(Y_test.reshape(-1, 1))

# Оценка модели
mse = np.mean((predictions - Y_test_scaled)**2)
mae = np.mean(np.abs(predictions - Y_test_scaled))

print(f'MSE: {mse}')
print(f'MAE: {mae}')

# График результатов
plt.figure(figsize=(12, 6))
plt.plot(scaler_close.inverse_transform(data[['Close']]), label='Original Data')
plt.plot(range(look_back + 1, look_back + 1 + len(predictions)), predictions, label='Predicted Data', color='red')
plt.title('BTC-USD Price Prediction')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()