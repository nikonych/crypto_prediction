import random

import yfinance as yf
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Загрузка данных
data = yf.download('BTC-USD', '2024-05-23', '2024-05-24', interval='1h')


# Подготовка данных
data['Date'] = data.index
data['Target'] = data['Close'].shift(-1)

# Удаление последних строк с NaN значениями
data = data[:-1]

# Выбор фичей и таргета
X = data[['Close']]
y = data['Target']

# Разделение данных на тренировочную и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Обучение модели
model = LinearRegression()
model.fit(X_train, y_train)

# Предсказание
predictions = model.predict(X_test)

# Оценка модели
plt.figure(figsize=(14,7))
plt.plot(y_test.values, label='Actual Price')
plt.plot(predictions, label='Predicted Price')
plt.legend()
plt.show()

# Предсказание цены на завтра
last_close = data.iloc[-1]['Close']
predicted_price = model.predict([[last_close]])
print(f"Предсказанная цена BTC-USD на завтра: {predicted_price[0]}")


# Функция для предсказания на несколько дней вперед
def predict_future_prices(model, last_close, days_ahead):
    future_prices = []
    current_price = last_close

    for _ in range(days_ahead):
        next_price = model.predict([[current_price]])[0]
        future_prices.append(next_price)
        current_price = next_price

    return future_prices


# Предсказание цен на следующие 7 дней
last_close = data.iloc[-1]['Close']
days_ahead = 7
future_prices = predict_future_prices(model, last_close, days_ahead)

# Отображение предсказанных цен
print(f"Предсказанные цены BTC-USD на следующие {days_ahead} дней: {future_prices}")

# График предсказанных цен
plt.figure(figsize=(14, 7))
dates = pd.date_range(start=data.index[-1], periods=days_ahead + 1)[1:]
plt.plot(dates, future_prices, marker='o', label='Predicted Future Prices')
plt.title('Predicted BTC-USD Prices for Next 7 Days')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()


# Оценка точности модели на случайных днях
num_random_days = 10
random_days = random.sample(range(len(X_test)), num_random_days)
actual_prices = y_test.iloc[random_days].values
predicted_prices = predictions[random_days]

# Расчет метрик
mse = mean_squared_error(actual_prices, predicted_prices)
mae = mean_absolute_error(actual_prices, predicted_prices)

print(f'Mean Squared Error (MSE): {mse}')
print(f'Mean Absolute Error (MAE): {mae}')

# График фактических и предсказанных цен для случайных дней
plt.figure(figsize=(14,7))
plt.plot(actual_prices, marker='o', label='Actual Prices')
plt.plot(predicted_prices, marker='x', label='Predicted Prices')
plt.title('Actual vs Predicted Prices on Random Days')
plt.xlabel('Random Day Index')
plt.ylabel('Price')
plt.legend()
plt.show()