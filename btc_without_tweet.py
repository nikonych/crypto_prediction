import numpy as np
from keras.src.saving import load_model

# Загрузка модели
model = load_model('lstm_bitcoin_model.keras')


def prepare_data_for_prediction(df, n_steps, scaler):
    data = df[['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']].values
    scaled_data = scaler.transform(data)

    def create_input(data, n_steps=1):
        X = []
        for i in range(len(data) - n_steps + 1):
            a = data[i:i + n_steps, :]
            X.append(a)
        return np.array(X)

    X = create_input(scaled_data, n_steps)
    X = X.reshape((X.shape[0], X.shape[1], X.shape[2]))

    return X


def predict_price_on_date(date_str):
    # Преобразуем дату в формат datetime
    target_date = pd.to_datetime(date_str)

    # Получаем последние n_steps дней до текущей даты
    end_idx = len(btcDF) - 1
    start_idx = end_idx - n_steps + 1

    if start_idx < 0:
        raise ValueError(f"Недостаточно данных для прогноза на дату {date_str}.")

    input_data = btcDF.iloc[start_idx:end_idx + 1]
    X_input = prepare_data_for_prediction(input_data, n_steps, scaler)

    # Предсказание
    yhat = model.predict(X_input[-1].reshape(1, n_steps, X_input.shape[2]))
    inv_yhat = np.concatenate((yhat, X_input[-1, :, -5:]), axis=1)  # -5, т.к. у нас 6 столбцов всего
    inv_yhat = scaler.inverse_transform(inv_yhat)
    predicted_price = inv_yhat[:, 3]  # Используем 'Close' (индекс 3) после инверсии

    return predicted_price[0]


# Пример использования функции
date_str = '2024-05-24'
predicted_price = predict_price_on_date(date_str)
print(f'Предсказанная цена биткоина на дату {date_str}: ${predicted_price:.2f}')
