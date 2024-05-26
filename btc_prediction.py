import numpy as np
from keras.src.saving import load_model

# Загрузка модели из файла
model = load_model("lstm_code_trading_model.keras")

# Прогноз на следующий день
def predict_next_day(model, last_sequence):
    prediction = model.predict(last_sequence)
    return prediction

# Прогноз на неделю вперёд
def predict_next_week(model, last_sequence, days=7):
    predictions = []
    current_sequence = last_sequence
    for _ in range(days):
        prediction = model.predict(current_sequence)
        predictions.append(prediction)
        current_sequence = np.append(current_sequence[:, 1:, :], [[prediction]], axis=1)
    return predictions

# Пример использования
# Предположим, что `last_sequence` - это последний набор данных, который использовался для прогнозирования
last_sequence = np.array([X_test[-1]])  # Пример: последний набор данных тестовой выборки

# Прогноз на следующий день
next_day_prediction = predict_next_day(model, last_sequence)
print("Прогноз на следующий день:", next_day_prediction)

# Прогноз на неделю вперёд
next_week_predictions = predict_next_week(model, last_sequence, days=7)
print("Прогноз на неделю вперёд:", next_week_predictions)
