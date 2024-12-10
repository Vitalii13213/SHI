import pickle
import numpy as np
from sklearn import linear_model
import sklearn.metrics as sm
import matplotlib.pyplot as plt

# Вхідний файл, який містить дані
input_file = 'data_singlevar_regr.txt'

# Завантаження даних
data = np.loadtxt(input_file, delimiter=',')
X, y = data[:, :-1], data[:, -1]

# Розбивка даних на навчальний та тестовий набори
num_training = int(0.8 * len(X))
num_test = len(X) - num_training

# Тренувальні дані
X_train, y_train = X[:num_training], y[:num_training]

# Тестові дані
X_test, y_test = X[num_training:], y[num_training:]

# Створення об'єкта лінійного регресора
regressor = linear_model.LinearRegression()
regressor.fit(X_train, y_train)

# Прогнозування результату
y_test_pred = regressor.predict(X_test)

# Побудова графіка
plt.figure(figsize=(10, 6))
plt.scatter(X_test, y_test, color='green', label='Реальні дані')
plt.plot(X_test, y_test_pred, color='black', linewidth=4, label='Передбачені значення')
plt.title('Лінійна регресія: Реальні vs Передбачені значення')
plt.xlabel('Вхідна змінна')
plt.ylabel('Цільова змінна')
plt.legend()
plt.show()

# Обчислення метричних параметрів регресора
print("Показники продуктивності лінійного регресора:")
print("Середня абсолютна похибка =", round(sm.mean_absolute_error(y_test, y_test_pred), 2))
print("Середньоквадратична похибка =", round(sm.mean_squared_error(y_test, y_test_pred), 2))
print("Медіанна абсолютна похибка =", round(sm.median_absolute_error(y_test, y_test_pred), 2))
print("Показник поясненої дисперсії =", round(sm.explained_variance_score(y_test, y_test_pred), 2))
print("R2 показник =", round(sm.r2_score(y_test, y_test_pred), 2))

# Збереження моделі
output_model_file = 'model.pkl'
with open(output_model_file, 'wb') as f:
    pickle.dump(regressor, f)

# Завантаження моделі
with open(output_model_file, 'rb') as f:
    regressor_model = pickle.load(f)

# Прогнозування з використанням завантаженої моделі
y_test_pred_new = regressor_model.predict(X_test)
print("\nНова середня абсолютна похибка =", round(sm.mean_absolute_error(y_test, y_test_pred_new), 2))