import pickle
import numpy as np
from sklearn import linear_model
import sklearn.metrics as sm
import matplotlib.pyplot as plt

# Створення вхідного файлу з наданими даними
input_data = """
-0.86,4.38
2.58,6.97
4.17,7.01
2.6,5.44
5.13,6.45
3.23,5.49
-0.26,4.25
2.76,5.94
0.47,4.8
-3.9,2.7
0.27,3.26
2.88,6.48
-0.54,4.08
-4.39,0.09
-1.12,2.74
2.09,5.8
-5.78,0.16
1.77,4.97
-7.91,-2.26
4.86,5.75
-2.17,3.33
1.38,5.26
0.54,4.43
3.12,6.6
-2.19,3.77
-0.33,2.4
-1.21,2.98
-4.52,0.29
-0.46,2.47
-1.13,4.08
4.61,8.97
0.31,3.94
0.25,3.46
-2.67,2.46
-4.66,1.14
-0.2,4.31
-0.52,1.97
1.24,4.83
-2.53,3.12
-0.34,4.97
5.74,8.65
-0.34,3.59
0.99,3.66
5.01,7.54
-2.38,1.52
-0.56,4.55
-1.01,3.23
0.47,4.39
4.81,7.04
2.38,4.46
-3.32,2.41
-3.86,1.11
-1.41,3.23
6.04,7.46
4.18,5.71
-0.78,3.59
-2.2,2.93
0.76,4.16
2.02,6.43
0.42,4.92
"""

# Збереження даних у текстовий файл
with open('data_singlevar_regr2.txt', 'w') as f:
    f.write(input_data.strip())

# Завантаження даних
data = np.loadtxt('data_singlevar_regr2.txt', delimiter=',')
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
plt.figure(figsize=(12, 8))
plt.scatter(X_test, y_test, color='green', label='Реальні дані')
plt.plot(X_test, y_test_pred, color='red', linewidth=4, label='Лінія регресії')
plt.title('Лінійна регресія: Реальні vs Передбачені значення')
plt.xlabel('Вхідна змінна X')
plt.ylabel('Цільова змінна Y')
plt.legend()
plt.grid(True)
plt.show()

# Обчислення метричних параметрів регресора
print("Показники продуктивності лінійного регресора:")
print("Середня абсолютна похибка =", round(sm.mean_absolute_error(y_test, y_test_pred), 2))
print("Середньоквадратична похибка =", round(sm.mean_squared_error(y_test, y_test_pred), 2))
print("Медіанна абсолютна похибка =", round(sm.median_absolute_error(y_test, y_test_pred), 2))
print("Показник поясненої дисперсії =", round(sm.explained_variance_score(y_test, y_test_pred), 2))
print("R2 показник =", round(sm.r2_score(y_test, y_test_pred), 2))

# Додаткова інформація про модель
print("\nПараметри лінійної регресії:")
print("Коефіцієнт нахилу (вага) =", round(regressor.coef_[0], 2))
print("Точка перетину =", round(regressor.intercept_, 2))

# Збереження моделі
output_model_file = 'model_task2.pkl'
with open(output_model_file, 'wb') as f:
    pickle.dump(regressor, f)

# Демонстрація прогнозування
print("\nПрогнозування для кількох вхідних значень:")
test_inputs = np.array([[-1.0], [0.5], [3.0]])
predictions = regressor.predict(test_inputs)
for input_val, prediction in zip(test_inputs, predictions):
    print(f"Для X = {input_val[0]}, прогнозоване Y = {round(prediction, 2)}")