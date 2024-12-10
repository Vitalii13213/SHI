import numpy as np
from sklearn import linear_model
import sklearn.metrics as sm
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt

# Завантаження даних
input_file = 'data_multivar_regr.txt'
data = np.loadtxt(input_file, delimiter=',')

# Розділення на вхідні та вихідні змінні
X, y = data[:, :-1], data[:, -1]

# Розбивка даних на навчальний та тестовий набори
num_training = int(0.8 * len(X))
num_test = len(X) - num_training

# Тренувальні дані
X_train, y_train = X[:num_training], y[:num_training]

# Тестові дані
X_test, y_test = X[num_training:], y[num_training:]

# Створення лінійного регресора
linear_regressor = linear_model.LinearRegression()
linear_regressor.fit(X_train, y_train)

# Прогнозування лінійним регресором
y_test_pred_linear = linear_regressor.predict(X_test)

# Метрики лінійного регресора
print("Лінійний регресор:")
print("Середня абсолютна похибка =", round(sm.mean_absolute_error(y_test, y_test_pred_linear), 2))
print("Середньоквадратична похибка =", round(sm.mean_squared_error(y_test, y_test_pred_linear), 2))
print("Медіанна абсолютна похибка =", round(sm.median_absolute_error(y_test, y_test_pred_linear), 2))
print("Показник поясненої дисперсії =", round(sm.explained_variance_score(y_test, y_test_pred_linear), 2))
print("R2 показник =", round(sm.r2_score(y_test, y_test_pred_linear), 2))

# Поліноміальна регресія
polynomial = PolynomialFeatures(degree=10)
X_train_transformed = polynomial.fit_transform(X_train)
X_test_transformed = polynomial.transform(X_test)

# Поліноміальний регресор
poly_linear_model = linear_model.LinearRegression()
poly_linear_model.fit(X_train_transformed, y_train)

# Прогнозування поліноміальним регресором
y_test_pred_poly = poly_linear_model.predict(X_test_transformed)

# Метрики поліноміального регресора
print("\nПоліноміальний регресор:")
print("Середня абсолютна похибка =", round(sm.mean_absolute_error(y_test, y_test_pred_poly), 2))
print("Середньоквадратична похибка =", round(sm.mean_squared_error(y_test, y_test_pred_poly), 2))
print("Медіанна абсолютна похибка =", round(sm.median_absolute_error(y_test, y_test_pred_poly), 2))
print("Показник поясненої дисперсії =", round(sm.explained_variance_score(y_test, y_test_pred_poly), 2))
print("R2 показник =", round(sm.r2_score(y_test, y_test_pred_poly), 2))

# Перевірка на конкретній точці
datapoint = [[7.75, 6.35, 5.56]]
poly_datapoint = polynomial.transform(datapoint)

print("\nПрогнозування для точки", datapoint[0])
print("Лінійна регресія:", round(linear_regressor.predict(datapoint)[0], 2))
print("Поліноміальна регресія:", round(poly_linear_model.predict(poly_datapoint)[0], 2))

# Графічне порівняння результатів
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.scatter(y_test, y_test_pred_linear, color='blue', label='Лінійна регресія')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.title('Лінійна регресія')
plt.xlabel('Реальні значення')
plt.ylabel('Передбачені значення')
plt.legend()

plt.subplot(1, 2, 2)
plt.scatter(y_test, y_test_pred_poly, color='green', label='Поліноміальна регресія')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.title('Поліноміальна регресія')
plt.xlabel('Реальні значення')
plt.ylabel('Передбачені значення')
plt.legend()

plt.tight_layout()
plt.show()