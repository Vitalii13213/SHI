import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split

# Завантаження даних про діабет
diabetes = datasets.load_diabetes()
X = diabetes.data
y = diabetes.target

# Поділ на навчальну та тестову вибірки
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.5, random_state=0)

# Створення та навчання моделі лінійної регресії
regr = linear_model.LinearRegression()
regr.fit(Xtrain, ytrain)

# Прогнозування для тестової вибірки
ypred = regr.predict(Xtest)

# Розрахунок метрик
r2 = r2_score(ytest, ypred)
mae = mean_absolute_error(ytest, ypred)
mse = mean_squared_error(ytest, ypred)

# Друк результатів
print("Коефіцієнти регресії:", regr.coef_)
print("Перетин (intercept):", regr.intercept_)
print("Коефіцієнт R2:", r2)
print("Середня абсолютна помилка (MAE):", mae)
print("Середньоквадратична помилка (MSE):", mse)

# Побудова графіка
plt.figure(figsize=(10, 6))
plt.scatter(ytest, ypred, color='blue', edgecolors=(0, 0, 0), label='Передбачені значення')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2, label='Ідеальна лінія прогнозування')
plt.xlabel('Виміряні значення')
plt.ylabel('Передбачені значення')
plt.title('Лінійна регресія: Виміряні vs Передбачені значення')
plt.legend()
plt.tight_layout()
plt.show()