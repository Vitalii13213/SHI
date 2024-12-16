import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# Генеруємо дані
np.random.seed(42)  # Для відтворюваності
m = 100
X = 6 * np.random.rand(m, 1) - 5
y = 0.5 * X**2 + X + 2 + np.random.randn(m, 1)

# Лінійна регресія
lin_reg = LinearRegression()
lin_reg.fit(X, y)

# Побудова графіка лінійної регресії
X_new = np.linspace(X.min(), X.max(), 500).reshape(-1, 1)
y_lin_pred = lin_reg.predict(X_new)

plt.scatter(X, y, color="blue", label="Дані")
plt.plot(X_new, y_lin_pred, color="red", label="Лінійна регресія")
plt.title("Лінійна регресія")
plt.xlabel("X")
plt.ylabel("y")
plt.legend()
plt.show()

# Поліноміальна регресія (2-го ступеня)
poly_features = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly_features.fit_transform(X)

lin_reg_poly = LinearRegression()
lin_reg_poly.fit(X_poly, y)

# Графік поліноміальної регресії
X_poly_new = poly_features.transform(X_new)
y_poly_pred = lin_reg_poly.predict(X_poly_new)

plt.scatter(X, y, color="blue", label="Дані")
plt.plot(X_new, y_poly_pred, color="green", label="Поліноміальна регресія (2-го ступеня)")
plt.title("Поліноміальна регресія")
plt.xlabel("X")
plt.ylabel("y")
plt.legend()
plt.show()

# Виведення коефіцієнтів
print("Лінійна регресія:")
print(f"Перехоплення: {lin_reg.intercept_}, Коефіцієнти: {lin_reg.coef_}")

print("\nПоліноміальна регресія:")
print(f"Перехоплення: {lin_reg_poly.intercept_}, Коефіцієнти: {lin_reg_poly.coef_}")
