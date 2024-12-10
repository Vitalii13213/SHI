import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# Генерація даних
np.random.seed(42)
m = 100
X = 6 * np.random.rand(m, 1) - 5
y = 0.5 * X ** 2 + X + 2 + np.random.randn(m, 1)


# Функція для побудови кривих навчання
def plot_learning_curves(model, X, y):
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    train_errors, val_errors = [], []
    for m in range(1, len(X_train) + 1):
        model.fit(X_train[:m], y_train[:m])
        y_train_predict = model.predict(X_train[:m])
        y_val_predict = model.predict(X_val)
        train_errors.append(mean_squared_error(y_train[:m], y_train_predict))
        val_errors.append(mean_squared_error(y_val, y_val_predict))

    plt.plot(np.sqrt(train_errors), "r-+", linewidth=2, label="Train")
    plt.plot(np.sqrt(val_errors), "b-", linewidth=3, label="Validation")
    plt.xlabel("Training set size")
    plt.ylabel("RMSE")
    plt.legend()
    plt.grid()


# Лінійна регресія
lin_reg = LinearRegression()
plt.figure(figsize=(12, 6))
plt.title("Learning Curves for Linear Regression")
plot_learning_curves(lin_reg, X, y)
plt.show()

# Поліноміальна регресія (2-го ступеня)
poly_reg_2 = Pipeline([
    ("poly_features", PolynomialFeatures(degree=2, include_bias=False)),
    ("lin_reg", LinearRegression())
])
plt.figure(figsize=(12, 6))
plt.title("Learning Curves for Polynomial Regression (degree=2)")
plot_learning_curves(poly_reg_2, X, y)
plt.show()

# Поліноміальна регресія (10-го ступеня)
poly_reg_10 = Pipeline([
    ("poly_features", PolynomialFeatures(degree=10, include_bias=False)),
    ("lin_reg", LinearRegression())
])
plt.figure(figsize=(12, 6))
plt.title("Learning Curves for Polynomial Regression (degree=10)")
plot_learning_curves(poly_reg_10, X, y)
plt.show()