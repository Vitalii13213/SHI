import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import RidgeClassifier
from sklearn import metrics

# Завантаження набору даних Iris
iris = load_iris()
X, y = iris.data, iris.target

# Розділення даних на навчальну та тестову вибірки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# Створення та налаштування класифікатора
clf = RidgeClassifier(
    tol=1e-2,  # допустима похибка для зупинки оптимізації
    solver="sag",  # метод стохастичного градієнтного спуску
    alpha=1.0  # параметр регуляризації (контролює складність моделі)
)

# Навчання класифікатора
clf.fit(X_train, y_train)

# Прогнозування
y_pred = clf.predict(X_test)

# Розрахунок показників якості
print('Точність (Accuracy):', np.round(metrics.accuracy_score(y_test, y_pred), 4))
print('Точність (Precision):', np.round(metrics.precision_score(y_test, y_pred, average='weighted'), 4))
print('Повнота (Recall):', np.round(metrics.recall_score(y_test, y_pred, average='weighted'), 4))
print('F1 Score:', np.round(metrics.f1_score(y_test, y_pred, average='weighted'), 4))

# Коефіцієнт Коена Каппа
print('Коефіцієнт Коена Каппа:', np.round(metrics.cohen_kappa_score(y_test, y_pred), 4))

# Коефіцієнт кореляції Метьюза
print('Коефіцієнт кореляції Метьюза:', np.round(metrics.matthews_corrcoef(y_test, y_pred), 4))

# Детальний звіт про класифікацію
print('\nЗвіт про класифікацію:\n', metrics.classification_report(y_test, y_pred))

# Матриця плутанини (Confusion Matrix)
mat = metrics.confusion_matrix(y_test, y_pred)

# Побудова теплової карти матриці плутанини
plt.figure(figsize=(8, 6))
sns.heatmap(mat, annot=True, fmt='d', cmap='Blues')
plt.title('Матриця плутанини')
plt.xlabel('Передбачені мітки')
plt.ylabel('Дійсні мітки')
plt.tight_layout()
plt.savefig("Confusion.jpg")
plt.close()