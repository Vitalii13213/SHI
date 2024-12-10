import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.datasets import make_classification

# Генерація синтетичного набору даних
X, y = make_classification(n_samples=1000, n_features=20,
                           n_informative=10, n_redundant=5,
                           random_state=42)

# Розділення даних на навчальну та тестову вибірки
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42)

# Стандартизація даних
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Створення SVM з поліноміальним ядром
poly_svm = SVC(kernel='poly', degree=8)
poly_svm.fit(X_train_scaled, y_train)

# Прогнозування
y_pred = poly_svm.predict(X_test_scaled)

# Оцінка якості класифікації
print("Результати класифікації (Поліноміальне ядро):")
print(classification_report(y_test, y_pred))

# Матриця плутанини
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Матриця плутанини (Поліноміальне ядро)')
plt.colorbar()
plt.xlabel('Передбачені класи')
plt.ylabel('Реальні класи')
plt.tight_layout()
plt.savefig('polynomial_svm_confusion_matrix.png')
plt.close()