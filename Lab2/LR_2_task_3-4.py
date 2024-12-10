import numpy as np
import pandas as pd
import matplotlib.pyplot as pyplot
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.multiclass import OneVsRestClassifier


def main():
    # КРОК 1: Завантаження датасету
    iris = load_iris()
    X = iris.data
    y = iris.target
    feature_names = iris.feature_names
    target_names = iris.target_names

    print("КРОК 1: Завантаження датасету")
    print("Розмірність датасету:", X.shape)
    print("Назви ознак:", feature_names)
    print("Назви класів:", target_names)
    print("\nПерші 5 зразків:")
    for i in range(5):
        print(f"Зразок {i + 1}: {X[i]} - Клас: {target_names[y[i]]}")

    # КРОК 2: Візуалізація даних
    print("\nКРОК 2: Візуалізація даних")
    pyplot.figure(figsize=(15, 5))

    # Діаграма розмаху
    pyplot.subplot(131)
    pyplot.boxplot(X)
    pyplot.title('Діаграма розмаху')
    pyplot.xticks(range(1, 5), feature_names, rotation=45)

    # Гістограма розподілу
    pyplot.subplot(132)
    for i in range(4):
        pyplot.hist(X[:, i], alpha=0.5, label=feature_names[i])
    pyplot.title('Гістограма розподілу')
    pyplot.legend()

    # Матриця діаграм розсіювання
    pyplot.subplot(133)
    pyplot.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis')
    pyplot.title('Діаграма розсіювання')
    pyplot.xlabel(feature_names[0])
    pyplot.ylabel(feature_names[1])

    pyplot.tight_layout()
    pyplot.show()

    # КРОК 3: Підготовка даних
    print("\nКРОК 3: Підготовка даних")
    # Розділення на навчальну та тестову вибірки
    X_train, X_validation, Y_train, Y_validation = train_test_split(
        X, y, test_size=0.20, random_state=1
    )
    print("Розмір навчальної вибірки:", X_train.shape)
    print("Розмір тестової вибірки:", X_validation.shape)

    # КРОК 4: Вибір та налаштування моделей
    print("\nКРОК 4: Вибір та налаштування моделей")
    models = []
    models.append(('LR', Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', OneVsRestClassifier(LogisticRegression(solver='liblinear')))
    ])))
    models.append(('LDA', LinearDiscriminantAnalysis()))
    models.append(('KNN', Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', KNeighborsClassifier())
    ])))
    models.append(('CART', DecisionTreeClassifier()))
    models.append(('NB', GaussianNB()))
    models.append(('SVM', Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', SVC(gamma='auto'))
    ])))

    # КРОК 5: Оцінка моделей
    print("\nКРОК 5: Оцінка моделей")
    results = []
    names = []
    for name, model in models:
        kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
        cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
        results.append(cv_results)
        names.append(name)
        print(f'{name}: {cv_results.mean():.4f} ({cv_results.std():.4f})')

    # Порівняння алгоритмів
    pyplot.figure(figsize=(10, 6))
    pyplot.boxplot(results, labels=names)
    pyplot.title('Порівняння алгоритмів')
    pyplot.ylabel('Точність')
    pyplot.show()

    # КРОК 6: Вибір найкращої моделі
    print("\nКРОК 6: Вибір найкращої моделі")
    best_model = models[1][1]  # LDA показала найкращі результати
    best_model.fit(X_train, Y_train)

    # КРОК 7: Прогнозування
    print("\nКРОК 7: Прогнозування")
    predictions = best_model.predict(X_validation)

    print("Точність моделі:", accuracy_score(Y_validation, predictions))
    print("\nМатриця помилок:")
    print(confusion_matrix(Y_validation, predictions))
    print("\nЗвіт класифікації:")
    print(classification_report(Y_validation, predictions))

    # КРОК 8: Прогноз для нового зразка
    print("\nКРОК 8: Прогноз для нового зразка")
    X_new = np.array([[5.0, 2.9, 1.0, 0.2]])
    prediction = best_model.predict(X_new)

    print("Новий зразок:", X_new[0])
    print("Прогнозований клас:", target_names[prediction[0]])


if __name__ == "__main__":
    main()