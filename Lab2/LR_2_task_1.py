import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsOneClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Вхідний файл, який містить дані
input_file = 'income_data.txt'

# Читання даних
X = []
y = []
count_class1 = 0
count_class2 = 0
max_datapoints = 25000

with open(input_file, 'r') as f:
    for line in f.readlines():
        if count_class1 >= max_datapoints and count_class2 >= max_datapoints:
            break
        if '?' in line:
            continue

        data = line[:-1].split(', ')
        if data[-1] == '<=50K' and count_class1 < max_datapoints:
            X.append(data[:-1])
            y.append(0)
            count_class1 += 1
        elif data[-1] == '>50K' and count_class2 < max_datapoints:
            X.append(data[:-1])
            y.append(1)
            count_class2 += 1

# Перетворення на масив numpy
X = np.array(X)
y = np.array(y)

# Перетворення рядкових даних на числові
label_encoders = []
X_encoded = np.empty(X.shape, dtype=object)

for i in range(X.shape[1]):
    if not X[0, i].isdigit():
        label_encoder = LabelEncoder()
        X_encoded[:, i] = label_encoder.fit_transform(X[:, i])
        label_encoders.append(label_encoder)
    else:
        X_encoded[:, i] = X[:, i]

X_encoded = X_encoded.astype(float)

# Розділення даних на навчальну та тестову вибірки
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=5)

# Створення та навчання SVM-класифікатора
classifier = OneVsOneClassifier(LinearSVC(random_state=0))
classifier.fit(X_train, y_train)

# Прогнозування та обчислення метрик
y_pred = classifier.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"Accuracy: {accuracy * 100:.2f}%")
print(f"Precision: {precision * 100:.2f}%")
print(f"Recall: {recall * 100:.2f}%")
print(f"F1 Score: {f1 * 100:.2f}%")

# Тестування класифікатора з новою точкою даних
input_data = ['37', 'Private', '215646', 'HS-grad', '9', 'Never-married', 'Handlers-cleaners', 'Not-in-family', 'White', 'Male', '0', '0', '40', 'United-States']

input_data_encoded = []
for i, item in enumerate(input_data):
    if item.isdigit():
        input_data_encoded.append(float(item))
    else:
        label_encoder_index = i - sum(1 for x in input_data[:i] if x.isdigit())
        if item in label_encoders[label_encoder_index].classes_:
            input_data_encoded.append(label_encoders[label_encoder_index].transform([item])[0])
        else:
            input_data_encoded.append(-1)

input_data_encoded = np.array(input_data_encoded).reshape(1, -1)

predicted_class = classifier.predict(input_data_encoded)
print("Predicted class:", "<=50K" if predicted_class[0] == 0 else ">50K")