import matplotlib.pyplot as plt
import numpy as np

# Визначення функцій
def or_gate(x1, x2):
    return x1 | x2

def and_gate(x1, x2):
    return x1 & x2

def xor_gate(x1, x2):
    return or_gate(x1, x2) & ~and_gate(x1, x2)

# Створення координатної сітки
x1 = np.array([0, 0, 1, 1])
x2 = np.array([0, 1, 0, 1])
y = np.array([xor_gate(a, b) for a, b in zip(x1, x2)])

# Налаштування графіку
plt.figure(figsize=(8, 6))

# Розділення за класами
for label in [0, 1]:
    plt.scatter(x1[y == label], x2[y == label], label=f'{label}', s=200,
                color='blue' if label == 0 else 'red')

plt.title('XOR Function Visualization', fontsize=15)
plt.xlabel('x1', fontsize=12)
plt.ylabel('x2', fontsize=12)
plt.xticks([0, 1])
plt.yticks([0, 1])
plt.grid(True, linestyle='--', linewidth=0.5)
plt.legend()

plt.tight_layout()
plt.show()
