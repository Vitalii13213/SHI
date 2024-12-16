import numpy as np
import cv2
import pyautogui
import tensorflow as tf
import time
from mss import mss
import ctypes
from concurrent.futures import ThreadPoolExecutor

# Завантаження навченої моделі
model = tf.keras.models.load_model('crystal_model_v3.h5')

# Функція для розпізнавання кристалів з використанням навченої моделі
def detect_crystal(image, model):
    resized_image = cv2.resize(image, (150, 150))
    rgb_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
    expanded_image = np.expand_dims(rgb_image, axis=0)
    predictions = model.predict(expanded_image)
    return predictions[0][0] > 0.5

# Функція для визначення координат кристалів та бомб на зображенні
def find_crystal_and_bomb_coordinates(image):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Визначення діапазону зеленого кольору (для кристалів)
    lower_green = np.array([30, 40, 40])
    upper_green = np.array([90, 255, 255])
    green_mask = cv2.inRange(hsv_image, lower_green, upper_green)

    # Морфологічні операції для очищення маски від шумів
    kernel = np.ones((5, 5), np.uint8)
    green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_CLOSE, kernel)
    green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_OPEN, kernel)

    # Визначення діапазону сірого кольору (для бомб)
    lower_gray = np.array([0, 0, 50])
    upper_gray = np.array([180, 50, 220])
    gray_mask = cv2.inRange(hsv_image, lower_gray, upper_gray)

    # Знаходження контурів на масках
    green_contours, _ = cv2.findContours(green_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    gray_contours, _ = cv2.findContours(gray_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    crystal_coordinates = []
    bomb_coordinates = []

    for contour in green_contours:
        if cv2.contourArea(contour) > 250:  # Фільтрація за розміром, виключення маленьких об'єктів
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                crystal_coordinates.append((cX, cY))

    for contour in gray_contours:
        if cv2.contourArea(contour) > 150:  # Фільтрація за розміром
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                bomb_coordinates.append((cX, cY))

    return crystal_coordinates, bomb_coordinates

def get_selected_area():
    print("Наведіть курсор на верхній лівий кут області та зачекайте 3 секунди...")
    time.sleep(3)
    x1, y1 = pyautogui.position()
    print(f"Верхній лівий кут: ({x1}, {y1})")

    print("Наведіть курсор на нижній правий кут області та зачекайте 3 секунди...")
    time.sleep(3)
    x2, y2 = pyautogui.position()
    print(f"Нижній правий кут: ({x2}, {y2})")

    width = x2 - x1
    height = y2 - y1
    print(f"Ширина: {width}, Висота: {height}")
    return x1, y1, width, height

# Функція для натискання мишкою
def click(x, y):
    ctypes.windll.user32.SetCursorPos(x, y)
    ctypes.windll.user32.mouse_event(2, 0, 0, 0, 0)  # Натискання лівої кнопки миші
    ctypes.windll.user32.mouse_event(4, 0, 0, 0, 0)  # Відпускання лівої кнопки миші

executor = ThreadPoolExecutor(max_workers=10)

def screenshot_and_click(x1, y1, width, height, end_time):
    sct = mss()
    monitor = {"top": y1, "left": x1, "width": width, "height": height}
    clicked_coordinates = []

    while time.time() < end_time:
        current_time = time.time()
        clicked_coordinates = [(coord, timestamp) for coord, timestamp in clicked_coordinates if current_time - timestamp < 1]

        screenshot = sct.grab(monitor)
        frame = np.array(screenshot)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        crystal_coords, bomb_coords = find_crystal_and_bomb_coordinates(frame)

        for coordinates in crystal_coords:
            if coordinates:
                safe_to_click = True
                for bomb in bomb_coords:
                    distance = np.sqrt((coordinates[0] - bomb[0]) ** 2 + (coordinates[1] - bomb[1]) ** 2)
                    if distance < 50:
                        safe_to_click = False
                        break

                if safe_to_click and (coordinates not in [coord for coord, _ in clicked_coordinates]):
                    click_x, click_y = x1 + coordinates[0], y1 + coordinates[1]
                    executor.submit(click, click_x, click_y)
                    clicked_coordinates.append((coordinates, current_time))
                    break

# Отримання координат області екрану, виділеної користувачем
x1, y1, width, height = get_selected_area()

# Встановлення часу завершення роботи програми
end_time = time.time() + 36  # 36 секунд від поточного часу

# Запуск функції знімання скріншотів і натискання на кристали
screenshot_and_click(x1, y1, width, height, end_time)

print("Робота програми завершена через 36 секунди.")
