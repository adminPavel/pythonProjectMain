import sqlite3
import matplotlib.pyplot as plt
import numpy as np

# Подключение к базе данных
conn = sqlite3.connect('DataBase/resource1_data.db')
cursor = conn.cursor()

# Выполнение запроса SELECT
cursor.execute("SELECT * FROM people_count")

# Получение результатов запроса
rows = cursor.fetchall()

# Извлечение дат, времени и количества из записей
dates = [row[0] for row in rows]
times = [row[1] for row in rows]
counts = [row[2] for row in rows]

# Преобразование дат в числовой формат для оси X
x_values = np.arange(len(dates))

# Создание графика
plt.plot(x_values, counts)
plt.xlabel('Дата')
plt.ylabel('Количество')
plt.title('График количества людей на рабочем месте линия филе НПК')

# Установка подписей оси X
x_ticks = np.arange(0, len(dates), 7)  # Подписи каждую неделю
x_labels = [dates[i] for i in x_ticks]  # Подписи соответствующих дат
plt.xticks(x_ticks, x_labels, rotation='vertical')

plt.tight_layout()

# Показ графика
plt.show()

# Закрытие соединения с базой данных
conn.close()