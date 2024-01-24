import sqlite3
from datetime import datetime
import time

# Подключение к базе данных
conn = sqlite3.connect('DataBase/resource1_data.db')
cursor = conn.cursor()
cursor.execute('''CREATE TABLE IF NOT EXISTS people_count
                  (date DATE, time TEXT, count INTEGER)''')

# Запись данных
count = 0  # Начальное значение счётчика
while True:
    # Получение текущей даты и времени
    current_time = datetime.now()
    date = current_time.date()
    our_time = current_time.time().strftime('%H:%M:%S')  # Преобразование времени в строку

    # Увеличение счётчика
    count += 1

    # Вставка записи в таблицу
    cursor.execute("INSERT INTO people_count VALUES (?, ?, ?)", (date, our_time, count))

    # Коммит изменений
    conn.commit()

    print("INSERT INTO people_count VALUES (?, ?, ?)", (date, our_time, count))

    # Задержка для достижения нужной частоты записи
    time.sleep(0.05)  # Задержка в 1/20 секунды

# Закрытие соединения с базой данных
conn.close()