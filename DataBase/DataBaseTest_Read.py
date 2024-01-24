import sqlite3

# Подключение к базе данных
conn = sqlite3.connect('DataBase/resource4_data.db')
cursor = conn.cursor()

# Выполнение запроса SELECT
cursor.execute("SELECT * FROM people_and_breast_count")

# Установка размера пакета (количество записей, загружаемых за один раз)
batch_size = 1000

while True:
    # Чтение пакета данных
    rows = cursor.fetchmany(batch_size)

    # Если достигнут конец данных, выход из цикла
    if not rows:
        break

    # Обработка пакета данных
    for row in rows:
        # Вывод данных в терминал или выполнение других операций
        print(row)

# Закрытие соединения с базой данных
conn.close()