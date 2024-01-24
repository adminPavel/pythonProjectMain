import socket
import cv2
import time
import numpy as np
import struct

server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.bind(('127.0.0.1', 42514))
server_socket.listen(1)

while True:
    try:
        client_socket, client_address = server_socket.accept()

    except ConnectionResetError:
        print("Something wierd happened")
        time.sleep(1)
        continue
    
    while True:
        try:
            # Принимаем размер фрейма
            size_data = client_socket.recv(4)
            if not size_data:
                time.sleep(1)
                break
            size = struct.unpack("!I", size_data)[0]

            # Принимаем фрейм
            data = b''
            while len(data) < size:
                packet = client_socket.recv(size - len(data))
                if not packet:
                    time.sleep(1)
                    break
                data += packet

            frame = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)

            if frame is None:
                continue

            print("Socket data size:", len(data))
            print("Socket frame size:", len(frame))

            cv2.imshow('frame', frame)

            if cv2.waitKey(1) == 27:
                client_socket.close()
                break

        except ConnectionResetError:
            print("Server is not responding for some time already")
            time.sleep(1)
            client_socket.close()
            break

    client_socket.close()
