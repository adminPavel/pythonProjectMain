import socket
import cv2
import numpy as np
import time

while True:
    try:
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client_socket.connect(('127.0.0.1', 8080))
        client_socket.settimeout(10)

    except ConnectionRefusedError:
        print("Server is not found in the network")
        time.sleep(1)
        continue
        
    while True:
        try:
            data = client_socket.recv(1024000)
            if not data:
                # print("break")
                time.sleep(1)
                break

            frame = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)

            if frame is None:
                continue

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

