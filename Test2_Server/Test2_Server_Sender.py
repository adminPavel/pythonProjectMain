import socket
import cv2
import time
import numpy as np

server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.bind(('127.0.0.1', 8080))
server_socket.listen(1)
    
path_to_video = 'My_Face_Video.mp4'
cap = cv2.VideoCapture(path_to_video)

while True:
    try:
        client_socket, client_address = server_socket.accept()

    except ConnectionResetError:
        print("Something wierd happened")
        time.sleep(1)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("cap.read() is empty!!!!!!!!!")
            cap = cv2.VideoCapture(path_to_video)
            continue

        bytes_array = cv2.imencode('.jpg', frame)[1].tobytes()

        try:
            print("bytes_array_size", len(bytes_array))
            client_socket.sendall(bytes_array)
        except socket.error:
            print("Client disconnected")
            break