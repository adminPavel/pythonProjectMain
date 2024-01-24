import socket
import cv2
import struct
import time

path_to_video = 'My_Face_Video.mp4'
cap = cv2.VideoCapture(path_to_video)

while True:
    try:
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client_socket.connect(('194.87.252.140', 42514))
        # client_socket.connect(('127.0.0.1', 42514))
        client_socket.settimeout(10)
        # client_socket.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 102400)
        # client_socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    except ConnectionRefusedError:
        print("Server is not found in the network")
        time.sleep(1)
        continue
        continue
    except socket.timeout:
        print("Connection attempt timed out.")
        time.sleep(1)
        continue
    except TimeoutError:
        print("Could not connect to the server. Please try again.")
        time.sleep(1)

    while True:
        ret, frame = cap.read()

        if not ret:
            # print("cap.read() is empty!!!!!!!!!")
            cap = cv2.VideoCapture(path_to_video)
            continue

        frame = cv2.resize(frame, (0, 0), fx=0.3, fy=0.3, interpolation=cv2.INTER_LINEAR)
        frame_bytes = cv2.imencode('.jpg', frame)[1].tobytes()
        
        # Отправляем размер фрейма перед самим фреймом
        size = struct.pack("!I", len(frame_bytes))
        try:
            client_socket.sendall(size)
            client_socket.sendall(frame_bytes)
            
            buffer_size = client_socket.getsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF)
            print("Socket Buffer size:", buffer_size)
            print("Real Buffer size:", len(frame_bytes))
            
            # time.sleep(0.01)
        except socket.error:
            print("Client disconnected")
            break
