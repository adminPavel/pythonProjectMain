import cv2
from ultralytics import YOLO
import supervision as sv
from supervision import Point
import numpy as np
from supervision.draw.color import Color
import time
from ultralytics.utils import ROOT, yaml_load
from ultralytics.utils.checks import check_yaml
import multiprocessing as mp
from multiprocessing import Value
import platform
from shapely.geometry import Polygon
import openpyxl
import datetime
import atexit
import sqlite3
import torch
from torchvision.ops.boxes import box_convert
from PIL import Image
from flask import Flask, Response
import socket
import struct
import math
import sys
import pickle
import torchvision.transforms as T
import os

print("Path at terminal when executing this file")
print(os.getcwd() + "\n")

CLASSES = yaml_load(check_yaml('coco128.yaml'))['names']
colors = np.random.uniform(0, 255, size=(len(CLASSES), 3))
operating_system = platform.system()
path = "Media/My_Face_Video3.mp4"#0#"Media/"cars(720p).mp4

stream = path
# stream = "Media\Picture.jpg"
#stream = 0
stream_from_file = True

data_base_name = 'DataBase/resource4_data.db'

window_name = "Face_Recognition"

LOGGER_FRAME_READ_AND_RESIZE = False
LOGGER_YOLO_DETECTIONS = False
LOGGER_YOLO_CLASSIFICATIONS = False
LOGGER_DISPLAY_WINDOW = False
LOGGER_FRAME_WRITE = False

def preprocess_image(image_bgr: np.ndarray) -> torch.Tensor:
    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image_pillow = Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
    image_transformed, _ = transform(image_pillow, None)
    return image_transformed

def draw_bounding_box(img, class_id, confidence, bbox):
    label = f'{CLASSES[class_id]} ({str(class_id)}) ({confidence:.2f})'
    (x, y, x2, y2) = bbox
    color = colors[class_id]
    cv2.rectangle(img, (x, y), (x2, y2), color, 2)
    cv2.putText(img, label, (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

def Frame_Read_and_Resize(original_frames_queue: mp.Queue, frames_queue: mp.Queue, video_recording_exist: Value, stream: str, target_resolution_for_model: int, target_crop, program_must_stop: Value, fps: Value):
    print("Process Frame_Read_and_Resize started")
    cap = cv2.VideoCapture(path)#"Media\Wedding.mp4"
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps.value = int(cap.get(cv2.CAP_PROP_FPS))
    while not frame_height:
        cap = cv2.VideoCapture(path)#"Media\Wedding.mp4"
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps.value = int(cap.get(cv2.CAP_PROP_FPS))
        print("frame_width:", frame_width)
        print("frame_height:", frame_height)

    attempts_counter = 0
    frame_counter = 0 # for debug
    if LOGGER_FRAME_READ_AND_RESIZE:
        time_of_model_processing = time.time()
        average_fps = 0
        fps_list_lenght = 30
        fps_list = [0.0] * fps_list_lenght
    while True:
        ret, frame = cap.read()
        if not ret:
            # if not stream_from_file:
            if True:
                # cv2.imshow("Stream1", original_frame)
                if attempts_counter > 3:
                    #program_must_stop.value = 1 # При чтении с файла это фигня
                    # cap.release()
                    # break
                    print("Renew VIDEO stream")
                    cap = cv2.VideoCapture(path)
                    attempts_counter = 0
                    pass
                attempts_counter = attempts_counter + 1
                print("Frame was not retrived ret. We sleep 1 sec now:", ret, attempts_counter)
                time.sleep(1)
            else:
                pass

        else:
            frame_counter = frame_counter + 1
            if not stream_from_file:
            # if video_recording_exist.value:
                if True:
                    if not original_frames_queue.full():
                        original_frames_queue.put(frame)

                if not frames_queue.full():
                    if target_resolution_for_model:
                        frame = cv2.resize(frame, (target_resolution_for_model, int(target_resolution_for_model / (frame_width/frame_height))))

                    frames_queue.put(frame)
            else:
                frames_queue.put(frame)

            # ================== Подсчёт среднего значения петли
            if LOGGER_FRAME_READ_AND_RESIZE:
                if len(fps_list) >= fps_list_lenght:
                    fps_list.pop(0)
                time_difference = time.time() - time_of_model_processing
                time_of_model_processing = time.time()
                if time_difference != 0:
                    fps_list.append(1000 / (time_difference * 1000))
                average_fps = sum(fps_list) / len(fps_list)
                print('=================================================')
                print('frame_counter:', frame_counter)
                print('Frame Width:', frame_width)
                print('Frame Height:', frame_height) #int(640 / (frame_width/frame_height)))
                print('Frame_Read_and_Resize Loop time:', round(time.time() - time_of_model_processing, 3))
                print('FPS:', round(average_fps, 3))
        
        if program_must_stop.value == 1:
            cap.release()
            print("Frame_Read_and_Resize task DONE")
            break

def YoLo_Detections(model_path: str, frames_queue: mp.Queue, frames_queue_after_detection: mp.Queue, target_crop, number_of_people_queue, program_must_stop: Value):
    print("Process YoLo_Detections started")

    if LOGGER_YOLO_DETECTIONS:
        time_of_model_processing = time.time()
        average_fps = 0
        fps_list_lenght = 30
        fps_list = [0.0] * fps_list_lenght

    frame_width = 1280
    file_write_period = 1 # seconds
    people_counter_lenght = 20
    people_counter_list = [0] * people_counter_lenght

    box_annotator = sv.BoxAnnotator(
        # color=Color.white(),
        text_color=Color.black(),
        thickness=int(frame_width/384), # int(3840/384)   10
        text_thickness=int(frame_width/1280), #int(3840/1280)   3
        text_scale=frame_width/3840, #int(3840/3840)   1.0
        text_padding=int(frame_width/350)
    )

    model = YOLO(model_path)
    while True:
        # if not frames_queue.empty():
        if True:
            frame = frames_queue.get(block=True)
            frame_height, frame_width, frame_channels = frame.shape
            # Use model.track or model.predict
            for result in model.predict(source=frame, conf=0.35, iou=0.6, stream=True, device="cpu", agnostic_nms=True, verbose=False): #, persist=True, verbose=False, device="mps", device="cpu", device="cuda", show=True, imgsz=frame_width
                
                detections = sv.Detections.from_ultralytics(result)

                # detections = detections[(detections.class_id == 0)]
                # detections = detections[detections.confidence > 0.3]

                if result.boxes.id is not None:
                    detections.tracker_id = result.boxes.id.cpu().numpy().astype(int)

                labels = [
                    # f"{CLASSES[class_id]} {tracker_id} ({confidence:0.2f})"
                    # f"{CLASSES[class_id]} {confidence:0.2f}"
                    f"EBALO {confidence:0.2f}"
                    for xyxy, mask, confidence, class_id, tracker_id
                    in detections
                ]
                frame = box_annotator.annotate(
                    scene=frame, 
                    detections=detections,
                    labels=labels
                )

                for xyxy, mask, confidence, class_id, tracker_id in detections:
                    frame =  frame[int(xyxy[1]):int(xyxy[3]), int(xyxy[0]):int(xyxy[2])]

                frames_queue_after_detection.put(frame)

            if LOGGER_YOLO_DETECTIONS:
                if len(fps_list) >= fps_list_lenght:
                    fps_list.pop(0)
                time_difference = time.time() - time_of_model_processing
                time_of_model_processing = time.time()
                if time_difference != 0:
                    fps_list.append(1000 / (time_difference * 1000))
                average_fps = sum(fps_list) / len(fps_list)
                print('=================================================')
                print('Frame Width:', frame_width)
                print('Frame Height:', frame_height)
                print('YoLo_Detections Loop time:', round(time_difference, 3))
                print('FPS:', round(average_fps, 3))

        else:
            print('no frame received for detections')

        if program_must_stop.value == 1:  
            print("YoLo_Detections task DONE")
            break

def YoLo_Emotions_Classifications(model_path: str, frames_queue: mp.Queue, frames_queue_after_detection2: mp.Queue, target_crop, emotions_value: mp.Queue, program_must_stop: Value):
    print("Process YoLo_Emotions_Classifications started")

    if LOGGER_YOLO_CLASSIFICATIONS:
        time_of_model_processing = time.time()
        average_fps = 0
        fps_list_lenght = 30
        fps_list = [0.0] * fps_list_lenght


    model = YOLO(model_path)
    while True:
        if not frames_queue.empty():
            frame = frames_queue.get()
            if not np.any(frame):
                continue
            frame_height, frame_width, frame_channels = frame.shape

            #print(frame)
            #print(type(frame))
            result=model(source=frame)
            # 0: 224x224 Neutral 0.61, Happy 0.24, Sad 0.16, 0.0ms
            # Process results list
            Neutral = 0.0
            Happy = 0.0
            Sad = 0.0
            for single_result in result:
                for k in range(3):
                    emo=single_result.names[single_result.probs.top5[k]]
                    conf=single_result.probs.top5conf[k].item()
                    if emo == 'Neutral':
                        Neutral = conf
                    elif emo == 'Happy':
                        Happy = conf
                    elif emo == 'Sad':
                        Sad = conf

            if not emotions_value.full():
                emotions_value.put([Neutral, Happy, Sad])

            # print(sys.getsizeof(Neutral))
            # print(sys.getsizeof(Happy))
            # print(sys.getsizeof(Sad))
        
            frames_queue_after_detection2.put(frame)

            if LOGGER_YOLO_CLASSIFICATIONS:
                if len(fps_list) >= fps_list_lenght:
                    fps_list.pop(0)
                time_difference = time.time() - time_of_model_processing
                time_of_model_processing = time.time()
                if time_difference != 0:
                    fps_list.append(1000 / (time_difference * 1000))
                average_fps = sum(fps_list) / len(fps_list)
                print('=================================================')
                print('Frame Width:', frame_width)
                print('Frame Height:', frame_height)
                print('YoLo_Detections Loop time:', round(time_difference, 3))
                print('FPS:', round(average_fps, 3))
                print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++")
                print('Neutral:', Neutral)
                print('Happy:', Happy)
                print('Sad:', Sad)
                print("-------------------------------------------------------")

        if program_must_stop.value == 1:
            break

def Display_Window(frame_in: mp.Queue, frame_out: mp.Queue, video_recording_exist: Value, program_must_stop: Value, fps: Value):
    print("Process Display_Window started")
    while True:
        if not frame_in.empty():
            frame1 = frame_in.get()
            frame_height, frame_width, frame_channels = frame1.shape
            break
    frame1 = None
    time_difference = 0
    time_of_display_loop_processing = time.time()
    average_fps = 0
    fps_list_lenght = 50
    fps_list = [0.0] * fps_list_lenght
    window_target_size = frame_width
    while True:
        if not frame_in.empty():
        #if True:
            frame1 = frame_in.get(block=True)
            if video_recording_exist.value:
                if not frame_out.full():
                    frame_out.put(frame1)
                else:
                    print("WARNING frames_queue_after_detection2")

            cv2.imshow(window_name, frame1)
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(window_name, window_target_size, int(window_target_size * (frame_height/frame_width)))
            
            if operating_system == "Darwin": cv2.moveWindow(window_name, 0, -69)
            else: cv2.moveWindow(window_name, 0, 0)
            
            if (cv2.waitKey(1) == 27):
                print("Display_Window task DONE1")
                cv2.destroyAllWindows()
                program_must_stop.value = 1
                break

            # ================== Подсчёт среднего значения петли
            if len(fps_list) >= fps_list_lenght:
                fps_list.pop(0)
            time_difference = time.time() - time_of_display_loop_processing
            time_of_display_loop_processing = time.time()
            if LOGGER_DISPLAY_WINDOW:
                if time_difference != 0:
                    fps_list.append(1000 / (time_difference * 1000))
                average_fps = sum(fps_list) / len(fps_list)
                print('=================================================')
                print('Frame Width:', frame_width)
                print('Frame Height:', frame_height)
                print('Display_Window Loop time:', round(time_difference, 3))
                print('FPS:', round(average_fps, 3))
            needed_sleep_time = (1000/fps.value/1000) - time_difference - 0.002
            if needed_sleep_time > 0:
                # time.sleep(needed_sleep_time)
                pass
        else:
            if (cv2.waitKey(1) == 27):
                print("Display_Window task DONE2")
                cv2.destroyAllWindows()
                program_must_stop.value = 1
                break
            pass

def Save_Video(frame: mp.Queue, video_recording_exist: Value, program_must_stop: Value, fps: Value):
    print("Process Save_Video started")
    video_recording_exist.value = 1
    while True:
        if not frame.empty():
            frame1 = frame.get()
            frame_height, frame_width, frame_channels = frame1.shape
            break
    output_path = f'Output_Media/face_tests_{datetime.datetime.now().strftime("%d%m%Y_%H%M%S")}.mov'
    fourcc = cv2.VideoWriter_fourcc(*'avc1')  # Кодек для видео в формате MOV
    # fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Кодек для видео в формате MP4
    print('Save_Video task Frame Width:', frame_width)
    print('Save_Video task Frame Height:', frame_height)
    print('Save_Video task Video FPS:', fps.value)
    output_video = cv2.VideoWriter(output_path, fourcc, int(fps.value), (frame_width, frame_height))
    if LOGGER_FRAME_WRITE:
        time_difference = 0
        time_of_video_writer_processing = time.time()
        average_fps = 0
        fps_list_lenght = 30
        fps_list = [0.0] * fps_list_lenght
    while True:
        if not frame.empty():
        # if True:
            output_video.write(frame.get(block=True))
            if LOGGER_FRAME_WRITE:
                if len(fps_list) >= fps_list_lenght:
                    fps_list.pop(0)
                time_difference = time.time() - time_of_video_writer_processing
                time_of_video_writer_processing = time.time()
                if time_difference != 0:
                    fps_list.append(1000 / (time_difference * 1000))
                average_fps = sum(fps_list) / len(fps_list)
                print('=================================================')
                print('Frame Width:', frame_width)
                print('Frame Height:', frame_height)
                print('Save_Video Loop time:', round(time_difference, 3))
                print('FPS:', round(average_fps, 3))
        if program_must_stop.value == 1:
            output_video.release()
            print("Save_Video task DONE")
            break

# def DataBase_Write(emotions_value: mp.Queue, program_must_stop):
#     print("Process DataBase_Write started")
#     # Подключение к базе данных
#     conn = sqlite3.connect(data_base_name, check_same_thread=False)
#     cursor = conn.cursor()
#     while True:
#         # number_of_people = number_of_people_queue.get(block=True)
#         # current_time = datetime.now()
#         # our_date = current_time.date()
#         # our_time = current_time.time().strftime('%H:%M:%S')  # Преобразование времени в строку
#
#         number_of_people = number_of_people_queue.get(block=True)
#         volume_of_breast = volume_of_breast_queue.get(block=True)
#         our_date = datetime.datetime.now().strftime("%d.%m.%Y")
#         our_time = datetime.datetime.now().strftime("%H:%M:%S")
#         cursor.execute('''CREATE TABLE IF NOT EXISTS people_and_breast_count
#                         (date DATE, time TEXT, people_count INTEGER, breast_count INTEGER)''')
#
#         cursor.execute("INSERT INTO people_and_breast_count VALUES (?, ?, ?, ?)", (our_date, our_time, number_of_people, volume_of_breast))
#         conn.commit()
#
#         if program_must_stop.value == 1:
#             print("DataBase_Write task DONE")
#             break
#     conn.close()

def Flask_Server_Task(frames_queue: mp.Queue, frames_queue2: mp.Queue, video_recording_exist: Value, program_must_stop):
    print("Process Flask_Server_Task started")
    app = Flask(__name__)

    def get_frames():
        while True:
            frame = frames_queue.get(block=True)
            if frame is None:
                continue
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    @app.route('/video_feed')
    def video_feed():
        return Response(get_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

    app.run(host='0.0.0.0', port=8080)

    # def run_flask():
    #     app.run(host='0.0.0.0', port=8080)

    # flask_thread = threading.Thread(target=run_flask)
    # flask_thread.start()

    # flask_thread.join(timeout=None)  # Дождитесь завершения потока перед выходом из процесса

def Socket_Client_Task_Send(frames_queue: mp.Queue, frames_queue2: mp.Queue, video_recording_exist: Value, program_must_stop):
    print("Process Socket_Client_Task_Send started")
    while True:
        try:
            client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            client_socket.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 1024000)
            # client_socket.settimeout(10)
            #client_socket.connect(('194.87.252.140', 42515))
            client_socket.connect(('127.0.0.1', 42514))
            # client_socket.connect(('127.0.0.1', 42515))
        except socket.timeout:
            print("Connection attempt timed out.")
            time.sleep(1)
            continue
        except ConnectionRefusedError:
            print("Server is not found in the network")
            time.sleep(1)
            continue

        while True:
            frame = frames_queue.get(block=True)
            if frame is None:
                continue

            # frame = cv2.resize(frame, (0, 0), fx=0.3, fy=0.3, interpolation=cv2.INTER_LINEAR)
            frame_bytes = cv2.imencode('.jpg', frame)[1].tobytes()
            
            # Отправляем размер фрейма перед самим фреймом
            size = struct.pack("!I", len(frame_bytes))
            try:
                client_socket.sendall(size)
                client_socket.sendall(frame_bytes)
                
                buffer_size = client_socket.getsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF)
                # print("Socket Buffer size:", buffer_size)
                # print("Real Buffer size:", len(frame_bytes))
                
                # time.sleep(0.01)
            except socket.error:
                print("Client disconnected")
                break

def Socket_Client_Task_Send2(emotions_value: mp.Queue, program_must_stop):
    print("Process Socket_Client_Task_Send2 started")
    while True:
        try:
            client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            client_socket.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 102400)
            # client_socket.settimeout(10)
            client_socket.connect(('194.87.252.140', 42516))
            # client_socket.connect(('127.0.0.1', 42516))
            # client_socket.connect(('127.0.0.1', 42516))
        except socket.timeout:
            print("Connection attempt timed out.")
            time.sleep(1)
            continue
        except ConnectionRefusedError:
            print("Server is not found in the network")
            time.sleep(1)
            continue

        while True:
            data_frame = emotions_value.get(block=True)
            if data_frame is None:
                continue

            # frame = cv2.resize(frame, (0, 0), fx=0.3, fy=0.3, interpolation=cv2.INTER_LINEAR)
            emotions_bytes = pickle.dumps(data_frame)
            
            # Отправляем размер фрейма перед самим фреймом
            size = struct.pack("!I", 72)
            try:
                # client_socket.sendall(size)
                client_socket.sendall(emotions_bytes)
                print("DATA Sent")
                
                # time.sleep(0.01)
            except socket.error:
                print("Client disconnected")
                break

if __name__ == "__main__":

    frames_queue_size = 1
    original_frames_queue = mp.Queue(maxsize=frames_queue_size)
    frames_queue = mp.Queue(maxsize=frames_queue_size)
    frames_queue_after_detection = mp.Queue(maxsize=frames_queue_size)
    frames_queue_after_detection2 = mp.Queue(maxsize=frames_queue_size)
    frames_queue_after_detection_for_recording = mp.Queue(maxsize=frames_queue_size)
    emotions_value = mp.Queue(maxsize=frames_queue_size)
    program_must_stop = Value('i', 0)
    fps = Value('i', 30)
    video_recording_exist = Value('i', 0)

    model = "Models\yolov8_face.pt"
    model2 = "Models\yolov8_emotion_classification.pt"
    target_resolution_for_model = 200 #450 # if 0 then original frame size used #change 0 to 200
    target_crop = [] #[4.00, 5.14, 1.28, 1.29] #[] если не надо кропать
    
    p0 = mp.Process(target=Frame_Read_and_Resize, args=(original_frames_queue,
                                                        frames_queue,
                                                        video_recording_exist,
                                                        stream,
                                                        target_resolution_for_model,
                                                        target_crop,
                                                        program_must_stop,
                                                        fps,))
    p0.start()

    p1 = mp.Process(target=YoLo_Detections, args=(model,
                                                  frames_queue,
                                                  frames_queue_after_detection,
                                                  target_crop,
                                                  emotions_value,
                                                  program_must_stop,))
    p1.start()

    p2 = mp.Process(target=YoLo_Emotions_Classifications, args=(model2,
                                                          frames_queue_after_detection,
                                                          frames_queue_after_detection2,
                                                          target_crop,
                                                          emotions_value,
                                                          program_must_stop,))
    #p2.start()

    p3 = mp.Process(target=Display_Window, args=(frames_queue_after_detection,
                                                 frames_queue_after_detection_for_recording,
                                                 video_recording_exist, # Переписывать фрейм в очередь для записи или нет
                                                 program_must_stop,
                                                 fps,))
    p3.start()

    p4 = mp.Process(target=Save_Video, args=(frames_queue_after_detection_for_recording,
                                             video_recording_exist,
                                             program_must_stop,
                                             fps,))
    # p4.start()
    
    # p5 = mp.Process(target=DataBase_Write, args=(emotions_value,
    #                                              program_must_stop,))
    # p6.start()
    
    p6 = mp.Process(target=Flask_Server_Task, args=(frames_queue,
                                                    original_frames_queue,
                                                    video_recording_exist,
                                                    program_must_stop,))
    # p6.start()
    
    p7 = mp.Process(target=Socket_Client_Task_Send, args=(frames_queue_after_detection2,
                                                    original_frames_queue,
                                                    video_recording_exist,
                                                    program_must_stop,))
    #p7.start()
    
    p8 = mp.Process(target=Socket_Client_Task_Send2, args=(emotions_value,
                                                    program_must_stop,))
    #p8.start()

    p0.join() #    p2.join()




    print("Display_Window Process finished. Terminating all processes in the loop")

    p0.terminate()
    p1.terminate()
    p2.terminate()
    time.sleep(5)
    p4.terminate()
    #p5.terminate()
    p6.terminate()
    p7.terminate()
    p8.terminate()
