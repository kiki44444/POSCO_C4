import cv2
import numpy as np

def shoot(server) : 
    cam = cv2.VideoCapture(0)
    cam.set(3,320)
    cam.set(4,240)
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 100]
    ret, frame = cam.read()
    result, frame = cv2.imencode('.jpg', frame, encode_param)
    data = np.array(frame)
    stringData = data.tostring()
    server.sendall((str(len(stringData))).encode().ljust(16) + stringData)
    cam.release()
    