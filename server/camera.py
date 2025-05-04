import socket
import cv2
import numpy as np

#socket에서 수신한 버퍼를 반환하는 함수
def recvall(sock, count):
    # 바이트 문자열
    buf = b''
    while count:
        newbuf = sock.recv(count)
        if not newbuf: return None
        buf += newbuf
        count -= len(newbuf)
    return buf

def imageshoot(client) :   
    length = recvall(client, 16)
    stringData = recvall(client, int(length))
    data = np.fromstring(stringData, dtype = 'uint8')
    frame = cv2.imdecode(data, cv2.IMREAD_COLOR)
    cv2.imwrite('detected_img.jpg', frame)
    

