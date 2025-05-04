#### requirements for SST #### 
# !pip install speechRecognition
# !pip install --upgrade google-cloud-texttospeech
# !pip install pygame

# ---------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------

# 소켓 통신
import socket

# NLP 모듈
import ras_nlp.stt as stt
import ras_nlp.tts as tts

# Object Detection 모듈
import camera_sensor
import time


HOST = '141.223.140.80' #고다 워크스테이션
PORT = 12345 # Port 번호

try:
    # 클라이언트와 소켓 연결
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((HOST,PORT))
    print('Client Connected to Main Server')

    for i in range(0,100,2) : 
        
            audio_sentence = stt.speech_to_text()
            s.sendall(audio_sentence.encode()) # 전체 문장 서버에 전달

            # 2) 서버에서 상황을 분류하고
            # 2-1) 길안내가 필요한 상황이면 '길안내를 시작합니다' -> 자율주행 
            # 2-2) 작품 설명이 필요한 상황이면
            # 2-2-1) 작품명이 이미 질문에 포함되어 있을 때 -> 해당 답변 출력
            # 2-2-2) 작품명이 없을 때 (ex. 이 작품에 대해 알려줘) -> 작품 인식 -> 인식 결과에 대한 답변을 받아옴 
            answer = s.recv(1024).decode()
            print(answer)

            if answer == '길안내' :
                dest = s.recv(1024).decode()
                tts.text_to_speech(dest, 'output{}.mp3'.format(i))
                tts.play_audio('output{}.mp3'.format(i))
                if '안내' in dest : 
                    time.sleep(60)
                    arrived = s.recv(1024).decode()
                    tts.play_audio('navigation_end.mp3')

            elif answer == '작품을 인식 중 입니다. 조금만 기다려주세요.' :
                tts.play_audio('detection.mp3')
                camera_sensor.shoot(s)
                answer = s.recv(1024).decode()
                print(answer)
                tts.text_to_speech(answer, 'output{}.mp3'.format(i))
                tts.play_audio('output{}.mp3'.format(i))
            
            else : # answer == 작품명 또는 작가명
                tts.text_to_speech(answer, 'output{}.mp3'.format(i))
                tts.play_audio('output{}.mp3'.format(i))
            
            time.sleep(2)
            tts.play_audio('end.mp3')

except Exception as e :
    print('Failed. Error message : ', e)

finally : 
    print('Close')
    s.close()
