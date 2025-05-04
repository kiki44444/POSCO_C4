#### 필수 requirements####
# !conda install –c conda-forge cudatoolkit=11.3 cudnn=8.1
# !pip install tensorflow-gpu==2.6.0
# !conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 -c pytorch

#### requirements for kobert####
# !pip install transformers
# !pip install gluonnlp numpy
# !pip install git+https://git@github.com/SKTBrain/KoBERT.git@master

#### requirements for NER #### 
# !pip install konlpy
# !git clone https://github.com/lovit/customized_konlpy.git
# !pip install customized_konlpy
# !pip install Twitter

#### requirements for camera ####
# requirements.txt

#-----------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------
import time

#소켓 통신
import socket

#NLP 모듈
import server_nlp.kobert_nlp as kobert
import server_nlp.ner as ner
import server_nlp.openai_gpt as openai_gpt

#Object Detection 모듈
import cv2
import camera as cam
import yolo

HOST = '141.223.140.80' # 고다 워크스테이션
PORT = 12345 # Port 번호
API_KEY = "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"
df, context_embeddings = openai_gpt.gpt_ready(API_KEY)


try:
    # 클라이언트와 소켓 연결
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind((HOST,PORT))
    s.listen(10)
    print('Socket Waiting')
    conn, addr = s.accept()
    print('Connection Complete') # 연결 성공 시 본격 코드 진행, connection 실패 시 소켓 close

    # 상황 분류를 위한 kobert 모델 
    model = kobert.make_train_model()
    print('NLP Model is Ready')
    
    while True :
        # 1) 라즈베리파이에서 stt를 통해 문장을 받아온다
        audio_sentence = conn.recv(1024).decode() 
        print(audio_sentence)

        # 2) 문장을 kobert 모델을 통해 상황 분류
        # 0 : 길안내가 필요한 상황 / 1 : 작품에 대한 설명을 해줘야하는 상황
        intention = kobert.predict(model, audio_sentence)
        print(intention)

        # 2-1 ) 길안내 필요한 상황
        if intention == 0 : # 길안내
                print('자율주행')
                # @1 answer = '길안내를 시작합니다.' 
                answer = '길안내'
                conn.send(answer.encode())
                ################자율주행##################
        
        # 2-2 ) 설명이 필요한 상황

        elif intention == 1 : 
                print('NER로 넘어가요')
                
                # 2-2-1 ) NER을 통해 '어떤 작품'에 대한 설명인지 인식
                answer = ner.ner_function(audio_sentence)
                
                if answer == '작품을 인식 중 입니다. 조금만 기다려주세요.' : # 작품명이 포함되어 있지 않은 경우 
                        print(answer)
                        conn.send(answer.encode())
                        cam.imageshoot(conn)
                        img = cv2.imread("detected_img.jpg")
                        name = yolo.detection(img)
                        print(name)
                        query = name + " 작품에 대해 설명해줘."
                        output = openai_gpt.gpt_qa(query, df, context_embeddings)
                        conn.send(output.encode())
                else : # 작품명 및 작가명이 문장 안에 포함 되어있는 경우
                        conn.send(answer.encode())
                        query = name + " 에 대해 설명해줘."
                        output = openai_gpt.gpt_qa(query, df, context_embeddings)
                        conn.send(output.encode())
                
        time.sleep(5)
        conn.send('안내를 종료합니다. 행복한 하루 되세요. 감사합니다.'.encode())
        
except Exception as e : 
        print('Failed, Error Message : ', e)
finally :
        print('Close')
        s.close()

