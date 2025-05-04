import speech_recognition as sr
import threading
import ras_nlp.tts as tts

cmd_mode = False
text = ''

def check_item(my_list):
    posent = ['엑센트', '도슨트', '또 센트', '더센트', '4센트','코스트', '호스트', '포슨트','폰트', '퍼센트', '토스트', '포스텔', '호스텔', '포스트', '오순태', '첫 번째','오승태','변태', '손태', '콘센트', '4%', '4퍼센트'
    , '픈트', '4센트', '40%', '보쌈 뜻', '서전트', '부산 트', '현태', '보선 트', '토스트', '40%', '고스트', '힌트', '토렌트', '더즌트', '어시스턴트', '텐트', '코스트', '부스트', '보스턴', '포스 뜻', '포승 뜻']
    for token in my_list:
        if token in posent:
            return True
    return False

def handle_message(msg):
    global cmd_mode
    global text
    
    my_list = list(msg.split())
    
    if cmd_mode == True:
        text = ' '.join(s for s in my_list) #문자열로 변환
        return True
    else :
        if check_item(my_list):
            print('저를 부르셨나요? 질문을 말하세요.')
            tts.play_audio('pocent.mp3')
            cmd_mode = True
            start_time = threading.Timer(15, reset_mode)
            start_time.start()
        return False

def reset_mode():
    global cmd_mode
    print('듣기를 종료할게요. 질문에 대해서 고민하는 동안 조금만 기다려주세요.')
    cmd_mode = False

def speech_to_text():
    global text
    
    while True:
        r = sr.Recognizer()
        with sr.Microphone() as source:
            audio = r.listen(source)
        
        try :
            msg = r.recognize_google(audio, language='ko')
            if handle_message(msg):
                break
        except :
            pass
    return text
