import socket

HOST = ''  # Server IP or Hostname
PORT = 12345 
# Pick an open Port (1000+ recommended), must match the client sport

# TCP
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
print('Socket created')

# 서버의 IP와 PORT 번호 지정
s.bind((HOST, PORT))

# 클라이언트의 접속 대기
s.listen(5) #클라이언트 연결을 5개까지 받는다
print('Socket awaiting messages')

#연결, conn에는 소켓 객체, addr은 소켓에 바인드된 주소
conn, addr = s.accept()

print('Connected')

# awaiting for message
while True:
    data = conn.recv(1024)
    data_decode = data.decode('utf-8')
    print('I sent a message back in response to: ', data_decode)
    reply = ''

	# process your message
    if data_decode == 'Hello':
        reply = 'Hi, back!'
    elif data_decode == 'This is important':
        reply = 'OK, I have done the important thing you have asked me!'
	#and so on and on until...
    elif data_decode == 'quit':
        reply = 'Terminate'
        reply = reply.encode('utf-8')
        conn.send(reply)
        break
    else:
        reply = 'Unknown command'

	# Sending reply
    reply = reply.encode('utf-8')
    conn.send(reply)
    
conn.close() 
# Close connections