import socket

HOST = '141.223.140.80' # 고다 워크스테이션
PORT = 12345 

# TCP


try : 
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    print('Socket 1 created')
    s.bind((HOST, PORT))
    s.listen(5) 
    print('Socket 1 awaiting messages')
    conn1, addr = s.accept()
    print('Client 1 Connected Clear')


    s2 = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    print('Socket 2 Created')
    s2.bind((HOST, PORT+1))
    s2.listen(5)
    print('Socket 2 awaiting messages')
    conn2, addr = s2.accept()
    print('Client 2 Connected Clear')


    # awaiting for message
    while True:
        print('Message sent from client 1')
        data_1 = conn1.recv(1024).decode()
        print(data_1)

        print("Let's send message to client 2")
        conn2.send(data_1.encode())
        print('Message sent successful')
        break
except Exception as e  :
    print('Error Message : ', e) 
    conn1.close() 
    conn2.close()

finally : 
    conn1.close()
    conn2.close()



# Close connections