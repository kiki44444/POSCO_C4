import socket

HOST = '141.223.140.80' # 고다 워크스테이션
# Enter IP or Hostname of your server
PORT = 12345
# Pick an open Port (1000+ recommended), must match the server port


try : 
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((HOST,PORT))


    #Lets loop awaiting for your input
    while True:
            command = input('Enter your command: ')
            s.send(command.encode())
            print('Message successfully sent')
            break
except Exception as e : 
    print('Failed, error message : ', e)
    s.close()

finally :
    s.close()