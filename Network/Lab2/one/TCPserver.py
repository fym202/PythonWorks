# -*- coding: utf-8 -*-
"""
Created on 2022-04-05 17:01

@author: Fan yi ming

Func: TCP connect
"""
import socket
# 创建socket
serverPort = 12000
serverSocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
# 激活socket
serverSocket.bind(('', serverPort))
serverSocket.listen(1)  # 监听
print('The server(TCP) is ready to receive!')
while True:
    connectionSocket, addr = serverSocket.accept()
    sentence = connectionSocket.recv(1024).decode()  # 接收1024字节
    capitalizeSentence = sentence.upper()  # 全部变为大写
    connectionSocket.send(capitalizeSentence.encode())
    connectionSocket.close()



