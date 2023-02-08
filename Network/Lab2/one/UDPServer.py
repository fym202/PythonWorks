# -*- coding: utf-8 -*-
"""
Created on 2022-04-05 19:42

@author: Fan yi ming

Func: UDP server
"""
import socket
# 创建socket
serverPort = 12001
serverSocket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
# 激活socket
serverSocket.bind(('', serverPort))
print('The server is ready to receive!')
while True:
    message, ClientAddress = serverSocket.recvfrom(2048)
    modifiedMessage = message.decode().upper()
    serverSocket.sendto(modifiedMessage.encode(), ClientAddress)  # 发送给对应端口
