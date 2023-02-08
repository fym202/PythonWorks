# -*- coding: utf-8 -*-
"""
Created on 2022-04-05 19:46

@author: Fan yi ming

Func: UDP Client
"""
import socket
# 1. 创建socket
serverPort = 12001
servername = '10.68.137.99'
ClientSocket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
# 2. 不需要连接，直接发送
sentence = input('Input lowercase sentence:')
ClientSocket.sendto(sentence.encode(), (servername, serverPort))
modifiedSentence = ClientSocket.recv(2048)
print('From Server: ', modifiedSentence.decode())
ClientSocket.close()
