# -*- coding: utf-8 -*-
"""
Created on 2022-04-05 18:16

@author: Fan yi ming

Func:Client of TCP  发送给服务器数据，等待服务器返回
"""
import socket
# 1. 创建socket
servername = '10.68.137.99'
serverPort = 12000
ClientSocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
# 2. 连接成功
ClientSocket.connect((servername, serverPort))
sentence = input('Input lowercase sentence:')
ClientSocket.send(sentence.encode())
modifiedSentence = ClientSocket.recv(1024)
print('From Server: ', modifiedSentence.decode())
ClientSocket.close()

