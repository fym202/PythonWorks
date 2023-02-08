# -*- coding: utf-8 -*-
"""
Created on 2022-04-05 20:05

@author: Fan yi ming

Func: 多线程的TCP连接
"""
from threading import Thread
import socket

# 定义函数
def connect(new_socket, clidentAddress):
    print('建立连接:', clidentAddress)
    while True:
        sentence = new_socket.recv(1024)
        if sentence:
            print('接收到', clidentAddress, '的数据', sentence.decode('gbk'))
            print("请输入返回", clidentAddress, "的消息：")
            send_data = input()
            new_socket.send(send_data.encode('gbk'))
        else:
            break
    # 关闭连接
    new_socket.close()
    print("来自", clidentAddress, "的连接关闭")


# 创建socket
serverPort = 12000
serverSocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
# 激活socket
serverSocket.bind(('', serverPort))
serverSocket.listen(1)  # 监听
print('The server(TCP) is ready to receive!')
try:
    while True:
        connectionSocket, addr = serverSocket.accept()
        t = Thread(target=connect, args=(connectionSocket, addr))
        t.start()
except:
    print("something wrong0")
finally:
    serverSocket.close()