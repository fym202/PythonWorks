# -*- coding: utf-8 -*-
"""
Created on 2022-04-05 21:35

@author: Fan yi ming

Func: 初始版本实现聊天127.0.0.1-12003
"""
from threading import Thread
import socket
import os
import time
class ServerPart:
    def __init__(self, Addr, name):
        self.Addr = Addr
        self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server.bind(self.Addr)
        self.server.listen(5)
        print(name + "客户服务端启动---")

    def wait_accept(self):
        new_socket, clientAddr = self.server.accept()
        return new_socket, clientAddr

    def handle_request(self, new_socket, clientAddr):
        print("建立一个连接，来自", clientAddr)
        while True:
            try:
                # 接收对方发送过来的数据
                recv_data = new_socket.recv(1024).decode('gbk')  # 接收1024个字节
                if recv_data:
                    print(clientAddr, '的信息:', recv_data)
                else:
                    break
            except:
                print("error!")
                break
        new_socket.close()
        print("来自", clientAddr, "的连接关闭")

    def run(self):
        print('running')
        while True:
            ConnectionSocket, clientAddress = self.wait_accept()
            t = Thread(target=self.handle_request, args=(ConnectionSocket, clientAddress))
            t.start()
        self.server.close()


class ClientPart:
    def __init__(self):
        self.serverName = '127.0.0.1'
        self.serverPort = 12005
        self.ClientSocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    def connect(self, Addr=None):
        if not Addr:
            Addr = (self.serverName, self.serverPort)
        time.sleep(10)  # 等待三秒
        self.ClientSocket.connect(Addr)
        while True:
            sentence = input('I say to '+Addr[0]+'-'+str(Addr[1])+":")
            if sentence == 'q':  # 退出
                break
            self.ClientSocket.send(sentence.encode('gbk'))
        self.ClientSocket.close()

    def run(self):
        t = Thread(target=self.connect)
        t.run()


def main():
    # 运行服务端
    Addr = ('127.0.0.1', 12003)  # 服务器地址
    name = '用户A'
    server = ServerPart(Addr, name)
    Thread(target=server.run).start()
    # 运行客户端
    print('Client running')
    client = ClientPart()
    client.run()

if __name__ == '__main__':
    main()