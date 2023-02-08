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
import FileModel
debug_on = False
class ServerPart:
    def __init__(self, Addr, name):
        self.Addr = Addr
        self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server.bind(self.Addr)
        self.server.listen(5)
        self.myprocess = "normal"
        print(name + "客户服务端启动---")

    def wait_accept(self):
        new_socket, clientAddr = self.server.accept()
        return new_socket, clientAddr

    def handle_request(self, new_socket, clientAddr):
        print("建立一个连接，来自", clientAddr)
        while True:
            # 接收对方发送过来的数据
            recv_data = new_socket.recv(1024).decode('gbk')  # 接收1024个字节
            if recv_data:
                print(clientAddr, '的信息:', recv_data)
                # 判断是否传文件
                if recv_data == "File":
                    FileSocket, Address = self.wait_accept()
                    fileReceive = FileModel.FileSR()
                    fileReceive.FileReceive(FileSocket)
            else:
                break
        new_socket.close()
        print("来自", clientAddr, "的连接关闭")

    def run(self):
        # print('running')
        ConnectionSocket, clientAddress = self.wait_accept()
        t = Thread(target=self.handle_request, args=(ConnectionSocket, clientAddress))
        t.start()


class ClientPart:
    def __init__(self, Addr):
        self.serverName = Addr[0]
        self.serverPort = Addr[1]
        self.ClientSocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    def connect(self, Addr=None):
        if not Addr:
            Addr = (self.serverName, self.serverPort)
        time.sleep(5)  # 等待三秒
        self.ClientSocket.connect(Addr)
        # 基础路径
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        while True:
            sentence = input('I say to '+Addr[0]+'-'+str(Addr[1])+":")
            # ① 退出情况
            if sentence == 'q':  # 退出
                break
            self.ClientSocket.send(sentence.encode('gbk'))
            #  ② 发送文件的情况
            if sentence == "File":
                filesender = FileModel.FileSR()
                filename = input("请输入文件名：")
                file_path = os.path.join(BASE_DIR, "Data", filename)
                if debug_on:
                    print(file_path)
                # 新开一个socket
                FileSocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                FileSocket.connect(Addr)
                filesender.FileSend(FilePath=file_path,
                                serverSocket=FileSocket)
        self.ClientSocket.close()

    def run(self):
        t = Thread(target=self.connect)
        t.run()


# ServerAddr 是自己的服务器的地址，ClientAddr 是要连接的服务器地址
def main(Name, ServerAddr, ClientAddr):
    # 运行服务端
    # Addr = ('127.0.0.1', 12003)  # 服务器地址
    Addr = ServerAddr
    name = Name
    server = ServerPart(Addr, name)
    Thread(target=server.run).start()
    # 运行客户端
    # print('Client running')
    client = ClientPart(ClientAddr)
    client.run()

if __name__ == '__main__':

    main()