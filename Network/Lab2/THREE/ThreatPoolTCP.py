# -*- coding: utf-8 -*-
"""
Created on 2022-04-05 21:06

@author: Fan yi ming

Func: use threat pool
"""
import socket
from concurrent.futures import ProcessPoolExecutor

class MyTcpServer:
    def __init__(self):
        self.address = ('', 12000)
        self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server.bind(self.address)
        self.server.listen(5)  # 参数为指定的链接数
        print("服务端启动，等待连接...")

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
                    print('接收到', clientAddr, '的数据为:', recv_data)
                    recv_data = '处理后的：'+recv_data
                    new_socket.send(recv_data.encode('gbk'))
                else:
                    break
            except:
                print("error!")
                break
        new_socket.close()
        print("来自", clientAddr, "的连接关闭")


if __name__ == '__main__':
    server = MyTcpServer()
    pool = ProcessPoolExecutor(2)    # 创建一个2个线程的线程池
    while True:
        ConnectionSocket, clientAddress = server.wait_accept()
        pool.submit(server.handle_request, ConnectionSocket, clientAddress)	  #提交任务