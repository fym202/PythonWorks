# -*- coding: utf-8 -*-
"""
Created on 2022-04-07 14:21

@author: Fan yi ming

Func:  写一写文件传输类
"""
import socket
import os
import numpy as np
# 文件的收发，写成工具类好了
class FileSR:
    def __init__(self):
        pass

    def FileSend(self, FilePath, serverSocket):
        # 1. 判断是否存在此文件
        if os.path.exists(FilePath):
            #    2. 获取文件信息
            fsize = os.path.getsize(FilePath)
            fmb = fsize / float(1024 * 1024)  # 转为兆
            fblock = fsize / 1024
            senddata = "文件名：%s  文件大小：%.2fMB" % (FilePath, fmb)
            serverSocket.send(senddata.encode("gbk"))
            print(senddata)
            options = "y"
            # 判断是否接收
            if options == "y":
                print("上传中：>>>>>>>", end="")
                # 3. 打开并传输
                with open(FilePath, 'rb') as f:
                    # 　计算总数据包数目
                    nums = fsize / 1024
                    cnum = 0
                    while True:
                        file_data = f.read(1024)
                        cnum = cnum + 1
                        if file_data:
                            serverSocket.send(file_data)
                            # print(">", end="")
                        # 数据读完，退出循环
                        else:
                            serverSocket.send(file_data)
                            print("请求的文件数据发送完成")
                            break
        else:
            print("文件不存在！!!")
        serverSocket.close()
    def FileReceive(self, ClientSocket):
        # 1. 接收文件信息
        file_info = ClientSocket.recv(1024)
        decode_info = file_info.decode("gbk")
        print(decode_info)
        filename = decode_info.split("：")[1].split(" ")[0] # 取出路径和名字
        filename = os.path.split(filename)[1]
        option = "y"
        if option == "y":
            # 2. 发送接收
            print("正在下载>>>>>>>>")
            recvpath = "./DataReceive/"
            if not os.path.exists(recvpath):
                os.mkdir(recvpath)
            with open(recvpath + filename, "wb") as file:
                # 目前接收到的数据包数目
                cnum = 0

                while True:
                    # 循环接收文件数据【recv】
                    file_data = ClientSocket.recv(1024)

                    # 接收到数据
                    if file_data:
                        # 写入数据
                        file.write(file_data)
                        cnum = cnum + 1
                        # print(str(cnum)+":"+str(file_data))
                        # progress =cnum/fileszie2*100
                        # print("当前已下载：%.2f%%"%progress,end = "\r")
                    # 接收完成
                    else:
                        print("下载结束！")
                        break
        else:
            print("退出接收文件程序")