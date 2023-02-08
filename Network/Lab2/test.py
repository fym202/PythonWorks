# -*- coding: utf-8 -*-
"""
Created on 2022-04-05 20:01

@author: Fan yi ming

Func:test bingxing
"""
from time import sleep
import socket
import os
# 获取本机ip
def get_host_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(('8.8.8.8', 80))
        ip = s.getsockname()[0]
    finally:
        s.close()

    return ip
print(get_host_ip())
filename = "Filename：./Data/2.jpg  size： 1MB"
# 基础路径
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
files = os.path.join(BASE_DIR, "ChatFile/Data", "2.jpg")
print(files)
if os.path.exists(BASE_DIR):
    print("good")

fi = "E:\\Python\\pythonworks\\Projects\\Network\\Lab2\\ChatFile\\Data\\1.jpg"
print()
# print(filename.split("：")[1].split(" ")[0].split("/")[-1])