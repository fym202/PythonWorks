# -*- coding: utf-8 -*-
"""
Created on 2022-04-07 15:41

@author: Fan yi ming

Func: 用户1
"""
import chatClient

Name = "用户1"
ServerAddr = ('127.0.0.1', 10000)
ClientAddr = ('127.0.0.1', 10001)
# 运行函数
chatClient.main(Name, ServerAddr, ClientAddr)