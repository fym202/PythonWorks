# -*- coding: utf-8 -*-
"""
Created on 2022-04-07 15:44

@author: Fan yi ming

Func:
"""
import chatClient

Name = "用户2"
ServerAddr = ('127.0.0.1', 10001)
ClientAddr = ('127.0.0.1', 10000)
# 运行函数
chatClient.main(Name, ServerAddr, ClientAddr)