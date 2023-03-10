# -*- coding: utf-8 -*-
# -*- coding:utf-8 -*-
import socket
import os
import threading

# 获取本机ip
def get_host_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(('8.8.8.8', 80))
        ip = s.getsockname()[0]
    finally:
        s.close()

    return ip


# 处理客户端请求下载文件的操作（从主线程提出来的代码）
# 传输文件需要： 1 文件名（带路径），一定要判断文件名是否存在
# 2. 需要已有的TCP连接的socket
def deal_client_request(ip_port, service_client_socket):
    # 连接成功后，输出“客户端连接成功”和客户端的ip和端口
    print("客户端连接成功", ip_port)
    # 接收客户端的请求信息【recv】
    file_name = service_client_socket.recv(1024)
    # 解码
    file_name_data = file_name.decode("utf-8")
    # 判断文件是否存在
    if os.path.exists(file_name_data):
        #输出文件字节数
        fsize = os.path.getsize(file_name_data)
        #转化为兆单位
        fmb = fsize/float(1024*1024)
        #要传输的文件信息
        senddata = "文件名：%s  文件大小：%.2fMB"%(file_name_data, fmb)
        #发送和打印文件信息【send】
        service_client_socket.send(senddata.encode("utf-8"))
        print("请求文件名：%s  文件大小：%.2f MB"%(file_name_data, fmb))
        #接受客户是否需要下载【recv】
        options = service_client_socket.recv(1024)
        if options.decode("utf-8") == "y":
            # 打开文件
            with open(file_name_data, "rb") as f:
                #　计算总数据包数目
                nums = fsize/1024
                #　当前传输的数据包数目
                cnum = 0

                while True:
                    file_data = f.read(1024)
                    cnum = cnum + 1
                    #progress = cnum/nums*100
                    print(str(cnum) + ":" + str(file_data))
                    #print("当前已下载：%.2f%%"%progress,end = "\r")
                    if file_data:
                        # 只要读取到数据，就向客户端进行发送【send】
                        service_client_socket.send(file_data)
                        print(str(cnum) + ":" + str(file_data))
                    # 数据读完，退出循环
                    else:

                        print("请求的文件数据发送完成")
                        break
        else:
            print("下载取消！")
    else:
        print("下载的文件不存在！")
    # 关闭服务当前客户端的套接字【close】
    service_client_socket.close()


if __name__ == '__main__':
    # 获取本机ip
    print("TCP文件传输服务器，本机IP:" + get_host_ip())

    # 把工作目录切换到data目录下
    os.chdir("../Data")
    # 创建套接字【socket】
    tcp_server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # 绑定端口号【bind】
    tcp_server_socket.bind(("", 3356))
    # 设置监听，将主动套接字变为被动套接字【listen】
    tcp_server_socket.listen(128)

    # 循环调用【accept】，可以支持多个客户端同时连接，和多个客户端同时下载文件
    while True:
        service_client_socket, ip_port = tcp_server_socket.accept()
        # 连接成功后打印套接字号
        #print(id(service_client_socket))

        # 创建子线程
        sub_thread = threading.Thread(target=deal_client_request, args=(ip_port, service_client_socket))
        # 启动子线程
        sub_thread.start()