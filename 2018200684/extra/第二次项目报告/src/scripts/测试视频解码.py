#!/usr/bin/python
# -*- coding: utf8 -*-

import threading
import socket
import sys
import time
import cv2
import os
import libh264decoder

#设置tello三个数据端口
command_address = ('192.168.10.1', 8889)   #主机名或者IP地址+端口号


#设置电脑端口
host = ''
port1 = 9000   #电脑用来发射指令的端口
port2 = 8890   #电脑用来接收状态的端口
port3 = 11111  #电脑用来接收视频流的端口
local_command_address = (host,port1)
local_state_address=(host, port2)
local_video_address=(host, port3)

decoder = libh264decoder.H264Decoder()   #设置解码器

def recv_command_reply():  # 这是一个可以实时接收tello发出的信息的线程
    while True:
        try:
            data, server = sock.recvfrom(1024)
            # data, server = sock.recvfrom(15)
            print(data.decode(encoding="utf-8"))  # 二进制解码
            # print(server)
        except Exception:
            print ('\nExit . . .\n')
            break


def recv_video():  # 这是一个可以实时接收tello发出的信息的线程
    while True:
        try:
            data, server = sock_video.recvfrom(int(1e7))
            print server  # 二进制解码
            # a.write(data)
            # print("writing data...",end="")
            print(data)
        except Exception:
            print ('\nExit . . .\n')
            break

# Create a UDP socket
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind(local_command_address)   #将socket绑定到一个地址和端口上，通常用于socket服务端

sock_state=socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock_state.bind(local_state_address)

sock_video=socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock_video.bind(local_video_address)



#接收命令返回信号
command_recvThread = threading.Thread(target=recv_command_reply)
command_recvThread.start()

#接收视频返回信号
video_recvThread = threading.Thread(target=recv_video)
video_recvThread.start()



while True:
    print(video_recvThread.is_alive(),command_recvThread.is_alive())
    try:
        msg = input("请输入命令");

        if not msg:
            break

        if 'end' in msg:
            print ('...')
            sock.close()
            #a.close()
            break

        # Send data
        msg = msg.encode(encoding="utf-8")    #将字符串转化为二进制
        sent = sock.sendto(msg, command_address)
    except KeyboardInterrupt:
        print ('\n . . .\n')
        sock.close()
        break

    except NameError or SyntaxError:
        print("这不是有效的命令")


