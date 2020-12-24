#!/usr/bin/python
# -*- coding: utf8 -*-
import threading
import socket
import sys
import time
import cv2
import copy
import os
from PIL import Image
import numpy as np
import libh264decoder
import winsound

# 设置tello三个数据端口

command_address = ('192.168.10.1', 8889)  # 主机名或者IP地址+端口号

# 设置电脑端口
host = ''
port1 = 9000  # 电脑用来发射指令的端口
port2 = 8890  # 电脑用来接收状态的端口
port3 = 11111  # 电脑用来接收视频流的端口
port4 = 9001

local_command_address = (host, port1)
local_state_address = (host, port2)
local_video_address = (host, port3)
local_keep_address = (host, port4)

decoder = libh264decoder.H264Decoder()  # 设置解码器


def recv_command_reply():  # 这是一个可以实时接收tello发出的信息的线程
    while True:
        # try:
        for i in range(1):
            data, server = sock.recvfrom(1024)
            # data, server = sock.recvfrom(15)
            if data.decode(encoding="utf-8") == "ok":  # 二进制解码
                winsound.Beep(500, 300)
            else:
                winsound.Beep(1000, 1000)
                print(data.decode(encoding="utf-8"))

            # print(server)
        # except Exception:
        # print ('\nExit . . .\n')
        # break


def keep_command():
    while True:
        msg = "command".encode(encoding="utf-8")  # 将字符串转化为二进制
        sent = sock.sendto(msg, command_address)
        time.sleep(5)


def recv_video():  # 这是一个可以实时接收tello发出的信息的线程
    video_data = ""
    count = 0
    while True:

        data, server = sock_video.recvfrom(2048)
        video_data += data
        if len(data) != 1460:
            new_im = _h264_decode(video_data)  # 这里解码出了一个图
            if new_im != []:
                new_im = copy.deepcopy(np.array(new_im[0]).transpose()[::-1].transpose())
                if new_im.shape == (720, 960, 3):
                    process(new_im)  # 传入图像并返回命令
                    count += 1
                    # time.sleep(2)

                video_data = ""


def process(im_array):  # 这个函数用于接受图像array，更新窗口和传出指令
    face_position = face_detect(im_array)  # 使用级联分类器得到人脸的位置
    show(face_position, im_array)  # 在可视化窗口上更新人脸，并圈出人脸的范围
    send_command(face_position)  # 根据人脸的位置给出指令


def show(face_position, im_array):
    pass


def face_detect(im_array):
    pass


def send_command(xyz):
    if xyz is None:
        pass
    elif len(xyz)==3:
        fo, le, up = xyz
        forward=int(fo*5)
        left=int(le/8)
        rise=int(up/8)
        temp=[]
        for i in [forward, left, rise]:
            if abs(i) < 16:
                temp.append("0")
            elif abs(i) < 20:
                temp.append(str(np.sign(i)*21))
            else:
                temp.append(str(i))
        msg=" ".join(["go"]+temp+[str(21)])
        print(msg)
        sock.sendto(msg.encode(encoding="utf-8"), command_address)
    else:
        pass


def _h264_decode(packet_data):
    res_frame_list = []
    frames = decoder.decode(packet_data)
    for framedata in frames:
        (frame, w, h, ls) = framedata
        if frame is not None:
            # print 'frame size %i bytes, w %i, h %i, linesize %i' % (len(frame), w, h, ls)

            frame = np.fromstring(frame, dtype=np.ubyte, count=len(frame), sep='')
            frame = (frame.reshape((h, ls / 3, 3)))
            frame = frame[:, :w, :]
            res_frame_list.append(frame)

    return res_frame_list


def manual(command):
    for i in range(1):
        if len(command) > 1:
            drct, param = command[0], command[1:]
            try:
                x = eval(param)
                if x <= 20:
                    x = 21
                elif x >= 500:
                    x = 499

                if drct == "w":
                    return "forward" + " " + str(x)
                elif drct == "s":
                    return "back" + " " + str(x)
                elif drct == "a":
                    return "left" + " " + str(x)
                elif drct == "d":
                    return "right" + " " + str(x)

                elif drct == "f":
                    return "up" + " " + str(x)
                elif drct == "c":
                    return "down" + " " + str(x)

                elif drct == "q":
                    return "ccw" + " " + str(x)
                elif drct == "e":
                    return "cw" + " " + str(x)

            except:
                return command

        elif len(command) == 1:
            if command == "w":
                return "forward" + " " + "35"
            elif command == "s":
                return "back" + " " + "35"
            elif command == "a":
                return "left" + " " + "35"
            elif command == "d":
                return "right" + " " + "35"

            elif command == "f":
                return "up" + " " + "35"
            elif command == "c":
                return "down" + " " + "35"

            elif command == "q":
                return "ccw" + " " + "35"
            elif command == "e":
                return "cw" + " " + "35"
            elif command == "o":
                return "streamon"
            elif command == "p":
                return "streamoff"
            else:
                return command


        else:
            return command


def face_detect(image):
    Ideal_area = 102.01
    cascPath = "haarcascade_frontalface_default.xml"
    faceCascade = cv2.CascadeClassifier(cascPath)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    center_y, center_x, bands = np.array(image.shape) / 2

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(1, 1),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    if len(faces) == 1:

        for (x, y, w, h) in faces:
            # 框出人脸范围
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            bounding_centre_coordinates_x = (x + x + w) / 2
            bounding_centre_coordinates_y = (y + y + h) / 2
            cv2.circle(image, (bounding_centre_coordinates_x, bounding_centre_coordinates_y), 5, (0, 255, 0), -1)
            image = image.copy()
            cv2.imshow("Faces Detected", image)
            cv2.waitKey(500)  # 是不是太长了？0.5秒也不长啊？
            Area_z = w * h
            # print("x", bounding_centre_coordinates_x)
            # print("y", bounding_centre_coordinates_y)
            # print("s", Area_z)
            # print((-center_x + bounding_centre_coordinates_x), (center_y - bounding_centre_coordinates_y),
            #     (-1 * Ideal_area + Area_z) / 100)
            x_axis = -bounding_centre_coordinates_x + center_x
            y_axis = center_y - bounding_centre_coordinates_y
            z_axis = np.log((Ideal_area/Area_z * 1000) ** 0.5)
            #print("delta forward left up", z_axis, x_axis, y_axis)
            return z_axis, x_axis, y_axis
    # else:
    # cv2.imshow("Faces Detected", image)
    # cv2.waitKey(500)


# Create UDP sockets
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind(local_command_address)  # 将socket绑定到一个地址和端口上，通常用于socket服务端

sock_state = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock_state.bind(local_state_address)

sock_video = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock_video.bind(local_video_address)

sock_keep = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock_keep.bind(local_keep_address)

# 接收命令返回信号
command_recvThread = threading.Thread(target=recv_command_reply)
command_recvThread.start()

# 接收视频返回信号
video_recvThread = threading.Thread(target=recv_video)
video_recvThread.start()

# 实时维护控制状态
keep_SendingThread = threading.Thread(target=keep_command)
keep_SendingThread.start()

sock.sendto("streamoff".encode(encoding="utf-8"), command_address)

while True:
    # print(video_recvThread.is_alive(), command_recvThread.is_alive())
    try:
        msg = manual(input("请输入命令"))
        # print(msg)
        if not msg:
            break

        if 'end' in msg:
            print ('...')
            sock.close()

        # Send data
        msg = msg.encode(encoding="utf-8")  # 将字符串转化为二进制
        sent = sock.sendto(msg, command_address)
        # print(sent)
    except KeyboardInterrupt:
        print ('\n . . .\n')
        sock.close()
        break

    except NameError or SyntaxError:
        print("这不是有效的命令")

sock_keep.close()
sock_video.close()
sock_state.close()
