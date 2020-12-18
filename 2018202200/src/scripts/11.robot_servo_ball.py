#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
    * @par Copyright (C): 2010-2020, Hunan CLB Tech
    * @file        robot_sevo_ball
    * @version      V1.0
    * @details
    * @par History
    @author: zhulin
"""
from __future__ import division
import cv2
import time  
import numpy as np
import Adafruit_PCA9685
import threading

#初始化PCA9685和舵机
servo_pwm = Adafruit_PCA9685.PCA9685()  # 实例话舵机云台

# 设置舵机初始值，可以根据自己的要求调试
servo_pwm.set_pwm_freq(60)  # 设置频率为60HZ
servo_pwm.set_pwm(5,0,350)  # 底座舵机
servo_pwm.set_pwm(4,0,370)  # 倾斜舵机
time.sleep(1)

#初始化摄像头并设置阙值
usb_cap = cv2.VideoCapture(0)

# 设置球体追踪的HSV值，上下限值
ball_yellow_lower=np.array([9,135,231])
ball_yellow_upper=np.array([31,255,255])

# 设置显示的分辨率，设置为320×240 px
usb_cap.set(3, 320)
usb_cap.set(4, 240)

#舵机云台的每个自由度需要4个变量
pid_thisError_x=500       #当前误差值
pid_lastError_x=100       #上一次误差值
pid_thisError_y=500
pid_lastError_y=100

pid_x=0
pid_y=0

# 舵机的转动角度
pid_Y_P = 280
pid_X_P = 300           #转动角度
pid_flag=0


# 机器人舵机旋转
def Robot_servo(X_P,Y_P):
    servo_pwm.set_pwm(5,0,650-pid_X_P)
    servo_pwm.set_pwm(4,0,650-pid_Y_P)

# 循环函数
while True:    
    ret,frame = usb_cap.read()

    #高斯模糊处理
    frame=cv2.GaussianBlur(frame,(5,5),0)
    hsv= cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)

    #ROI及找到形态学找到小球进行处理
    mask=cv2.inRange(hsv,ball_yellow_lower,ball_yellow_upper) # 掩膜处理
    mask=cv2.erode(mask,None,iterations=2)
    mask=cv2.dilate(mask,None,iterations=2)
    mask=cv2.GaussianBlur(mask,(3,3),0)
    res=cv2.bitwise_and(frame,frame,mask=mask)
    cnts=cv2.findContours(mask.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[-2]  #发现小球
    #当找到小球处理
    if len(cnts)>0:
        cap_cnt=max(cnts,key=cv2.contourArea)
        (pid_x,pid_y),radius=cv2.minEnclosingCircle(cap_cnt)
        cv2.circle(frame,(int(pid_x),int(pid_y)),int(radius),(255,0,255),2)

        # 误差值处理
        pid_thisError_x=pid_x-160
        pid_thisError_y=pid_y-120

        #PID控制参数
        pwm_x = pid_thisError_x*3+1*(pid_thisError_x-pid_lastError_x)
        pwm_y = pid_thisError_y*3+1*(pid_thisError_y-pid_lastError_y)

        #迭代误差值操作
        pid_lastError_x = pid_thisError_x
        pid_lastError_y = pid_thisError_y

        pid_XP=pwm_x/100
        pid_YP=pwm_y/100
        
        # pid_X_P pid_Y_P 为最终PID值
        pid_X_P=pid_X_P-int(pid_XP)
        pid_Y_P=pid_Y_P-int(pid_YP)

        #限值舵机在一定的范围之内
        if pid_X_P>670:
            pid_X_P=650
        if pid_X_P<0:
            pid_X_P=0
        if pid_Y_P>650:
            pid_Y_P=650
        if pid_X_P<0:
            pid_Y_p=0

    servo_tid=threading.Thread(target=Robot_servo,args=(pid_X_P,pid_Y_P))  # 多线程
    servo_tid.setDaemon(True)
    servo_tid.start()   # 开启线程

    cv2.imshow("MAKEROBO Robot", frame)  # 显示图像
    if cv2.waitKey(1)==119:
        break
    
usb_cap.release()
cv2.destroyAllWindows()
