#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
    * @par Copyright (C): 2010-2020, Hunan CLB Tech
    * @file        robot_sevo_face
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
servo_pwm = Adafruit_PCA9685.PCA9685() # 实例话舵机云台

# 设置舵机初始值，可以根据自己的要求调试
servo_pwm.set_pwm_freq(60) # 设置频率为60HZ
servo_pwm.set_pwm(5,0,350) # 底座舵机
servo_pwm.set_pwm(4,0,230) # 倾斜舵机  420
time.sleep(1)

#初始化摄像头并设置阙值
usb_cap = cv2.VideoCapture(0)

# 设置显示的分辨率，设置为320×240 px
usb_cap.set(3, 320)
usb_cap.set(4, 240)

#引入分类器
makerobo_face_cascade = cv2.CascadeClassifier( 'face1.xml' )

pid_x=0
pid_y=0
pid_w=0
pid_h=0

#舵机云台的每个自由度需要4个变量
pid_thisError_x=0   #当前误差值
pid_lastError_x=0   #上一次误差值
pid_thisError_y=0
pid_lastError_y=0

# 舵机的转动角度
pid_X_P = 300
pid_Y_P = 420   #转动角度

pid_flag=0
makerobo_facebool = False

# 机器人舵机旋转
def Robot_servo():
    while True:
        servo_pwm.set_pwm(5,0,650-pid_X_P)
        servo_pwm.set_pwm(4,0,650-pid_Y_P)


servo_tid=threading.Thread(target=Robot_servo)  # 多线程
servo_tid.setDaemon(True)
servo_tid.start()                               # 开启线程

# 循环函数  
while True:   
    ret,frame = usb_cap.read()       # 加载图像
    gray= cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    #对灰度图进行.detectMultiScale()
    faces=makerobo_face_cascade.detectMultiScale(gray)
    max_face=0
    value_x=0
    
    # 找到人脸画矩形
    if len(faces)>0:
        (pid_x,pid_y,pid_w,pid_h) = faces[0]
        cv2.rectangle(frame,(pid_x,pid_y),(pid_x+pid_h,pid_y+pid_w),(0,255,0),2)
        result=(pid_x,pid_y,pid_w,pid_h)
        pid_x=result[0]+pid_w/2
        pid_y=result[1]+pid_h/2
        makerobo_facebool = True      

         # 误差值处理
        pid_thisError_x=pid_x-160
        pid_thisError_y=pid_y-120

        #自行对P和D两个值进行调整，检测两个值的变化对舵机稳定性的影响
        pwm_x = pid_thisError_x*5+1*(pid_thisError_x-pid_lastError_x)
        pwm_y = pid_thisError_y*5+1*(pid_thisError_y-pid_lastError_y)
        
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
    # 显示图像
    cv2.imshow("MAKEROBO Robot", frame)
    if cv2.waitKey(1)==119:
        break
    
usb_cap.release()
cv2.destroyAllWindows()
