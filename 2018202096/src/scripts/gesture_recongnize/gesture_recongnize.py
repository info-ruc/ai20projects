import os
import cv2
import time
from djitellopy import Tello
from aip import AipBodyAnalysis
from threading import Thread
import urllib
import base64
import tellopy

""" 你的 APPID AK SK """
APP_ID = '22934281'
API_KEY = 'H3FeXjMW25aWyhV2OraHn8Cr'
SECRET_KEY =  'PdAf6BBdCuBfre3blvrHrT3aWKHoPQ5q'
''' 调用'''
gesture_client = AipBodyAnalysis(APP_ID, API_KEY, SECRET_KEY)

capture = cv2.VideoCapture(0)#0为默认摄像头

img_count = 1
 # 读入Tello图像
def telloGetFrame(myDrone, w= 360,h=240):
    myFrame = myDrone.get_frame_read()
    myFrame = myFrame.frame
    print(type(myFrame))
    img = cv2.resize(myFrame,(w,h))
    return img

def camera():
    while True:
        ret, frame = capture.read()
        # cv2.imshow(窗口名称, 窗口显示的图像)
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) == ord('q'):
            break
Thread(target=camera).start()#引入线程防止在识别的时候卡死

def get_file_content(filePath):
    with open(filePath, 'rb') as fp:
        return fp.read()

def gesture_recognition():
    '''
    第一个参数ret 为True 或者False,代表有没有读取到图片
    第二个参数frame表示截取到一帧的图片
    '''
    ret, frame = capture.read()
    #print(type(frame))
    #只接受base64格式的图片   
    #base64_img = base64.b64encode(frame)
    #ur = urllib.parse.urlencode(base64_img,"UTF-8")
    cv2.imwrite("{}.png".format(img_count), frame)
    image = get_file_content("{}.png".format(img_count))
    gesture =  gesture_client.gesture(image)   #AipBodyAnalysis内部函数
    print(gesture)

while(True):
    gesture_recognition()
    img_count += 1
    time.sleep(3)
   