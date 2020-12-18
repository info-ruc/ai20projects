import cv2
import numpy as np
import zbar
from PIL import Image
import Adafruit_PCA9685
import pygame
import  RPi.GPIO as GPIO
import time 
import sys 

if sys.getdefaultencoding() != 'utf-8':
    reload(sys)
    sys.setdefaultencoding('utf-8')
    
# 树莓派小车电机驱动初始化
PWMA = 18
AIN1   =  22
AIN2   =  27

PWMB = 23
BIN1   = 25
BIN2  =  24
# 树莓派小车运动函数
def t_up(speed,t_time):
        L_Motor.ChangeDutyCycle(speed)
        GPIO.output(AIN2,False)#AIN2
        GPIO.output(AIN1,True) #AIN1

        R_Motor.ChangeDutyCycle(speed)
        GPIO.output(BIN2,False)#BIN2
        GPIO.output(BIN1,True) #BIN1
        time.sleep(t_time)
        
def t_stop(t_time):
        L_Motor.ChangeDutyCycle(0)
        GPIO.output(AIN2,False)#AIN2
        GPIO.output(AIN1,False) #AIN1

        R_Motor.ChangeDutyCycle(0)
        GPIO.output(BIN2,False)#BIN2
        GPIO.output(BIN1,False) #BIN1
        time.sleep(t_time)
        
def t_down(speed,t_time):
        L_Motor.ChangeDutyCycle(speed)
        GPIO.output(AIN2,True)#AIN2
        GPIO.output(AIN1,False) #AIN1

        R_Motor.ChangeDutyCycle(speed)
        GPIO.output(BIN2,True)#BIN2
        GPIO.output(BIN1,False) #BIN1
        time.sleep(t_time)

def t_left(speed,t_time):
        L_Motor.ChangeDutyCycle(speed)
        GPIO.output(AIN2,True)#AIN2
        GPIO.output(AIN1,False) #AIN1

        R_Motor.ChangeDutyCycle(speed)
        GPIO.output(BIN2,False)#BIN2
        GPIO.output(BIN1,True) #BIN1
        time.sleep(t_time)

def t_right(speed,t_time):
        L_Motor.ChangeDutyCycle(speed)
        GPIO.output(AIN2,False)#AIN2
        GPIO.output(AIN1,True) #AIN1

        R_Motor.ChangeDutyCycle(speed)
        GPIO.output(BIN2,True)#BIN2
        GPIO.output(BIN1,False) #BIN1
        time.sleep(t_time)  

GPIO.setwarnings(False) 
GPIO.setmode(GPIO.BCM)
GPIO.setup(AIN2,GPIO.OUT)
GPIO.setup(AIN1,GPIO.OUT)
GPIO.setup(PWMA,GPIO.OUT)

GPIO.setup(BIN1,GPIO.OUT)
GPIO.setup(BIN2,GPIO.OUT)
GPIO.setup(PWMB,GPIO.OUT)

L_Motor= GPIO.PWM(PWMA,100)
L_Motor.start(0)

R_Motor = GPIO.PWM(PWMB,100)
R_Motor.start(0)

#connect with the condition that car moves
enStop = 0
enRun = 1
enBack = 2
enLeft = 3
enRight = 4
car_State = 0
symbolPos = []

last_qr_data = 'no qrcode'
global qr_data

def draw_rect(img, pos, color, width):
    cv2.line(img, pos[0], pos[1], color, width)
    cv2.line(img, pos[0], pos[3], color, width)
    cv2.line(img, pos[2], pos[1], color, width)
    cv2.line(img, pos[2], pos[3], color, width)

def qr_scan_decode(img):
    global qr_data
    global last_qr_data
    pil= Image.fromarray(img).convert('L')  # gray
    width, height = pil.size
    raw = pil.tobytes()
    zarimage = zbar.Image(width, height, 'Y800', raw)
    scanner_Flag = scanner.scan(zarimage)
    if scanner_Flag == 1:
        for symbol in zarimage:
            if not symbol.count:
                print 'decoded', symbol.type, 'symbol', '"%s"' % symbol.data
                symbolPos = symbol.location
                draw_rect(img, symbolPos, (0,255,0), 3)
                qr_data = str(symbol.data)
            else:
                qr_data = 'no qrcode'
    else:
        qr_data = 'no qrcode'
    if last_qr_data != qr_data:       
        command_resolve(qr_data)
    last_qr_data = qr_data

def command_resolve(command):
    last_car_state = 0
    if command.find("Run",0,len(command)) != -1 :
        command.zfill(len(command))
        car_state = 1
    elif command.find("Back",0,len(command)) != -1 :
        command.zfill(len(command))
        car_state = 2
    elif command.find("Left",0,len(command)) != -1 :
        command.zfill(len(command))
        car_state = 3
    elif command.find("Right",0,len(command)) != -1:
        command.zfill(len(command))
        car_state = 4
    elif command.find("Stop",0,len(command)) != -1:            
        command.zfill(len(command))
        car_state = 0
    else:
        car_state = 5
        command.zfill(len(command))
    #print car_state
   
    if last_car_state != car_state :
        car_control(car_state)
    last_car_state = car_state

def car_control(car_state):
    
    if car_state == enRun:
        t_up(50,1)      # 前进
    elif car_state == enBack :
        t_down(50,1)    # 后退
    elif car_state == enLeft :
        t_left(50,1)    # 左转
    elif car_state == enRight :
        t_right(50,1)   # 右转
    elif car_state == enStop:
        t_stop(0)       # 停止
    else:
        t_stop(0)       # 停止

if __name__ == '__main__':
    # 创建一个QR
    scanner = zbar.ImageScanner()
    # 配置QR
    scanner.parse_config('enable')
    font = cv2.FONT_HERSHEY_SIMPLEX
    # 设置Camera
    cap = cv2.VideoCapture(0)
    if cap.isOpened()==0 :
        print("camera iserror")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

    while(True):
        grabbed, frame = cap.read()
        if not grabbed:
            break
        qr_scan_decode(frame)

        cv2.imshow('frame',frame)
        if cv2.waitKey(1) & 0xFF == 27: # 按ESC键退出
            break
# 释放Camera和GPIO
    #Run.release()
    cap.release()
    cv2.destroyAllWindows()
