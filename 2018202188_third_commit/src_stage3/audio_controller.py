#coding: utf-8
import pyaudio
import wave
import os
import sys
import pygame
from aip import AipSpeech
import Adafruit_PCA9685
import  RPi.GPIO as GPIO
import time

PWMA = 18
AIN1   =  22
AIN2   =  27

PWMB = 23
BIN1   = 25
BIN2  =  24

BtnPin  = 19
Gpin    = 5
Rpin    = 6

time.sleep(10)
baidu_APP_ID='23042746'
baidu_API_KEY='1fPzLAGLPPOvuMxMvGlcGsAj'
baidu_SECRET_KEY='4onmGVlLGFLlreRcbBQiHa5gRb9GVNcI'
aipSpeech=AipSpeech(baidu_APP_ID,baidu_API_KEY,baidu_SECRET_KEY)

def robot_speech(content):
    text=content
    result = aipSpeech.synthesis(text = text, 
                             options={'spd':4,'vol':9,'per':0,})
    if not isinstance(result,dict):
        with open('makerobo.mp3','wb') as f:
            f.write(result)  
    else:print(result)
    #我们利用树莓派自带的pygame
    pygame.mixer.init()
    pygame.mixer.music.load('/home/pi/CLBROBOT/makerobo.mp3')
    pygame.mixer.music.play()

pwm = Adafruit_PCA9685.PCA9685()
# Configure min and max servo pulse lengths
servo_min = 150  # Min pulse length out of 4096
servo_max = 600  # Max pulse length out of 4096
# 辅助功能，使设置舵机脉冲宽度更简单。
def set_servo_pulse(channel, pulse):
    pulse_length = 1000000    # 1,000,000 us per second
    pulse_length //= 60       # 60 Hz
    print('{0}us per period'.format(pulse_length))
    pulse_length //= 4096     # 12 bits of resolution
    print('{0}us per bit'.format(pulse_length))
    pulse *= 1000
    pulse //= pulse_length
    pwm.set_pwm(channel, 0, pulse)

def set_servo_angle(channel,angle):
    angle=4096*((angle*11)+500)/20000
    pwm.set_pwm(channel,0,int(angle))

# 频率设置为50hz，适用于舵机系统。
pwm.set_pwm_freq(50)
set_servo_angle(5,90)  # 底座舵机
set_servo_angle(4,90)  # 顶部舵机

time.sleep(0.5) 
# 树莓派小车运动函数
def t_up(speed,t_time):
        content='前进'
        robot_speech(content)
        L_Motor.ChangeDutyCycle(speed)
        GPIO.output(AIN2,False)#AIN2
        GPIO.output(AIN1,True) #AIN1

        R_Motor.ChangeDutyCycle(speed)
        GPIO.output(BIN2,False)#BIN2
        GPIO.output(BIN1,True) #BIN1
        time.sleep(t_time)
        
def t_stop(t_time):
        content='停止'
        robot_speech(content)
        L_Motor.ChangeDutyCycle(0)
        GPIO.output(AIN2,False)#AIN2
        GPIO.output(AIN1,False) #AIN1

        R_Motor.ChangeDutyCycle(0)
        GPIO.output(BIN2,False)#BIN2
        GPIO.output(BIN1,False) #BIN1
        time.sleep(t_time)
        
def t_down(speed,t_time):
        content='后退'
        robot_speech(content)
        L_Motor.ChangeDutyCycle(speed)
        GPIO.output(AIN2,True)#AIN2
        GPIO.output(AIN1,False) #AIN1

        R_Motor.ChangeDutyCycle(speed)
        GPIO.output(BIN2,True)#BIN2
        GPIO.output(BIN1,False) #BIN1
        time.sleep(t_time)

def t_left(speed,t_time):
        content='左转'
        robot_speech(content)   
        L_Motor.ChangeDutyCycle(speed)
        GPIO.output(AIN2,True)#AIN2
        GPIO.output(AIN1,False) #AIN1

        R_Motor.ChangeDutyCycle(speed)
        GPIO.output(BIN2,False)#BIN2
        GPIO.output(BIN1,True) #BIN1
        time.sleep(t_time)

def t_right(speed,t_time):
        content='右转'
        robot_speech(content) 
        L_Motor.ChangeDutyCycle(speed)
        GPIO.output(AIN2,False)#AIN2
        GPIO.output(AIN1,True) #AIN1

        R_Motor.ChangeDutyCycle(speed)
        GPIO.output(BIN2,True)#BIN2
        GPIO.output(BIN1,False) #BIN1
        time.sleep(t_time)  
        
def keysacn():
    val = GPIO.input(BtnPin)
    while GPIO.input(BtnPin) == False:
        val = GPIO.input(BtnPin)
    while GPIO.input(BtnPin) == True:
        time.sleep(0.01)
        val = GPIO.input(BtnPin)
        if val == True:
            GPIO.output(Rpin,1)
            while GPIO.input(BtnPin) == False:
                GPIO.output(Rpin,0)
        else:
            GPIO.output(Rpin,0)
            
GPIO.setwarnings(False) 
GPIO.setmode(GPIO.BCM)
GPIO.setup(Gpin, GPIO.OUT)     # 设置绿色Led引脚模式输出
GPIO.setup(Rpin, GPIO.OUT)     # 设置红色Led引脚模式输出
GPIO.setup(BtnPin, GPIO.IN, pull_up_down=GPIO.PUD_UP)    # 设置输入BtnPin模式，拉高至高电平(3.3V) 
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


CHUNK = 512
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
RECORD_SECONDS = 5
WAVE_OUTPUT_FILENAME = "output.wav"

def get_file_content(filePath):
    with open(filePath, 'rb') as fp:
        return fp.read()

count = 1
while(True):
    content='开始运行请下达指令'
    robot_speech(content)
    time.sleep(2)

    p = pyaudio.PyAudio()

    stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK)

    print("recording...")

    frames = []

    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)

    print("done")

    stream.stop_stream()
    stream.close()
    p.terminate()

    wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()

#pygame.mixer.init()
#pygame.mixer.music.load('/home/pi/CLBROBOT/lhmqcg2.mp3')
#pygame.mixer.music.play()

    result = aipSpeech.asr(get_file_content('output.wav'), 'wav', 16000, {
            'dev_pid': 1537,
            })
    if result["err_msg"] == "success.": 
        #print(result["result"])
        print(result["result"])
        if(count%4==1):
            t_up(50,3)
        if(count%4==2):
            t_left(50,3)
        if(count%4==3):
            t_down(50,3)
        if(count%4==0):
            t_down(50,3)
            t_stop(3)
    #pygame.mixer.music.load('/home/pi/CLBROBOT/wang.mp3')
    #pygame.mixer.music.play()
   
    else:
        print("bad")
       
    count=count+1