import pyaudio
import wave
from aip import AipSpeech
from playsound import playsound
import socket
import threading
import time 
import sys
import re
import tellopy


APP_ID = '22828037'        #新建AiPSpeech
API_KEY = '14HDtbjYr6ILvntBwy0c1TOh'
SECRET_KEY = 'XDEirKAF4E4kxzFpCoDgUDQ0ZLfPH5f9'

client = AipSpeech(APP_ID, API_KEY, SECRET_KEY)

drone=tellopy.Tello()
drone.connect()
'''
#连接tello

tello_address = ('192.168.10.1', 8889)

#本地计算机的Ip和端口
local_address = ('', 9000)
#local_address = ('192.168.10.2', 139)

#创建一个UDP连接用于发送命令
sock = socket.socket(socket.AF_INET,socket.SOCK_DGRAM)

# 绑定到本地地址和端口
sock.bind(local_address)
'''
print('请你输入语音') 

def record():
    CHUNK = 1024        
    FORMAT = pyaudio.paInt16        #量化位数
    CHANNELS = 1                     #采样管道数
    RATE = 16000                     #采样率  
    RECORD_SECONDS = 2          
    WAVE_OUTPUT_FILENAME = "output.wav" #文件保存的名称
    p = pyaudio.PyAudio()              #创建PyAudio的实例对象
    stream = p.open(format=FORMAT,      #调用PyAudio实例对象的open方法创建流Stream
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)
    frames = []                 #存储所有读取到的数据
    print('* 开始录音 >>>')     #打印开始录音
    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)   #根据需求，调用Stream的write或者read方法
        frames.append(data)
    print('* 结束录音 >>>')    #打印结束录音
    stream.close()   #调用Stream的close方法，关闭流
    p.terminate()   #调用pyaudio.PyAudio.terminate() 关闭会话
    wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')   #写入wav文件里面
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()

def cognitive():                           #读取文件
    def get_file_content(filePath):
        with open(filePath, 'rb') as fp:
            return fp.read()

    result = client.asr(get_file_content('output.wav'), 'wav', 16000, {
        'dev_pid': 1537,                   #识别本地文件
    } )
    result_text = result["result"][0]

    print("you said: " + result_text)

    return result_text



def action():   
    if result == "开始。":
        '''
        command = "command"
        send ( command )
        '''
        print("开始")
    if result == "起飞。":
        #send("takeoff")
        drone.takeoff()
        print("tello无人机起飞")
    if result == "降落。":
        #send("land")
        drone.land()
        print("tello无人机降落")
    if result == "向上。":
        #send("up 20")             #upX,X等于多少，就向上飞多少
        drone.up(20)
        print("tello无人机向上飞")
    if result == "向下。":
        #send("down 20")
        drone.down(20)
        print("tello无人机向下飞")
    if result == "向左。":
        #send("left 20")
        drone.left(20)
        print("tello无人机向左飞")
    if result == "向右。":
        #send("right 20")         #所有值都是可以任意更改的
        drone.right(20)
        print("tello无人机向右飞")
    if result == "向前。":
        #send("forward 20")
        drone.forward(30)
        print("tello无人机向前飞")
    if result == "向后。":
        #send("back 20")
        drone.backward(30)
        print("tello无人机向后飞")
    if result == "右倾斜。":
        #send("cw 90")
        drone.set_roll(0.5)
        print("tello无人机顺时针旋转")
    if result == "左倾斜。":
        #send("cw 90")
        drone.set_roll(-0.5)
        print("tello无人机逆时针旋转")
    if result == "翻转。":
        #send("stop")
        drone.flip_right()
        print("tello无人机悬停")
    
    else:
        pass

         
while(True):
    '''
    result=input()
    if result == 'end':
        break
    else:
        send(result)
    '''
    record()
    result = cognitive()
    print(result)
    if result=="结束。":
        drone.quit()
        break
    else:
        action()
    