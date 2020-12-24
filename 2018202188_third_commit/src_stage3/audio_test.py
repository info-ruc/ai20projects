#coding: utf-8
import pyaudio
import wave
import os
import sys
import pygame
from aip import AipSpeech
import time

baidu_APP_ID='23042746'
baidu_API_KEY='1fPzLAGLPPOvuMxMvGlcGsAj'
baidu_SECRET_KEY='4onmGVlLGFLlreRcbBQiHa5gRb9GVNcI'
baidu_aipSpeech=AipSpeech(baidu_APP_ID,baidu_API_KEY,baidu_SECRET_KEY)

time.sleep(10)
def robot_speech(content):
    text=content
    result = baidu_aipSpeech.synthesis(text = text, 
                             options={'spd':4,'vol':9,'per':0,})
    if not isinstance(result,dict):
        with open('makerobo.mp3','wb') as f:
            f.write(result)  
    else:print(result)
    #我们利用树莓派自带的pygame
    pygame.mixer.init()
    pygame.mixer.music.load('/home/pi/CLBROBOT/makerobo.mp3')
    pygame.mixer.music.play()
    
content='您好'
robot_speech(content)
time.sleep(1)

CHUNK = 512
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
RECORD_SECONDS = 2
WAVE_OUTPUT_FILENAME = "output.wav"

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

pygame.mixer.init()
#pygame.mixer.music.load('/home/pi/CLBROBOT/lhmqcg2.mp3')
#pygame.mixer.music.play()

def get_file_content(filePath):
    with open(filePath, 'rb') as fp:
        return fp.read()

result = baidu_aipSpeech.asr(get_file_content('output.wav'), 'wav', 16000, {
    'dev_pid': 1537,
})
if result["err_msg"] == "success.": 
        #print(result["result"])
   print(result["result"])
   pygame.mixer.music.load('/home/pi/CLBROBOT/wang.mp3')
   pygame.mixer.music.play()
else:
    print("bad")