import numpy as np
import cv2
import Adafruit_PCA9685
import RPi.GPIO as GPIO
import time

#pin of top and bottom servo
top_servo = 4
bottom_servo = 5

#initialize servo
pwm = Adafruit_PCA9685.PCA9685()
#min and max servo pulse
servo_min = 150
servo_max = 600

def set_servo_pulse(channel, pulse):
    pulse_length = 1000000 #1,000,000 us per second
    pulse_length //= 60 #60 Hz
    print('{0}us per period'.format(pulse_length))
    pulse_length //= 4096 # 12 bits of resolution
    print('{0}us per bit'.format(pulse_length))
    pulse *= 1000
    pulse //= pulse_length
    pwm.set_pwm(channel, 0, pulse)

def set_servo_angle(channel, angle):
    angle = 4096*((angle*11)+500)/20000
    pwm.set_pwm(channel, 0, int(angle))

#the frequency of servo is 50Hz
pwm.set_pwm_freq(50)
set_servo_angle(bottom_servo, 90) #bottom servo: 90 
set_servo_angle(top_servo, 90) #top servo: 145

