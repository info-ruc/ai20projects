#coding:utf-8
import RPi.GPIO as GPIO
import time
import sys

def setup():
    GPIO.setwarnings(False)
    GPIO.setmode(GPIO.BCM)
    setup_wheels()
    setup_button()
    setup_infrared()

def destroy():
    GPIO.cleanup()

if __name__ == '__main__':
    setup()