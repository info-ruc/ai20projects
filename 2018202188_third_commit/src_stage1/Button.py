#coding:utf-8
import RPi.GPIO as GPIO
import time
import sys

#the action of button
button_pin = 19
green_pin = 5 #light green lamp
red_pin = 6 #light red lamp

def setup_button():
    print('setup button')
    GPIO.setup(green_pin, GPIO.OUT)
    GPIO.setup(red_pin, GPIO.OUT)
    #initialize button_pin to high level
    GPIO.setup(button_pin, GPIO.IN, pull_up_down = GPIO.PUD_UP)

def button_scan():
    #when click the button, the car start to display movement
    print('button scaning')
    value = GPIO.input(button_pin)
    while not GPIO.input(button_pin):
        value = GPIO.input(button_pin)
    while GPIO.input(button_pin):
        time.sleep(0.01)
        value = GPIO.input(button_pin)
        if value:
            GPIO.output(green_pin, True)
            while not GPIO.input(button_pin):
                GPIO.output(red_pin, False)
        else:
            GPIO.output(red_pin, False)
