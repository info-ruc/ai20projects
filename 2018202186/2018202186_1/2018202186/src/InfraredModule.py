#coding:utf-8
import RPi.GPIO as GPIO
import time
import sys

#the action of wheels
#index 1 for forward voltage to force forward rotation to go ahead
#index 2 for reverse voltage to force reverse rotation to go back
#left wheels
left_pwm = 18
left_in_1 = 22
left_in_2 = 27

#right wheels
right_pwm = 23
right_in_1 = 25
right_in_2 = 24

#initialize PWM of wheels
#left_wheels = GPIO.PWM(left_pwm, 100)
#left_wheels.start(0) 
#right_wheels = GPIO.PWM(right_pwm, 100)
#right_wheels.start(0)

def setup_wheels():
    print('setup wheels')
    GPIO.setup(left_in_1, GPIO.OUT)
    GPIO.setup(left_in_2, GPIO.OUT)
    GPIO.setup(left_pwm, GPIO.OUT)
    GPIO.setup(right_in_1, GPIO.OUT)
    GPIO.setup(right_in_2, GPIO.OUT)
    GPIO.setup(right_pwm, GPIO.OUT)

    #initialize PWM of wheels
    global left_wheels
    global right_wheels
    left_wheels = GPIO.PWM(left_pwm, 100)
    left_wheels.start(0) 
    right_wheels = GPIO.PWM(right_pwm, 100)
    right_wheels.start(0)

#we can just use function 'move_car' and 'wheels_rotation'
#to realize the movement of the car
#however, we can also use the other 5 functions to control
#but they are a littile complicated
def move_car(speed, t, cmd):
    print('car move: ' + cmd)
    movement = {'ahead':(True, False, True, False),
                'back':(False, True, False, True),
                'stop':(False, False, False, False),
                'left':(False, True, True, False),
                'right':(True, False, False, True)}
    wheels_rotation(speed, t, movement[cmd])

def wheels_rotation(speed, t, cmd):
    left_wheels.ChangeDutyCycle(speed)
    GPIO.output(left_in_1, cmd[0])
    GPIO.output(left_in_2, cmd[1])
    
    right_wheels.ChangeDutyCycle(speed)
    GPIO.output(right_in_1, cmd[2])
    GPIO.output(right_in_2, cmd[3])

    time.sleep(t)

#the action of infrared sensor
left_infrared = 12
right_infrared = 16

def setup_infrared():
    print('setup infrared sensors')
    GPIO.setup(left_infrared, GPIO.IN)
    GPIO.setup(right_infrared, GPIO.IN)

def infrared_obstacle_avoidance(speed = 30):
    try:
        while True:
            left = GPIO.input(left_infrared)
            right = GPIO.input(right_infrared)
            if left==True and right==True:
                move_car(speed, 0, 'ahead')
            elif left==True and right==False:
                move_car(speed, 0, 'left')
            elif left==False and right==True:
                move_car(speed, 0, 'right')
            else:
                move_car(0, 0, 'stop')
                move_car(speed, 1, 'back')
                move_car(speed, 0, 'left')
    except KeyboardInterrupt:
        right_wheels.stop()
        left_wheels.stop()
        GPIO.cleanup()

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


def setup():
    GPIO.setwarnings(False)
    GPIO.setmode(GPIO.BCM)
    setup_wheels()
    setup_button()
    setup_infrared()

if __name__ == '__main__':
    button_scan()
    infrared_obstacle_avoidance(30)