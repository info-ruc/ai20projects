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

def go_ahead(speed, t):
    print('go ahead')
    left_wheels.ChangeDutyCycle(speed)
    GPIO.output(left_in_1, True)
    GPIO.output(left_in_2, False)
    
    right_wheels.ChangeDutyCycle(speed)
    GPIO.output(right_in_1, True)
    GPIO.output(right_in_2, False)

    time.sleep(t)

def go_back(speed, t):
    print('back away')
    left_wheels.ChangeDutyCycle(speed)
    GPIO.output(left_in_1, False)
    GPIO.output(left_in_2, True)
    
    right_wheels.ChangeDutyCycle(speed)
    GPIO.output(right_in_1, False)
    GPIO.output(right_in_2, True)

    time.sleep(t)

def go_stop(t):
    print('stop moving')
    left_wheels.ChangeDutyCycle(0)
    GPIO.output(left_in_1, False)
    GPIO.output(left_in_2, False)
    
    right_wheels.ChangeDutyCycle(0)
    GPIO.output(right_in_1, False)
    GPIO.output(right_in_2, False)

    time.sleep(t)

def go_left(speed, t):
    print('turn left')
    left_wheels.ChangeDutyCycle(speed)
    GPIO.output(left_in_1, False)
    GPIO.output(left_in_2, True)
    
    right_wheels.ChangeDutyCycle(speed)
    GPIO.output(right_in_1, True)
    GPIO.output(right_in_2, False)

    time.sleep(t)

def go_right(speed, t):
    print('turn right')
    left_wheels.ChangeDutyCycle(speed)
    GPIO.output(left_in_1, True)
    GPIO.output(left_in_2, False)
    
    right_wheels.ChangeDutyCycle(speed)
    GPIO.output(right_in_1, False)
    GPIO.output(right_in_2, True)

    time.sleep(t)