import numpy as np
import cv2
import Adafruit_PCA9685
import RPi.GPIO as GPIO
import time

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
    print('go ahead')
    left_wheels.ChangeDutyCycle(speed)
    GPIO.output(left_in_1, cmd[0])
    GPIO.output(left_in_2, cmd[1])
    
    right_wheels.ChangeDutyCycle(speed)
    GPIO.output(right_in_1, cmd[2])
    GPIO.output(right_in_2, cmd[3])

    time.sleep(t)

#open and use camera
video_capture = cv2.VideoCapture(0)
video_capture.set(3, 160) #window width
video_capture.set(4, 120) #window height

speed = 30

if __name__ == "__main__":
    setup_button()
    setup_wheels()
    button_scan()
    while True:
        ret, frame = video_capture.read()
        #crop the image
        crop_img = frame[50:110, 0:160]
        #convert to grayscale
        gray_img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
        #Gaussian filtering
        filter = cv2.GaussianBlur(gray_img, (5, 5), 0)
        #color thresholding
        ret, thresh = cv2.threshold(filter, 60, 255, cv2.THRESH_BINARY_INV)
        #erode and dilate to remove accidental line detections
        mask = cv2.erode(thresh, None, interations = 2)
        mask = cv2.dilate(mask, None, interations = 2)
        #find the contours of the frame
        image, contours, hierarchy = cv2.findContours(mask.copy(), 1, cv2.CHAIN_APPROX_NONE)
        #find the biggest contour
        if len(contours) > 0:
            c = max(contours, key = cv2.contourArea)
            M = cv2.moments(c)

            #find and draw the center of line
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
            cv2.line(crop_img, (cx, 0), (cx, 720), (255, 0, 0), 1)
            cv2.line(crop_img, (0, cy), (1280, cy), (255, 0, 0), 1)

            cv2.drawContours(crop_img, contours, -1, (0, 255, 0), 1)

            if cx < 110 and cx > 50:
                move_car(speed, 0, 'ahead')
            elif cx <= 50:
                move_car(speed, 0, 'left')
            else:
                move_car(speed, 0, 'right')
        else:
            print('can\'t find the track')
        #display the resulting frame
        cv2.imshow('frame', crop_img)
        if cv2.waitKey(1) & 0xFF==ord('q'):
            left_wheels.stop()
            right_wheels.stop()
            cv2.destroyAllWindows()
            GPIO.cleanup()
            break
