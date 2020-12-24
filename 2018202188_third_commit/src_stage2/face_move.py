import cv2
import numpy as np
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

# 从网络摄像头获取输入
cap = cv2.VideoCapture(0)
pwm = Adafruit_PCA9685.PCA9685()
# Configure min and max servo pulse lengths
servo_min = 150  # Min pulse length out of 4096
servo_max = 600  # Max pulse length out of 4096
# Helper function to make setting a servo pulse width simpler.
def set_servo_pulse(channel, pulse):
    pulse_length = 1000000    # 1,000,000 us per second
    pulse_length //= 60       # 60 Hz
    #print('{0}us per period'.format(pulse_length))
    pulse_length //= 4096     # 12 bits of resolution
    #print('{0}us per bit'.format(pulse_length))
    pulse *= 1000
    pulse //= pulse_length
    pwm.set_pwm(channel, 0, pulse)

def set_servo_angle(channel,angle):
    angle=4096*((angle*11)+500)/20000
    pwm.set_pwm(channel,0,int(angle))

# 频率设置为50hz，适用于舵机系统。
pwm.set_pwm_freq(50)
set_servo_angle(5,90)  #底座舵机
set_servo_angle(4,60)  # 顶部舵机

time.sleep(0.5)

# 将视频尺寸减小到320x240，这样rpi处理速度就会更快
cap.set(3,320)
cap.set(4,240)

#引入分类器
face_cascade = cv2.CascadeClassifier( 'face1.xml' )
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
        
def keyscan():
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

keyscan()

while True:   
    ret,frame = cap.read()
    gray= cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    #对灰度图进行.detectMultiScale()
    faces=face_cascade.detectMultiScale(gray)   
    t_stop(0)
    if len(faces)>0:
        #print('face found!')
        (x,y,w,h) = faces[0]
        cv2.rectangle(frame,(x,y),(x+h,y+w),(0,255,0),2)
        result=(x,y,w,h)
        x=result[0]+w/2
        y=result[1]+h/2
        facebool = True 
        x_p = 0
        for(x,y,w,h) in faces:
            #找到矩形的中心位置
            cv2.rectangle(frame,(x,y),(x+h,y+w),(0,255,0),2)
            result=(x,y,w,h)
            x=result[0]+w/2
            #y=result[1]+h/2 
            x_p = int(round(x))
            #print x_p
        # 设定一个范围
        x_Lower = 100
        x_Upper = 220
        # 判断X方向范围来判断机器人的运动
        if x_p > x_Lower and x_p < x_Upper:
            t_up(50,0)
        elif x_p < x_Lower:
            t_left(50,0)
        elif x_p > x_Upper:
            t_right(50,0)
                
        else:
                t_stop(0)

    # 显示窗口画面        
    cv2.imshow("capture", frame)        
    if cv2.waitKey(1) & 0xFF == ord('q'):
            cap.release()
            L_Motor.stop()
            R_Motor.stop()
            GPIO.cleanup()
            cv2.destroyAllWindows() 
            break
cap.release()
cv2.destroyAllWindows()       