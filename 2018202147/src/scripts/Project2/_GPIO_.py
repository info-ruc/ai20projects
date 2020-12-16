#coding:utf-8
'''
文 件 名：_GPIO_.py
功    能：引脚定义及初始化文件,提供IO口操作函数以及调速函数。需要设置引脚为输入/输出模式。
外部调用：
import _GPIO_ as GPIO
GPIOSet(GPIO.LED0)		#LED0引脚输出高电平
GPIOClr(GPIO.LED0)		#LED0引脚输出低电平
DigitalRead(IR_R)	#读取IR_R引脚状态
ENAset(100)			#设置ENA占空比来调速，0-100之间
ENBset(100)			#设置ENB占空比来调速，0-100之间
'''
import RPi.GPIO as GPIO
import time
#######################################
#############信号引脚定义##############
#######################################
########LED口定义#################
GPIO.setwarnings(False)
GPIO.setmode(GPIO.BCM)
LED0 = 10
LED1 = 9
LED2 = 25
########电机驱动接口定义#################
ENA = 13	#//L298使能A
ENB = 20	#//L298使能B
IN1 = 19	#//电机接口1
IN2 = 16	#//电机接口2
IN3 = 21	#//电机接口3
IN4 = 26	#//电机接口4


#########led初始化为000##########
GPIO.setup(LED0,GPIO.OUT,initial=GPIO.HIGH)
GPIO.setup(LED1,GPIO.OUT,initial=GPIO.HIGH)
GPIO.setup(LED2,GPIO.OUT,initial=GPIO.HIGH)
#########电机初始化为LOW##########
GPIO.setup(ENA,GPIO.OUT,initial=GPIO.LOW)
ENA_pwm=GPIO.PWM(ENA,1000)
ENA_pwm.start(0)
ENA_pwm.ChangeDutyCycle(100)
GPIO.setup(IN1,GPIO.OUT,initial=GPIO.LOW)
GPIO.setup(IN2,GPIO.OUT,initial=GPIO.LOW)
GPIO.setup(ENB,GPIO.OUT,initial=GPIO.LOW)
ENB_pwm=GPIO.PWM(ENB,1000)
ENB_pwm.start(0)
ENB_pwm.ChangeDutyCycle(100)
GPIO.setup(IN3,GPIO.OUT,initial=GPIO.LOW)
GPIO.setup(IN4,GPIO.OUT,initial=GPIO.LOW)


def	GPIOSet(gpio):
	GPIO.output(gpio,True)

def	GPIOClr(gpio):
	GPIO.output(gpio,False)
def	DigitalRead(gpio):
	return GPIO.input(gpio)
def ENAset(EA_num):
	ENA_pwm.ChangeDutyCycle(EA_num)
def ENBset(EB_num):
	ENB_pwm.ChangeDutyCycle(EB_num)
