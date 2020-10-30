#coding:utf-8
'''
文件名：_MOTOR_.py
功能：电机方向控制函数
外部调用：
from _MOTOR_ import CarMove
move = CarMove()
'''

# GPIO
import _GPIO_ as IO
import time
# 全局变量
import _GLOBAL_VARIABLE_ as GLOBAL
# 转向舵机
from _XiaoRGEEK_SERVO_ import XR_Servo
'''
from _XiaoRGEEK_SERVO_ import XR_Servo
Servo = XR_Servo()
Servo.XiaoRGEEK_SetServoAngle(ServoNum,angle)#设置ServoNum号舵机角度为angle
Servo.XiaoRGEEK_SaveServo()#存储所有角度为上电初始化默认值
Servo.XiaoRGEEK_ReSetServo()#恢复所有舵机角度为保存的默认值
'''

# ---------------------------------------------
# |					内部函数					|
# ---------------------------------------------

# 右马达速度
def __Right_Motor_Speed__(speed):
	IO.ENAset(speed)

# 左马达速度
def __Left_Motor_Speed__(speed):
	IO.ENBset(speed)

class CarMove:
	def __init__(self):
		self.speeds = [100]
		self.servo = XR_Servo()
		self.servo.XiaoRGEEK_SetServoAngle(1, 90)

	def go_forward(self):
		# 设置接口，使得马达前转
		IO.GPIOSet(IO.ENA)
		IO.GPIOSet(IO.ENB)
		IO.GPIOSet(IO.IN1)
		IO.GPIOClr(IO.IN2)
		IO.GPIOSet(IO.IN3)
		IO.GPIOClr(IO.IN4)
		IO.GPIOClr(IO.LED1)
		IO.GPIOClr(IO.LED2)

	def go_backward(self):
		IO.GPIOSet(IO.ENA)
		IO.GPIOSet(IO.ENB)
		IO.GPIOClr(IO.IN1)
		IO.GPIOSet(IO.IN2)
		IO.GPIOClr(IO.IN3)
		IO.GPIOSet(IO.IN4)
		IO.GPIOSet(IO.LED1)
		IO.GPIOClr(IO.LED2)

	def turn(self, servo_angle, duration):
		self.go_forward()
		self.servo.XiaoRGEEK_SetServoAngle(1, servo_angle)
		time.sleep(duration)
		self.servo.XiaoRGEEK_SetServoAngle(1, 90)
	
	def inplace_turn(self, times, duration, forward_servo_angle, backward_servo_angle = 90):
		for _ in range(times):
			self.go_forward()
			self.servo.XiaoRGEEK_SetServoAngle(1, forward_servo_angle)
			time.sleep(duration)
			self.go_backward()
			self.servo.XiaoRGEEK_SetServoAngle(1, backward_servo_angle)
			time.sleep(duration)
		self.servo.XiaoRGEEK_SetServoAngle(1, 90)
	
	def stop(self):
		IO.GPIOClr(IO.ENA)
		IO.GPIOClr(IO.ENB)
		IO.GPIOClr(IO.IN1)
		IO.GPIOClr(IO.IN2)
		IO.GPIOClr(IO.IN3)
		IO.GPIOClr(IO.IN4)
		IO.GPIOSet(IO.LED1)
		IO.GPIOClr(IO.LED2)
	
	def set_speed(self, speed):
		__Left_Motor_Speed__(speed)
		__Right_Motor_Speed__(speed)
		self.speeds.append(speed)
	
	def resume_speed(self):
		if len(speeds) > 1:
			self.speeds.pop()
			__Left_Motor_Speed__(self.speeds[-1])
			__Right_Motor_Speed__(self.speeds[-1])
		else:
			print("Can't resume speed.")
	
	def get_speed(self):
		return self.speeds[-1]

if __name__ == '__main__':
	car = CarMove()
	car.set_speed(40)
	car.turn(50, 1)
	car.inplace_turn(3, 1, 50, 90)