#coding:utf-8
# 调用电机控制库
from _MOTOR_ import CarMove
# python 标准库 --- sleep 用
import time
# 建立方向控制类
car = CarMove()
# 设定方向前进
car.go_forward()
# 调整电机速度
car.set_speed(20)
# 等待两秒
time.sleep(2)
# 设定方向后退
car.go_backward()
# 调整电机速度
car.set_speed(20)
# 等待两秒
time.sleep(2)