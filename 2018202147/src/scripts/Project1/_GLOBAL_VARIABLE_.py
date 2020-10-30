#coding:utf-8
'''
文 件 名：_GLOBAL_VARIABLE_.py
功    能：全局变量文件，部分参数可根据需求修改
'''
#
# 可手动修改的参数
servo_angle_max = 160		#舵机角度上限值，防止舵机卡死，可设置小于180的数值
servo_angle_min = 15		#舵机角度下限值，防止舵机卡死，可设置大于0的数值

#
# 禁止手动修改的参数
#
motor_flag = 1				#电机接线组合标志位，默认为1；上位机调整方向后，会下发实际标志位（1-8）
BT_Client = False
TCP_Client = False
socket_flag = 0
# 路径规划参数设置
RevStatus = 0				#状态参数
TurnAngle = 0				#运行角度参数
Golength = 0				#运行距离参数
# 上传数据参数设置
#sendbuf  = ['\xFF','00','00','00','\xFF']	#FF XX XX XX FF
#sendflag = 0				#发送标志位，0为不发送，1为发送

Path_Dect_on = 0			#摄像头巡线开始标志，0为停止，1为开始执行
Path_Dect_px = 320			#摄像头巡线中心坐标设置