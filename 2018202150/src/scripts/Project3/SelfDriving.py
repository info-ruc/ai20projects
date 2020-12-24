#coding:utf-8

from _MOTOR_ import CarMove
from Maze import MazeFindPath
from TrafficsignClf import TrafficsignClf
import os
import time
import cv2

# 方向类
class Direction:
    North = "North"
    South = "South"
    East = "East"
    West = "West"

# 命令
class Commands:
    # 驾驶
    Speed_Limit_30 = 'speed_limit_30'
    Speed_Limit_40 = 'speed_limit_40'
    Go_Straight = 'go_straight'
    Turn_Left = 'turn_left'
    Turn_Right = 'turn_right'
    Turn_Around = 'turn_around'
    Slow = 'slow'
    Stop = 'stop'

cap = cv2.VideoCapture(1)

# 自动驾驶小车类
# 最上层的抽象类
class SelfDrivingCar:
    global cap
    # 构造函数
    # 参数: 
    #       model_path: 交通标志识别模型
    def __init__(self, model_path):
        self.car = CarMove()
        self.start_engine(model_path)
    # 人类驾驶
    # 应该返回方向盘等人类操纵的命令
    def get_human_driving_command(self):
        human_commands = []
        '''
            此处是记录方向盘、油门刹车等数据
        '''
        command = ''
        command = input("输入指令以模拟人类驾驶: ")
        if command != '':
            human_commands.append(command)
        return human_commands

    def get_self_driving_command(self):
        # 从摄像头获取画面
        ret, frame = cap.read()
        if ret:
            # 暂时存储摄像头拍摄到的帧至此
            name = os.getcwd() + '/frame.jpg'
            cv2.imwrite(name, frame)
            _, sign = self.tfsClf.predict(name)
            print(sign)
            # 预测完后就可以删除图片了
            os.remove(name)
            #cv2.imshow('frame', frame)
            # self.cap.release()
            return [sign]
        else:
            #self.cap.release()
            return [Commands.Stop]

    def get_path_finding_command(self):
        # 获取“迷宫”（即附近的道路）信息
        # 获取方向
        # 在真实项目中，应该借助GPS、指南针等数据
        def get_maze_and_direction():
            maze = []
            dorm_maze = [
                [-1,  0,  0,  0,  0, -1,  0,  2],
                [-1,  0,  0, -1, -1, -1, -1,  0],
                [-1,  0,  0,  0,  0,  0,  0,  0],
                [-1,  0,  0, -1, -1, -1, -1,  0],
                [-1,  0,  0,  0, -1, -1, -1, -1],
                [-1, -1,  0, -1, -1, -1, -1, -1],
                [-1, -1,  0, -1, -1, -1, -1, -1],
                [-1, -1,  0, -1, -1, -1, -1, -1],
                [-1,  0,  0,  0, -1, -1, -1, -1],
                [-1,  0, -1,  0, -1, -1, -1, -1],
                [ 1,  0,  0,  0, -1, -1, -1, -1]
            ]
            direction = 'East'
            '''
                此处利用外界数据计算迷宫的情况
                以及车辆方向
            '''
            # return maze, direction
            return dorm_maze, direction
        # 默认设置速度
        commands = [Commands.Speed_Limit_40]
        # 获得路况和当前方向
        maze, nowDirection = get_maze_and_direction()
        paths = MazeFindPath(maze)
        # 无路可走
        if len(paths) == 0:
            print("No path.")
            commands.append(Commands.Stop)
            return commands

        # 最短路
        path = paths[0]
        for p in paths:
            if len(p) < len(path):
                path = p
        # 变化
        delta_path = [(path[j][0] - path[j - 1][0], path[j][1] - path[j - 1][1]) for j in range(1, len(path))]
        for move in delta_path:
        # 最多考虑三步
        # for move in delta_path[0: 3]:
            # 向“南”
            if move == (1, 0):
                if nowDirection == Direction.North:
                    # 180度转弯
                    commands.append(Commands.Turn_Around)
                elif nowDirection == Direction.South:
                    # 前进
                    commands.append(Commands.Go_Straight)
                elif nowDirection == Direction.East:
                    # 右转
                    commands.append(Commands.Turn_Right)
                elif nowDirection == Direction.West:
                    # 左转
                    commands.append(Commands.Turn_Left)
                nowDirection = Direction.South
            # 向“北”
            elif move == (-1, 0):
                if nowDirection == Direction.North:
                    commands.append(Commands.Go_Straight)
                elif nowDirection == Direction.South:
                    commands.append(Commands.Turn_Around)
                elif nowDirection == Direction.East:
                    commands.append(Commands.Turn_Left)
                elif nowDirection == Direction.West:
                    commands.append(Commands.Turn_Right)
                nowDirection = Direction.North
            # 向“东”
            elif move == (0, 1):
                if nowDirection == Direction.East:
                    commands.append(Commands.Go_Straight)
                elif nowDirection == Direction.West:
                    commands.append(Commands.Turn_Around)
                elif nowDirection == Direction.North:
                    commands.append(Commands.Turn_Right)
                elif nowDirection == Direction.South:
                    commands.append(Commands.Turn_Left)
                nowDirection = Direction.East
            # 向“西”
            elif move == (0, -1):
                if nowDirection == Direction.West:
                    commands.append(Commands.Go_Straight)
                elif nowDirection == Direction.East:
                    commands.append(Commands.Turn_Around)
                elif nowDirection == Direction.North:
                    commands.append(Commands.Turn_Left)
                elif nowDirection == Direction.South:
                    commands.append(Commands.Turn_Right)
                nowDirection = Direction.East
            else:
                print("Error")
        return commands
    
    def drive_according_to_command(self, commands):
        # 一个个命令解析
        for command in commands:
            # 速度限制
            if command == Commands.Speed_Limit_30:
                self.car.set_speed(30)
                self.car.go_forward()
                # time.sleep(1)
            elif command == Commands.Speed_Limit_40:
                self.car.set_speed(40)
                self.car.go_forward()
            # 转弯
            elif command == Commands.Go_Straight:
                self.car.set_speed(50)
                time.sleep(1)
            elif command == Commands.Turn_Left:
                self.car.set_speed(40)
                self.car.turn(40, 1)
                self.car.resume_speed()
            elif command == Commands.Turn_Right:
                self.car.set_speed(40)
                self.car.turn(140, 1)
                self.car.resume_speed()
            elif command == Commands.Turn_Around:
                self.car.set_speed(40)
                self.car.inplace_turn(6, 1, 140)
                self.car.resume_speed()
            # 减速
            elif command == Commands.Slow:
                self.car.set_speed(20)
                time.sleep(1)
                self.car.resume_speed()
            # 停车
            elif command == Commands.Stop:
                self.car.stop()
                return False
            elif command == 'N/A':
                self.car.set_speed(20)
                pass
            # 不识别
            else:
                print('Illegal instruction!')
                print(command)
                self.car.stop()
                return False
        return True

    def start_engine(self, model_path):
        self.tfsClf = TrafficsignClf(model_path)
        # 设置初始驾驶状态为True
        driving_status = True
        while driving_status:
            #commands = [self.get_human_driving_command(), \
            #    self.get_self_driving_command(), \
            #    self.get_path_finding_command()]
            commands = [ [], self.get_self_driving_command(), [] ]
            if len(commands[0]) > 0:
                driving_status = self.drive_according_to_command(commands[0])
            elif len(commands[1]) > 0:
                driving_status = self.drive_according_to_command(commands[1])
            elif len(commands[2]) > 0:
                driving_status = self.drive_according_to_command(commands[2]) 
            else:
                driving_status = False
        self.car.stop()
        cap.release()

if __name__ == '__main__':
    model_path = os.getcwd() + '/model_resave.pth'
    car = SelfDrivingCar(model_path)