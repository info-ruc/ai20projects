#coding:utf-8
from _MOTOR_ import CarMove
from Maze import MazeFindPath
import time

# 方向类
class Direction:
    North = "North"
    South = "South"
    East = "East"
    West = "West"

# 走迷宫
def DriveInMaze(maze, initial_direction):
    # 当前方向
    nowDirection = initial_direction
    # 车
    car = CarMove()
    car.set_speed(40)
    # 路
    paths = MazeFindPath(maze)
    if len(paths) == 0:
        print("No path.")
        return
    path = paths[0]
    for p in paths:
        if len(p) < len(path):
            path = p
    
    # 变化
    delta_path = [(path[j][0] - path[j - 1][0], path[j][1] - path[j - 1][1]) for j in range(1, len(path))]
    # 每一步
    for move in delta_path:
        # 向“南”
        if move == (1, 0):
            if nowDirection == Direction.North:
                # 180度转弯
                car.inplace_turn(6, 1, 140)
            elif nowDirection == Direction.South:
                # 前进
                car.go_forward()
                time.sleep(1)
            elif nowDirection == Direction.East:
                # 右转
                car.turn(140, 1)
            elif nowDirection == Direction.West:
                # 左转
                car.turn(40, 1)
            nowDirection = Direction.South
        # 向“北”
        elif move == (-1, 0):
            if nowDirection == Direction.North:
                car.go_forward()
                time.sleep(1)
            elif nowDirection == Direction.South:
                # 180度转弯
                car.inplace_turn(6, 1, 140)
            elif nowDirection == Direction.East:
                car.turn(40, 1)
            elif nowDirection == Direction.West:
                car.turn(140, 1)
            nowDirection = Direction.North
        # 向“东”
        elif move == (0, 1):
            if nowDirection == Direction.East:
                car.go_forward()
                time.sleep(1)
            elif nowDirection == Direction.West:
                # 180度转弯
                car.inplace_turn(6, 1, 140)
            elif nowDirection == Direction.North:
                car.turn(140, 1)
            elif nowDirection == Direction.South:
                car.turn(40, 1)
            nowDirection = Direction.East
        # 向“西”
        elif move == (0, -1):
            if nowDirection == Direction.West:
                car.go_forward()
                time.sleep(1)
            elif nowDirection == Direction.East:
                # 180度转弯
                car.inplace_turn(6, 1, 140)
            elif nowDirection == Direction.North:
                car.turn(40, 1)
            elif nowDirection == Direction.South:
                car.turn(140, 1)
            nowDirection = Direction.East
        else:
            print("Error")

if __name__ == "__main__":
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
    DriveInMaze(dorm_maze, "East")