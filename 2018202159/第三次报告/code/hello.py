import numpy as np
import cv2 as cv
from os import listdir
from sklearn.externals import joblib
from functools import reduce
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import math
import time
from easytello import tello

mytello = tello.Tello()

start_time = 0
start_flag = 0
run_flag = 0
end_time = 0
MIN_TIME = 1
MAX_TIME = 2
mode = 0
design = []

model_path = "D://Tello/model3/"
width, height = 300, 300 #设置拍摄窗口大小
x0,y0 = 55, 320

#程序希望首先每单位时间判断一次手势，每十次投票得到期间的真实手势
#我首先实现简单功能好了，预定义路径

#这个函数用于得到数组中的众数
def find_max(list):
    count = [0,0,0,0,0,0,0,0,0,0]
    for i in range(200):
        count[int(list[i])]+=1
    j = count[0]
    ret = 0
    for i in range(10):
        if count[i] > j:
            j = count[i]
            ret = i
    return ret

#这个函数用于响应指令
def solve_order(order):
    print("The order is:")
    #前进200cm，远地自转一圈并返回
    if order == 0:
        print(order)
        #mytello.takeoff()
        mytello.forward(160)
        mytello.cw(360)
        mytello.back(160)
        #mytello.land()

    #转向——完成一周行进（矩形）    
    elif order == 1:
        print(order)
        mytello.cw(90)
        mytello.forward(160)
        mytello.ccw(90)
        mytello.up(160)
        mytello.ccw(90)
        mytello.forward(160)
        mytello.cw(90)
        mytello.down(160)

    #圆行进，半径2m
    elif order == 2:
        print(order)
        mytello.curve(0,-160,160,0,0,320,50)
        mytello.curve(0,160,-160,0,0,-320,50)
        
    #八字    
    elif order == 3:
        print(order)
        mytello.curve(0,-80,80,0,0,160,50)
        mytello.curve(0,80,80,0,0,160,50)
        mytello.curve(0,-80,-80,0,0,-160,50)
        mytello.curve(0,80,-80,0,0,-160,50)

    #心
    elif order == 5:
        print(order)
        mytello.go(0,-160,160,50)
        mytello.curve(0,80,80,0,160,0,50)
        mytello.curve(0,80,80,0,160,0,50)
        mytello.go(0,-160,-160,50)
        
    #暴力关机，慎用
    elif order == 4:
        print(order)
        mytello.emergency()
        
    elif order == 6:
        print(order)
    elif order == 7:
        print(order)
    elif order == 8:
        print(order)
    else:
        print(order)

def solve_design(design):
    for i in range(len(design)):
        #用不了的手势1
        if design[i] == 0:
            print(design[i])
        
        #向前1m
        elif design[i] == 1:
            print(design[i])
            mytello.forward(100)

        #向后1m
        elif design[i] == 2:
            print(design[i])
            mytello.back(100)

        #顺时针90
        elif design[i] == 3:
            print(design[i])
            mytello.cw(90)

        #逆时针90
        elif design[i] == 4:
            print(design[i])
            mytello.ccw(90)

        #向上1m
        elif design[i] == 5:
            print(design[i])
            mytello.up(100)

        #向下1m
        elif design[i] == 6:
            print(design[i])
            mytello.down(100)

        #剩下的玩家请自定义
        elif design[i] == 7:
            print(design[i])
        elif design[i] == 8:
            print(design[i])
        else:
            print(design[i])

    #重调任务队列，使回到原点
    #玩家二次开发时请谨记修改此循环和下面的循环
    for i in range(len(design)):
        if design[i] == 1:
            design[i] = 2
        elif design[i] == 2:
            design[i] = 1
        elif design[i] == 3:
            design[i] = 4
        elif design[i] == 4:
            design[i] = 3
        elif design[i] == 5:
            design[i] = 6
        elif design[i] == 6:
            design[i] = 5
    
    #重做一遍design以回到原点
    #别忘了修改我
    for i in range(len(design)):
        #用不了的手势1
        if design[i] == 0:
            print(design[i])
        
        #向前1m
        elif design[i] == 1:
            print(design[i])
            mytello.forward(100)

        #向后1m
        elif design[i] == 2:
            print(design[i])
            mytello.back(100)

        #顺时针90
        elif design[i] == 3:
            print(design[i])
            mytello.cw(90)

        #逆时针90
        elif design[i] == 4:
            print(design[i])
            mytello.ccw(90)

        #向上1m
        elif design[i] == 5:
            print(design[i])
            mytello.up(100)

        #向下1m
        elif design[i] == 6:
            print(design[i])
            mytello.down(100)


def get_distance(point1,point2):
    x_dis = point1[0] - point2[0]
    y_dis = point1[0] - point2[0]
    return math.hypot(x_dis,y_dis)

def fourierDesciptor(res):
    #Laplacian算子进行八邻域检测
    gray = cv.cvtColor(res, cv.COLOR_BGR2GRAY)
    #cv.imshow("gray",gray)
    dst = cv.Laplacian(gray, cv.CV_16S, ksize = 3)
    #cv.imshow("dst",dst)
    Laplacian = cv.convertScaleAbs(dst)
    #cv.imshow("Lap",Laplacian)
    contour = find_contours(Laplacian)#提取轮廓点坐标
    contour_array = contour[0][:, 0, :]#注意这里只保留区域面积最大的轮廓点坐标
    #print(contour_array)
    ret_np = np.ones(dst.shape, np.uint8) #创建黑色幕布
    #print(contour)
    ret = cv.drawContours(ret_np,contour[0],-1,(255,255,255),1) #绘制白色轮廓
    cv.imshow("after",ret)
    contours_complex = np.empty(contour_array.shape[:-1], dtype=complex)
    contours_complex.real = contour_array[:,0]#横坐标作为实数部分
    contours_complex.imag = contour_array[:,1]#纵坐标作为虚数部分
    fourier_result = np.fft.fft(contours_complex)#进行傅里叶变换
    #fourier_result = np.fft.fftshift(fourier_result)
    descirptor_in_use = truncate_descriptor(fourier_result)#截短傅里叶描述子
    #print(descirptor_in_use)
    #reconstruct(ret, descirptor_in_use)
    return ret, descirptor_in_use
 
def find_contours(Laplacian):
    #binaryimg = cv2.Canny(res, 50, 200) #二值化，canny检测
    h,_ = cv.findContours(Laplacian,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_NONE) #寻找轮廓
    contour = h
    contour = sorted(contour, key = cv.contourArea, reverse=True)#对一系列轮廓点坐标按它们围成的区域面积进行排序
    return contour


def truncate_descriptor(fourier_result):
    descriptors_in_use = np.fft.fftshift(fourier_result)
    #取中间的MIN_DESCRIPTOR项描述子
    center_index = int(len(descriptors_in_use) / 2)
    low, high = center_index - int(32 / 2), center_index + int(32 / 2)
    descriptors_in_use = descriptors_in_use[low:high]
    descriptors_in_use = np.fft.ifftshift(descriptors_in_use)
    return descriptors_in_use

if __name__ == '__main__':
    mytello.streamon()
    #mytello.land()
    #mytello.takeoff()
    orders = []         #记录指令（含历史）
    captures = []       #记录间隔内投票选手
    second_count = 0    #记录响应间隔
    power_up = 0
    power_down = 2
    ans_flag = 0    #记录是否应该响应
    ans_pos = 0     #记录该响应哪个指令     #暂时不需要
    frame = cv.VideoCapture('udp://'+'192.168.10.1'+':11111')

    #载入模型
    clf = joblib.load(model_path + "svm_efd_" + "train_model.m")
    n = 0
    kk = 0
    while(1):
        if type(frame) != None:
            #这个是用来记录实时手势的信息
            feature = np.zeros((1,31))

            #首先调整屏幕位置（视角）
            ret,pict = frame.read()
            pict = cv.flip(pict,1)
            cv.imshow('origin_pict',pict)
            pict = pict[x0:x0+width,y0:y0+height]#取手势所在框图并进行处理
            cv.imshow('recg_pict',pict)
            key = cv.waitKey(5) & 0xFF
            if key == ord('i'):
                y0 += 5
            elif key == ord('k'):
	            y0 -= 5
            elif key == ord('l'):
    	        x0 += 5
            elif key == ord('j'):
	            x0 -= 5
            elif key == ord('q'):
                break
            elif key == ord('s'):
                start_flag = 1
            elif key == ord('r'):
                start_flag = 0
            elif key == ord('f'):
                mytello.takeoff()
                mytello.up(70)
            elif key == ord('w'):
                print(mytello.get_battery())
        
            #下面这一段代码用来处理输入图像
            ycrcb = cv.cvtColor(pict, cv.COLOR_BGR2YCrCb)
            (_, cr, _) = cv.split(ycrcb)
            cr1 = cv.GaussianBlur(cr, (5, 5), 0)
            _, skin = cv.threshold(cr1, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
            pict = cv.bitwise_and(pict,pict,mask = skin)
            kernel = np.ones((3,3), np.uint8)
            pict = cv.erode(pict, kernel)
            pict = cv.dilate(pict, kernel)

            _,descirptor_in_use = fourierDesciptor(pict)
            temp = abs(descirptor_in_use[1])
            i = 0
            #print(len(descirptor_in_use))
            for k in range(1, len(descirptor_in_use)):
                try:
                    x_record = int(100 * abs(descirptor_in_use[k]) / temp)
                except:
                    x_record = 100000
                feature[0,i] = int(x_record)
                i = i+1


            #下面这一部分代码用于重新创建数据集，毕竟数据集挖不干净很恶心
            #c是记录数据集，v是增加数字
            #由于编写者脑子有问题，手势i其实被记录为i-1
            if key == ord('c'):
                with open("D://Tello/img3/"+str(kk)+"_fft/" + str(n+1) + ".txt",'w', encoding='utf-8') as f:
                    for line in range(1,32):
                        f.write(str(int(feature[0][line-1])))
                        f.write(' ')
                        f.write('\n')
                n = n+1
                print(feature)
                print(n)
            elif key == ord('v'):
                n = 0
                kk = kk+1
                print(kk)
            #print(feature)
            valTest = clf.predict(feature)
            #print(valTest)
            #print(second_count)

            #展示图片，另一张图片在之前的函数里展示
            cv.imshow("test",pict)

            #用于控制接收命令后一段时间内不接收命令
            end_time = time.time()
            if start_flag == 1 and run_flag == 1 and (end_time - start_time > MAX_TIME or MIN_TIME > end_time - start_time):
                continue
            else:
                run_flag = 0

            #判断手势
            if second_count < 200:
                captures.append(valTest)
                second_count = second_count + 1
            else:
                an_order = find_max(captures)
                captures.clear()
                second_count = 0
                orders.append(an_order)
                print(orders[-1])
                ans_flag = 1

            #如果没有定义mode但已经启动程序，就先判断mode，手势5表示预定义，9表示自定义
            if ans_flag == 1 and mode == 0 and start_flag == 1:
                if orders[-1] == 4:
                    print('进入预定义模式')
                    mode = 1
                elif orders[-1] == 8:
                    print('进入自定义模式')
                    mode = 2
            elif ans_flag == 1 and mode == 1 and start_flag == 1:
                print('预定义判断')
                if orders[-1] != orders[-2]:
                    start_time = time.time()
                    run_flag = 1
                    solve_order(orders[-1])
            #自定义要求以1开始，以1结束
            elif ans_flag == 1 and mode == 2 and start_flag == 1:
                print('自定义指令判断')
                if len(design) == 0 and orders[-1] == 0:
                    print('自定义指令开始')
                    design.append(orders[-1])
                else:
                    if len(design) != 0 and orders[-1] != design[-1]:
                        print('自定义指令增加')
                        design.append(orders[-1])
            if mode == 2 and len(design) != 0 and len(design) != 1 and design[-1] == 0:
                print('执行自定义指令')
                start_time = time.time()
                run_flag = 1
                solve_design(design)
                design.clear()
            ans_flag = 0

    mytello.land()
    frame.release()
    cv.destroyAllWindows()