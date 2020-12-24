import cv2 as cv
import numpy as np
import math

width, height = 300, 300 #设置拍摄窗口大小
x0,y0 = 300, 100
path = 'D://Tello/img'

def get_distance(point1,point2):
    x_dis = point1[0] - point2[0]
    y_dis = point1[0] - point2[0]
    return math.hypot(x_dis,y_dis)

def fourierDesciptor(res):
    #Laplacian算子进行八邻域检测
    gray = cv.cvtColor(res, cv.COLOR_BGR2GRAY)
    dst = cv.Laplacian(gray, cv.CV_16S, ksize = 3)
    Laplacian = cv.convertScaleAbs(dst)
    contour = find_contours(Laplacian)#提取轮廓点坐标
    contour_array = contour[0][:, 0, :]#注意这里只保留区域面积最大的轮廓点坐标
    ret_np = np.ones(dst.shape, np.uint8) #创建黑色幕布
    ret = cv.drawContours(ret_np,contour[0],-1,(255,255,255),1) #绘制白色轮廓
    contours_complex = np.empty(contour_array.shape[:-1], dtype=complex)
    contours_complex.real = contour_array[:,0]#横坐标作为实数部分
    contours_complex.imag = contour_array[:,1]#纵坐标作为虚数部分
    fourier_result = np.fft.fft(contours_complex)#进行傅里叶变换
    #fourier_result = np.fft.fftshift(fourier_result)
    descirptor_in_use = truncate_descriptor(fourier_result)#截短傅里叶描述子
    print(descirptor_in_use)
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

for i in range(0, 10):
    for j in range(5, 100):
        roi = cv.imread(path + '/' + str(i) + '/' + str(2*j+1) + '.png')
        _,descirptor_in_use = fourierDesciptor(roi)
        fd_name = path + '/' + str(i) + '_fft/' + str(j+5) + '.txt'
        with open(fd_name, 'w', encoding='utf-8') as f:
            temp = abs(descirptor_in_use[1])
            for k in range(1, len(descirptor_in_use)):
                try:
                    x_record = int(100 * abs(descirptor_in_use[k]) / temp)
                except:
                    x_record = 100000
                f.write(str(x_record))
                f.write(' ')
                f.write('\n')
            print(i, '_', j, '完成')

'''frame = cv.VideoCapture(0)
frame.open(1)
while(1):
    ret,pict = frame.read()
    pict = cv.flip(pict,1)

    pict = pict[x0:x0+width,y0:y0+height]#取手势所在框图并进行处理
    key = cv.waitKey(1) & 0xFF
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

    ycrcb = cv.cvtColor(pict, cv.COLOR_BGR2YCrCb) # 分解为YUV图像,得到CR分量
    (_, cr, _) = cv.split(ycrcb)
    cr1 = cv.GaussianBlur(cr, (5, 5), 0) # 高斯滤波
    _, skin = cv.threshold(cr1, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)  # OTSU图像二值化
    #pict = cv.bitwise_and(pict,pict,mask = skin)
    #cv.imshow("skin",skin)


    #这是另一种做法，先图像与，再腐蚀膨胀
    pict = cv.bitwise_and(pict,pict,mask = skin)
    kernel = np.ones((3,3), np.uint8) #设置卷积核
    pict = cv.erode(pict, kernel) #腐蚀操作
    #cv.imshow("erosion",erosion)
    pict = cv.dilate(pict, kernel)#膨胀操作

    contour2,descriptor = fourierDesciptor(pict)

    cv.imshow("contour",contour2)

    #binaryimg = cv.Canny(pict, 50, 200) #二值化，canny检测

    #contours, _ = cv.findContours(skin.copy(), cv.RETR_TREE, cv.CHAIN_APPROX_NONE)

    #largecont  = max(contours, key = lambda contour: cv.contourArea(contour))

    #hull = cv.convexHull(largecont, returnPoints = False)
    #defects = cv.convexityDefects(largecont,hull)
    #print(defects.shape)   //是一个三元组

    #defect = 0
    #for i in range(defects.shape[0]):
        #print("i")
        #print(i)
        #s,e,f,_ = defects[i,0]
        #beg     = tuple(largecont[s][0])    #这三个值都是二维的
        #end     = tuple(largecont[e][0])
        #far     = tuple(largecont[f][0])
        #a = get_distance(beg,end)
        #b = get_distance(beg,far)
        #c = get_distance(end,far)
        #cos = (b**2+c**2-a**2)/((2*b*c)+1)
        #angle = math.acos(cos)
        #if angle <= math.pi/2:
        #    defect = defect + 1
        #    cv.circle(pict,far,3,(255,0,0),-1)
        #    cv.line(pict,beg,end,(255,0,0),1)

    if(ret):
        cv.imshow("apa",pict)

frame.release()
cv.destroyAllWindows()'''