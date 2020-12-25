import cv2
import numpy as np
import random

path = 'D://Tello/img'

def find_contours(Laplacian):
    #binaryimg = cv2.Canny(res, 50, 200) #二值化，canny检测
    h,_ = cv2.findContours(Laplacian,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE) #寻找轮廓
    contour = h
    contour = sorted(contour, key = cv2.contourArea, reverse=True)#对一系列轮廓点坐标按它们围成的区域面积进行排序
    return contour

def rotate(image, scale=0.9):
    angle = random.randrange(-90, 90)#随机角度
    w = image.shape[1]
    h = image.shape[0]
    #rotate matrix
    M = cv2.getRotationMatrix2D((w/2,h/2), angle, scale)
    #rotate
    image = cv2.warpAffine(image,M,(w,h))
    return image

'''for i in range(0, 10):
        cnt = 9
        for j in range(1, 9):
            roi = cv2.imread(path + '/' + str(i) + '/' + str(j)+'.png')
            for k in range(12):
                img_rotation = rotate(roi)#旋转
                cv2.imwrite(path + '/' + str(i) + '/' + str(cnt)+ '.png',img_rotation)
                cnt += 1
                img_flip = cv2.flip(img_rotation,1)#翻转
                cv2.imwrite(path + str(i) + '_' + str(cnt)+ '.png',img_flip)
                cnt += 1
            print(i,'_',j,'完成')
'''
pict = cv2.imread(path + '/1/2.png')
ycrcb = cv2.cvtColor(pict, cv2.COLOR_BGR2YCrCb)
(_, cr, _) = cv2.split(ycrcb)
cr1 = cv2.GaussianBlur(cr, (5, 5), 0)
_, skin = cv2.threshold(cr1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
pict = cv2.bitwise_and(pict,pict,mask = skin)
kernel = np.ones((3,3), np.uint8)
pict = cv2.erode(pict, kernel)
pict = cv2.dilate(pict, kernel)
gray = cv2.cvtColor(pict, cv2.COLOR_BGR2GRAY)
dst = cv2.Laplacian(gray, cv2.CV_16S, ksize = 3)
Laplacian = cv2.convertScaleAbs(dst)
contour = find_contours(Laplacian)#提取轮廓点坐标
contour_array = contour[0][:, 0, :]#注意这里只保留区域面积最大的轮廓点坐标
ret_np = np.ones(dst.shape, np.uint8) #创建黑色幕布
ret = cv2.drawContours(ret_np,contour[0],-1,(255,255,255),1) #绘制白色轮廓
cv2.imshow("test",ret)
cv2.waitKey(0)