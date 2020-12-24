import cv2 as cv
import random
import numpy as np

path = 'D://Tello/img'

width, height = 300, 300 #设置拍摄窗口大小
x0,y0 = 300, 100

def rotate(image, scale=0.9):
    angle = random.randrange(-90, 90)#随机角度
    w = image.shape[1]
    h = image.shape[0]
    #rotate matrix
    M = cv.getRotationMatrix2D((w/2,h/2), angle, scale)
    #rotate
    image = cv.warpAffine(image,M,(w,h))
    return image



frame = cv.VideoCapture(0)
frame.open(1)
cnt = 7
num = 0
while(1):
    ret,pict = frame.read()
    pict = cv.flip(pict,1)
    #ycrcb = cv.cvtColor(pict, cv.COLOR_BGR2YCrCb)
    #(_, cr, _) = cv.split(ycrcb)
    #cr1 = cv.GaussianBlur(cr, (5, 5), 0)
    #_, skin = cv.threshold(cr1, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    #pict = cv.bitwise_and(pict,pict,mask = skin)
    #kernel = np.ones((3,3), np.uint8)
    #pict = cv.erode(pict, kernel)
    #pict = cv.dilate(pict, kernel)
    pict = pict[x0:x0+width,y0:y0+height]#取手势所在框图并进行处理
    key = cv.waitKey(10) & 0xFF
    if key == ord('i'):
        y0 += 5
    elif key == ord('k'):
	    y0 -= 5
    elif key == ord('l'):
	    x0 += 5
    elif key == ord('j'):
	    x0 -= 5
    elif key == ord('c'):
        cnt += 1
        cv.imwrite(path + '/' + str(num) + '/' + str(cnt)+ '.png',pict)
    elif key == ord('v'):
        num += 1
        cnt = 7
    elif key == ord('q'):
        break
    ycrcb = cv.cvtColor(pict, cv.COLOR_BGR2YCrCb)
    (_, cr, _) = cv.split(ycrcb)
    cr1 = cv.GaussianBlur(cr, (5, 5), 0)
    _, skin = cv.threshold(cr1, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    pict = cv.bitwise_and(pict,pict,mask = skin)
    kernel = np.ones((3,3), np.uint8)
    pict = cv.erode(pict, kernel)
    pict = cv.dilate(pict, kernel)
    cv.imshow("test",pict)
frame.release()
cv.destroyAllWindows()