'''from easytello import tello

mytello = tello.Tello()
mytello.streamon()
mytello.streamoff()
'''
import cv2 as cv
frame = cv.VideoCapture(0)
frame.open(1)
while(1):
    ret,pict = frame.read()
    print(frame.isOpened())
    print(ret)
    #gray = cv.cvtColor(pict, cv.COLOR_BGR2GRAY)
    if(ret):
        cv.imshow("apa",pict)
    cv.waitKey(10)
