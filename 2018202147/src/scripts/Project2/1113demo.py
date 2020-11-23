import cv2
import os

from TrafficsignClf import TrafficsignClf

if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    framecount = 0
    # 暂时存储摄像头拍摄到的帧至此
    save_dir = '/Users/CuiGuanyu/Desktop'
    # 模型路径
    path = os.getcwd() + '/model.pth'
    # 从模型构建分类器
    clf = TrafficsignClf(path)
    while True:
        # 从摄像头获取画面
        ret, frame = cap.read()
        if ret:
            framecount += 1
            name = save_dir + '/frame' + str(framecount) + '.jpg'
            cv2.imwrite(name, frame)
            label, sign = clf.predict(name)
            print(sign)
            # 预测完后就可以删除图片了
            os.remove(name)
            cv2.imshow('frame', frame)
            # 按下 q 以退出
            if cv2.waitKey(500) & 0xff == ord('q'):
                break
        else:
            break
    cap.release()
    cv2.destroyAllWindows() 