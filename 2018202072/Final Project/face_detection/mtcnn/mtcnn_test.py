import cv2
from mtcnn.core.detect import create_mtcnn_net, MtcnnDetector
from mtcnn.core.vision import vis_face
import time


if __name__ == '__main__':

    pnet, rnet, onet = create_mtcnn_net(p_model_path="./original_model/pnet_epoch.pt", r_model_path="./original_model/rnet_epoch.pt", o_model_path="./original_model/onet_epoch.pt", use_cuda=False)
    mtcnn_detector = MtcnnDetector(pnet=pnet, rnet=rnet, onet=onet, min_face_size=24)
    for i in range(150):
        img = cv2.imread("./test_"+str(i).zfill(3)+".jpg")
        #img = cv2.imread("face1.jpg")
        img_bg = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #b, g, r = cv2.split(img)
    #img2 = cv2.merge([r, g, b])
        time_start=time.time()
        bboxs, landmarks = mtcnn_detector.detect_face(img)
        time_end=time.time()
        print('total time is ',time_end-time_start)
    # print box_align
        save_name = 'out'+str(i).zfill(3)+'.jpg'
        x,y,z=vis_face(img_bg,bboxs,landmarks, save_name)
    #print(x,y,z)
