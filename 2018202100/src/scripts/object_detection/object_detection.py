import cv2
import numpy as np
from threading import Thread
import time
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"] = '0'

#print(os.environ)

net = cv2.dnn.readNet('config/yolov4.weights', 'config/yolov4.cfg')
#net = cv2.dnn.readNet('yolov3-spp.weights', 'yolov3.cfg')
classes = []
with open('coco.names', 'r') as f:
    classes = f.read().splitlines()

#url = 'http://admin:admin@10.47.156.160:8081/'
capture = cv2.VideoCapture(0)#0为默认摄像头x

def object_detection():
    while True:
        start = time.clock()
        ret, img = capture.read()
        # fps = 
        # img = cv2.imread('image.jpg')

        hight, width, _ = img.shape
        width = int(2 * width)
        hight = int(2 * hight)
        img = cv2.resize(img, (width,hight))

        blob = cv2.dnn.blobFromImage(img, 1 / 255, (416, 416), (0, 0, 0), swapRB=True, crop=False)

        net.setInput(blob)

        output_layers_names = net.getUnconnectedOutLayersNames()
        layerOutputs = net.forward(output_layers_names)

        boxes = []
        confidences = []
        class_ids = []

        for output in layerOutputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
                    center_x = int(detection[0]*width)
                    center_y = int(detection[1]*hight)
                    w = int(detection[2] * width)
                    h = int(detection[3] * hight)
                    
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

        font = cv2.FONT_HERSHEY_PLAIN
        colors = np.random.uniform(0, 255, size=(len(boxes),3))

        if len(indexes) > 0:
            for i in indexes.flatten():
                x, y, w, h = boxes[i]
                label = str(classes[class_ids[i]])
                confidence = str(round(confidences[i], 2))
                color = colors[i]
                cv2.rectangle(img, (x, y), (x + w, y + h), color, 4)
                cv2.putText(img, label+" "+confidence, (x,y-10), font, 1, color, 2)

        cv2.imshow('Image', img)
        end = time.clock()
        print(end - start)
        if cv2.waitKey(1) & 0xFF == ord('q'):
           break

    capture.release()
    cv2.destroyAllWindows()

Thread(target=object_detection).start()
'''


#ret, img = capture.read()
# fps = 
img = cv2.imread('image.jpg')

hight, width, _ = img.shape
#width = int(0.5 * width)
#hight = int(0.5 * hight)
img = cv2.resize(img, (width,hight))
start = time.clock()
blob = cv2.dnn.blobFromImage(img, 1 / 255, (416, 416), (0, 0, 0), swapRB=True, crop=False)

net.setInput(blob)

output_layers_names = net.getUnconnectedOutLayersNames()
layerOutputs = net.forward(output_layers_names)

boxes = []
confidences = []
class_ids = []

for output in layerOutputs:
    for detection in output:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > 0.5:
            center_x = int(detection[0]*width)
            center_y = int(detection[1]*hight)
            w = int(detection[2] * width)
            h = int(detection[3] * hight)
            
            x = int(center_x - w / 2)
            y = int(center_y - h / 2)
            
            boxes.append([x, y, w, h])
            confidences.append(float(confidence))
            class_ids.append(class_id)

indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

font = cv2.FONT_HERSHEY_PLAIN
colors = np.random.uniform(0, 255, size=(len(boxes),3))

if len(indexes) > 0:
    for i in indexes.flatten():
        x, y, w, h = boxes[i]
        label = str(classes[class_ids[i]])
        confidence = str(round(confidences[i], 2))
        color = colors[i]
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        cv2.putText(img, label+" "+confidence, (x,y-10), font, 1, color, 1)
end = time.clock()
print(end - start)
cv2.imshow('Image', img)
cv2.waitKey(0)
'''