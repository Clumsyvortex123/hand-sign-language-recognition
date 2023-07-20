import cv2
import HandTrackingModule as h
import numpy as np
import math
import classifiermod as c

cam = cv2.VideoCapture(0)
detect = h.handDetector(maxHands= 1)
cls = c.Classifier("modelsign/keras_model.h5","modelsign/labels.txt")

tweak =20
bgsize = 300

folder = "data/y"
cntr = 0

labels = ["stop","peace", "F YOU","nice","handgun","fist"]

while True:
    success , img = cam.read()
    imgoutput = img.copy()
    hands, img = detect.findHands(img)
    if hands:
        hand = hands[0]
        x,y,w,h, = hand['bbox']

        wbg = np.ones((bgsize,bgsize,3),np.unit8)*255

        imagecrop = img[y-tweak:y+h+tweak, x-tweak:x+w+tweak]

        ics = imagecrop.shape

        ar = h/w

        if ar >1 :
            k = bgsize/h
            wc = math.ceil(k*w)
            imgresize = cv2.resize(imagecrop,(wc,bgsize))
            icsrez = imgresize.shape
            wgap = math.ceil((bgsize- wc)/2)
            wbg[:, wgap:wc + wgap] = imgresize
            pred, index = cls.getPrediction(wbg, draw=False)
            print(pred)
            print(index)

        else:
            k = bgsize / w
            hc = math.ceil(k * h)
            imgresize = cv2.resize(imagecrop, (bgsize, hc))
            icsrez = imgresize.shape
            hgap = math.ceil((bgsize - hc) / 2)
            wbg[hgap:hc + hgap, :] = imgresize
            pred, index = cls.getPrediction(wbg, draw=False)

        cv2.putText(imgoutput, labels[index], (x, y - 20), cv2.FONT_HERSHEY_COMPLEX, 2, (255, 0, 255), 2)
        cv2.rectangle(imgoutput, (x - tweak, y - tweak), (x + w + tweak, y + h + tweak), (255, 0, 255), 4)
        cv2.imshow("imagecrop", imagecrop)
        cv2.imshow("white bg", wbg)

    cv2.imshow("image", imgoutput)
    cv2.waitKey(1)













