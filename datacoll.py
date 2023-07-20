import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time

cap =cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)


tweak = 20
bgsize = 300

folder = "data/sarath"
counter = 0

while True:

    success,img = cap.read()
    hands,img = detector.findHands(img)
    if hands:
        hand= hands[0]
        x, y, w, h = hand['bbox']

        wbg = np.ones((bgsize,bgsize,3), np.uint8)*255

        imagecrop = img[y-tweak:y+h+tweak , x-tweak:x+w+tweak]

        ics = imagecrop.shape


        aspectratio = h/w

        if aspectratio > 1:
            k = bgsize/h
            wc = math.ceil(k*w)
            imgresize = cv2.resize(imagecrop,(wc,bgsize))
            icsrez = imgresize.shape
            wgap = math.ceil((bgsize-wc)/2)
            wbg[:, wgap:wc+wgap] = imgresize

        else:
            k = bgsize / w
            hc = math.ceil(k * h)
            imgresize = cv2.resize(imagecrop, (bgsize, hc))
            icsrez = imgresize.shape
            hgap = math.ceil((bgsize - hc) / 2)
            wbg[hgap:hc + hgap,:] = imgresize

        cv2.imshow("imagecrop", imagecrop)
        cv2.imshow("white bg", wbg)


    cv2.imshow("image", img)
    key = cv2.waitKey(1)
    if key == ord("s"):
        counter += 1
        cv2.imwrite(f'{folder}/image_{time.time()}.jpg',img)
        print(counter)