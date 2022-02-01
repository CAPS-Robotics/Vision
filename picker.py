import cv2
import numpy as np
import VisionTest as vision
import time

def callback(x):
    pass

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FPS, 10)
cap.set(3, 640)
cap.set(4, 480)
cv2.namedWindow('image')

ilowH = 0
ihighH = 179
ilowS = 0
ihighS = 255
ilowV = 0
ihighV = 255

# create trackbars for color change
cv2.createTrackbar('lowH','image',ilowH,179,callback)
cv2.createTrackbar('highH','image',ihighH,179,callback)

cv2.createTrackbar('lowS','image',ilowS,255,callback)
cv2.createTrackbar('highS','image',ihighS,255,callback)

cv2.createTrackbar('lowV','image',ilowV,255,callback)
cv2.createTrackbar('highV','image',ihighV,255,callback)

while 1:
    t = time.time()
    ret, frame = cap.read()

    ilowH = cv2.getTrackbarPos('lowH', 'image')
    ihighH = cv2.getTrackbarPos('highH', 'image')
    ilowS = cv2.getTrackbarPos('lowS', 'image')
    ihighS = cv2.getTrackbarPos('highS', 'image')
    ilowV = cv2.getTrackbarPos('lowV', 'image')
    ihighV = cv2.getTrackbarPos('highV', 'image')

    #hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # cv2.imshow('hsv', hsv)
    lower_hsv = np.array([ilowH, ilowS, ilowV])
    higher_hsv = np.array([ihighH, ihighS, ihighV])
    coords, img, maskedFrame = vision.FindColor(frame, lower_hsv, higher_hsv, ignored_size=500)
    #mask = cv2.inRange(hsv, lower_hsv, higher_hsv)
    #maskedFrame = cv2.bitwise_and(frame, frame, mask = mask)
    #frame = cv2.inRange(frame, lower_hsv, higher_hsv)
    cv2.imshow('mask', maskedFrame)
    cv2.imshow('image', img)
    #cv2.imshow('frame', frame)
    #print ilowH, ilowS, ilowV
    
    print(1 / (time.time() - t))
    
    if(cv2.waitKey(1) & 0xFF == ord('q')):
        break


cv2.destroyAllWindows()
cap.release()
