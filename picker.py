import cv2
import numpy as np
import VisionTest as vision
import time

def callback(x):
    pass

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FPS, 0)
cap.set(10, 50) #Brightness
cap.set(14, 50) #Gain
cap.set(11, 10)
cap.set(3, 1080) 
cap.set(4, 720)
cv2.namedWindow('image')

ilowH = 0
ihighH = 255#179
ilowS = 0
ihighS = 255
ilowV = 0
ihighV = 255
ignored_size = 0
filter_length = 0

# create trackbars for color change
cv2.createTrackbar('lowH','image',ilowH,255,callback)
cv2.createTrackbar('highH','image',ihighH,255,callback)

cv2.createTrackbar('lowS','image',ilowS,255,callback)
cv2.createTrackbar('highS','image',ihighS,255,callback)

cv2.createTrackbar('lowV','image',ilowV,255,callback)
cv2.createTrackbar('highV','image',ihighV,255,callback)

cv2.createTrackbar('ignored_size', 'image', ignored_size, 10000, callback)
cv2.createTrackbar('filter_length', 'image', filter_length, 1000, callback)

while 1:
    ret, frame = cap.read()

    ilowH = cv2.getTrackbarPos('lowH', 'image')
    ihighH = cv2.getTrackbarPos('highH', 'image')
    ilowS = cv2.getTrackbarPos('lowS', 'image')
    ihighS = cv2.getTrackbarPos('highS', 'image')
    ilowV = cv2.getTrackbarPos('lowV', 'image')
    ihighV = cv2.getTrackbarPos('highV', 'image')
    ignored_size = cv2.getTrackbarPos('ignored_size', 'image')
    filter_length = cv2.getTrackbarPos('filter_length', 'image')

    #hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # cv2.imshow('hsv', hsv)
    lower_hsv = np.array([ilowH, ilowS, ilowV])
    higher_hsv = np.array([ihighH, ihighS, ihighV])
    coords, img, maskedFrame = vision.FindColor(frame, lower_hsv, higher_hsv, ignored_size, filter_length)
    #mask = cv2.inRange(hsv, lower_hsv, higher_hsv)
    #maskedFrame = cv2.bitwise_and(frame, frame, mask = mask)
    #frame = cv2.inRange(frame, lower_hsv, higher_hsv)
    cv2.imshow('mask', maskedFrame)
    cv2.imshow('image', img)
    #cv2.imshow('frame', frame)
    #print ilowH, ilowS, ilowV
    
    
    if(cv2.waitKey(1) & 0xFF == ord('q')):
        break


cv2.destroyAllWindows()
cap.release()
