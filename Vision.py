import cv2
import numpy as np
import time

def FindColor(img, color, color_range, ignored_size=500, guassian_kernel_size=(21, 21)):
    #img = cv2.GaussianBlur(img, guassian_kernel_size, cv2.BORDER_DEFAULT)
    lower, upper = color_range[color]
    mask = cv2.inRange(img, lower, upper)
    mask_contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    coords = []
    for mask_contour in mask_contours:
        if cv2.contourArea(mask_contour) > ignored_size:
            x, y, w, h = cv2.boundingRect(mask_contour)
            coords.append((w//2 + x, h//2 + y, w*h))
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 0), 3)
            
    return coords, img, mask

def callback(x):
    pass

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FPS, 10)
cap.set(3, 640)
cap.set(4, 480)
cv2.namedWindow('image')

lower_red = np.array([35, 0, 91])
upper_red = np.array([76, 63, 255])

lower_blue = np.array([73, 38, 0])
upper_blue = np.array([147, 109, 54])

color_range = np.array([[lower_red, upper_red], [lower_blue, upper_blue]])

while 1:
    t = time.time()
    ret, frame = cap.read()
    
    coords, img, maskedFrame = FindColor(frame, 1, color_range, ignored_size=750)
    
    cv2.putText(img, f"FPS: {int(1 / (time.time() - t))}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    
    cv2.imshow('mask', maskedFrame)
    cv2.imshow('image', img)
    
    if(cv2.waitKey(1) & 0xFF == ord('q')):
        break


cv2.destroyAllWindows()
cap.release()