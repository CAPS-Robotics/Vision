
#! /usr/bin/python3

import cv2
import numpy as np
import time
import profiler

def FindColor(img, color, color_range, ignored_size=500):
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

cap.set(cv2.CAP_PROP_FPS, 0)
cap.set(10, 50) #Brightness
cap.set(14, 50) #Gain
cap.set(11, 10)
cap.set(3, 640) 
cap.set(4, 480)
cv2.namedWindow('image')

lower_red = np.array([0, 8, 118])
upper_red = np.array([104, 85, 225])

lower_blue = np.array([57, 26, 0])
upper_blue = np.array([117, 119, 40])

lower_reflective_green = np.array([190, 226, 0])
upper_reflective_green = np.array([245, 255, 127])

color_range = np.array([[lower_red, upper_red], [lower_blue, upper_blue], [lower_reflective_green, upper_reflective_green]])

@profiler.profile
def main():
    FPS = 0
    while 1:
        t = time.time()
        ret, frame = cap.read()

        #coords_red, img, maskedFrame_red = FindColor(frame, 0, color_range, ignored_size=1000)
        #coords_blue, img, maskedFrame_blue = FindColor(frame, 1, color_range, ignored_size=1000)
        coords_green, img, maskedFrame_green = FindColor(frame, 2, color_range, ignored_size=500)

        FPS = (FPS * 0.99) + ((1 / (time.time() - t)) * 0.0)
        cv2.putText(img, f"FPS: {int(FPS)}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        #cv2.imshow('mask', maskedFrame_red+maskedFrame_blue+maskedFrame_green)
        cv2.imshow('mask', maskedFrame_green)

        cv2.imshow('image', img)

        if(cv2.waitKey(1) & 0xFF == ord('q')):
            break


    cv2.destroyAllWindows()
    cap.release()

if __name__ == '__main__':
    main()