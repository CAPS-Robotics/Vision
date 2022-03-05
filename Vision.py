#! /usr/bin/python3

import threading
import cv2
import numpy as np
import time
import profiler
from networktables import NetworkTables as nt

def FindColor(img, color, color_range, ignored_size=500, filter_length=1080*720):
    #img = cv2.GaussianBlur(img, guassian_kernel_size=(11, 11), cv2.BORDER_DEFAULT)
    lower, upper = color_range[color]
    mask = cv2.inRange(img, lower, upper)
    mask_contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    coords = []
    for mask_contour in mask_contours:
        if cv2.contourArea(mask_contour) > ignored_size:
            x, y, w, h = cv2.boundingRect(mask_contour)
            if w <= filter_length and h <= filter_length:
                coords.append((w//2 + x, h//2 + y, w*h))
                #cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 0), 3)

    return coords, img, mask


def CenterOfMass(coords_green):
    L = len(coords_green)
    
    if not L:
        return None
    
    x = np.zeros((L))
    y = np.zeros((L))
    ax = np.zeros((L))
    for l in range(L):
        x[l], y[l], ax[l] = coords_green[l]
    
    ay = ax.copy()
    
    std = np.std(y)
    mean = np.sum(y) / L
    for l in range(L):
        try: 
            if y[l] < mean - (0.1 * std) or y[l] > mean + (0.1 * std):
                y = np.delete(y, l)
                ay = np.delete(ay, l)
                #print("Pop y")
        except IndexError:
            break
                
    std = np.std(x)
    mean = np.sum(x) / L
    for l in range(L):
        try:
            if x[l] < mean - (0.4 * std) or x[l] > mean + (0.4 * std):
                x = np.delete(x, l)
                ax = np.delete(ax, l)
                #print("Pop  x")
        except IndexError:
            break
            
        
    ax = ax/np.sum(ax)
    ay = ay/np.sum(ay)
    x *= ax
    y *= ay
    x = int(np.sum(x))
    y = int(np.sum(y))
    
    return np.array([x, y])

cond = threading.Condition()
notified = [False]
def connectionListener(connected, info):
    print(info, '; Connected=%s' % connected)
    with cond:
        notified[0] = True
        cond.notify()

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

lower_red = np.array([0, 8, 118])
upper_red = np.array([104, 85, 225])

lower_blue = np.array([57, 26, 0])
upper_blue = np.array([117, 119, 40])

lower_reflective_green = np.array([115, 182, 53])
upper_reflective_green = np.array([229, 224, 125])

color_range = np.array([[lower_red, upper_red], [lower_blue, upper_blue], [lower_reflective_green, upper_reflective_green]])

#@profiler.profile
def main():
    try:
        nt.initialize(server='roborio-2410-frc.local') #roborio address
        nt.addConnectionListener(connectionListener, immediateNotify=True)
        
        with cond:
            print("Waiting")
            if not notified[0]:
                cond.wait()
                
        print("Connected!")
        
    except KeyboardInterrupt:
        pass
    
    table = nt.getTable("SmartDashboard")
    
    FPS = 0
    average_center_of_mass = np.array([320, 240])
    
    #constants for distance calculation
    theta = 3
    hoop_height = 1
    camera_height = 1
    k = 2
    while True:
        t = time.time()
        ret, frame = cap.read()

        #coords_red, img, maskedFrame_red = FindColor(frame, 0, color_range, ignored_size=1000)
        #coords_blue, img, maskedFrame_blue = FindColor(frame, 1, color_range, ignored_size=1000)
        coords_green, img, maskedFrame_green = FindColor(frame, 2, color_range, ignored_size=10, filter_length=50)
        
        center_of_mass = CenterOfMass(coords_green)
        if np.sum(center_of_mass):
            average_center_of_mass = (average_center_of_mass * 0.71) + (center_of_mass * 0.29) #Smoothing
            x, y = average_center_of_mass
            
            turn = (x - 540) / 540
            if abs(turn) < 0.04625: #0.185 old
                turn = 0
            
            print(turn)
            table.putNumber('turn', turn)
            
            table.putBoolean('distance', y > k)

            #cv2.circle(img, (int(x), int(y)), radius=7, color=(0, 0, 0), thickness=-1)

        FPS = (FPS * 0.99) + ((1 / (time.time() - t)) * 0.01)
        #cv2.putText(img, f"FPS: {int(FPS)}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        
        #cv2.line(img, (515, 0), (515, 720), (255, 0, 0), 5)
        #cv2.line(img, (565, 0), (565, 720), (255, 0, 0), 5)

        cv2.imshow('mask', maskedFrame_green)
        cv2.imshow('image', img)

        if(cv2.waitKey(1) & 0xFF == ord('q')):
            break


    cv2.destroyAllWindows()
    cap.release()

if __name__ == '__main__':
    main()