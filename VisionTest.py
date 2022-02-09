import cv2
import numpy as np

def FindColor(img, lower, upper, ignored_size=500, filter_length=150):
    mask = cv2.inRange(img, lower, upper)
    mask_contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    coords = []
    if len(mask_contours) != 0:
        for mask_contour in mask_contours:
            if cv2.contourArea(mask_contour) > ignored_size:
                x, y, w, h = cv2.boundingRect(mask_contour)
                if w <= filter_length and h <= filter_length:
                    coords.append((w//2 + x, h//2 + y, w*h))
                    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 0), 3)
            
    return coords, img, mask

if __name__ == '__main__':
    cap = cv2.VideoCapture(cv2.CAP_V4L2)
    while True:
        ret, frame = cap.read()
        width = 600
        height = 600
        
        cv2.imshow('frame', ret)
        
        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    
    
