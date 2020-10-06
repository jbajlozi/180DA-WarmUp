import cv2
import numpy as np

cap = cv2.VideoCapture(0)

while(1):

    # Take each frame
    ret, frame = cap.read()

    #apply median filter (smoothes out)
    frame = cv2.medianBlur(frame,5)

    if not ret:
        break

    # Convert BGR to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # define range of blue color in HSV
    lower_green = np.array([50,40,0])
    upper_green = np.array([70,255,255])

    # Threshold the HSV image to get only green colors
    mask = cv2.inRange(hsv, lower_green, upper_green)

    # Bitwise-AND mask and original image
    res = cv2.bitwise_and(frame,frame,mask=mask)
    
    #find contours (same color)
    contours,_ = cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    for _,contour in enumerate(contours): 
        area = cv2.contourArea(contour) 

        #area of 2500 to filter out any noise (my object was big enough)
        if(area > 2500): 
            rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            frame = cv2.drawContours(frame,[box],0,(0,255,0),2)

    cv2.imshow('frame',frame)
    #cv2.imshow('mask',mask)
    #cv2.imshow('res',res)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()