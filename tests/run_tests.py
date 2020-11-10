'''
# DC Racer Testing and Reporting System (DC RTRS)

TODO - Description and Usage

'''

import cv2
import numpy as np
import platform
#import matplotlib.pyplot as plt

import time
import sys

from os import listdir
from os.path import isfile, join

##
## VARIABLES
##

# Turn on OpenCV Output
DEBUG = True
DEMO_MODE = True

# Video frame rate.  33 = ~30 fps, 50 = ~20fps
WAIT_TIME = 50

# Directory when capturing data for analysis
OUTPUT_DIR = "capture"




## TESTING WINDOWS AND DEBUGGING
# set up the windows for testing
if DEBUG:
    cv2.namedWindow("live")

# set some variables for testing output
font = cv2.FONT_HERSHEY_SIMPLEX



###  B E G I N  P R O G R A M  ###
roi_num = 0
im_num = 1
frame_num = 0
count = 0

## setup
dir_path = os.path.dirname(os.path.realpath(__file__))
dir_data = os.path.join(dir_path, 'data')

## result variables
results = {}

## import library / code for testing.
#from scripts.detect import detect

# load test cases
testfile = open('tests.csv', 'r')
header = True

for testcase in testfile:
    if header:
        header = False
        continue
    # for each test, run the detect function and store the result(s)
    folder = os.path.join(dir_data, testcase[1])
    onlyfiles = [f for f in listdir(folder) if isfile(join(folder, f))]

    results[testcase] = {'correct' : 0, 'incorrect' : 0,
                         'true_pos' : 0, 'false_pos' : 0,
                         'true_neg' : 0, 'false_neg' : 0,
                         'percentage' : 0, 'total_img' : len(onlyfiles)}    

    for f in onlyfiles:
        name = f
        img = os.path.join(folder, f)
        frame = cv2.imread(img)

        # process frame
        res = detect(frame)

        # process results
        # TODO - strip file name and get the detection value from the file name
        # TODO - compare with result from res

        # save results
        

while True:
    # grab a frame from file
    
    frame = cv2.imread("{1}/{0}_cam-image_array_.jpg".format(im_num, VIDEO_FILE))
    im_num += 1

    # convert to HSV image
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # trackbars
    if TRACKBARS:
        l_h = cv2.getTrackbarPos("L-H", "trackbars")
        l_s = cv2.getTrackbarPos("L-S", "trackbars")
        l_v = cv2.getTrackbarPos("L-V", "trackbars")
        u_h = cv2.getTrackbarPos("U-H", "trackbars")
        u_s = cv2.getTrackbarPos("U-S", "trackbars")
        u_v = cv2.getTrackbarPos("U-V", "trackbars")

        lower_trackbars = np.array([l_h, l_s, l_v])
        upper_trackbars = np.array([u_s, u_h, u_v])
    
        mask_tbars = cv2.inRange(hsv, lower_trackbars, upper_trackbars)
        cv2.imshow("trackbarview", mask_tbars)
    
    #mask = cv2.blur(mask, (5,5))
    mask_red = cv2.inRange(hsv, lower_red_stop, upper_red_stop)
    mask_red2 = cv2.inRange(hsv, lower_red2_stop, upper_red2_stop)
    mask_white = cv2.inRange(hsv, lower_white_stop, upper_stop)

    maskfilter = mask_red + mask_red2
    #mask = mask + mask_white

    ##mask = cv2.GaussianBlur(mask, (3,3), 0)
    #mask = cv2.erode(mask, kernel)
    #mask = cv2.dilate(, kernel, iterations=1)
    mask = cv2.morphologyEx(maskfilter, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=2)
    
    ##mask = cv2.erode(mask, kernel, iterations=1)
    #mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    #mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    #mask = cv2.erode(mask, kernel)

    img = cv2.bitwise_and(frame,frame,mask = mask)

    # find shapes
    # contours detection
    height, width = mask.shape[:2] 
    contours, _ = cv2.findContours(mask[0:int(height/2), 0:width], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # find the largest contour first that fits all our conditions.
    largest_area = -1
    rect = None
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        #approx = cv2.approxPolyDP(cnt, 0.01 * cv2.arcLength(cnt, True), True)
        x,y,w,h = cv2.boundingRect(cnt)
        vr = valid_range(x,y,w,h,frame)
        if not vr:
            continue

        if area > largest_area and area > AREA_SIZE:
            largest_area = area
            rect = cv2.boundingRect(cnt)

    # we found a rect that fits the conditions
    if largest_area > 0:
        # capture a ROI image and store for later
        x,y,w,h = rect
        roi = frame[y:y+h, x:x+w]

        # check if valid range
        vr = valid_range(x,y,w,h,frame)
        if not vr:
            continue
        
        #cv2.imwrite("{2}/{1}_roi-{0}.jpg".format(roi_num, frame_num, OUTPUT_DIR), roi)
        count += 1
        #cv2.drawContours(frame, [approx], 0, (0,0,0), 2)
        #print(len(cnt), " - ", count)
        if len(cnt) == 8:
            print("octagon!!", count)
            #x = approx.ravel()[0]
            #y = approx.ravel()[1]
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0,0,0), 2)
        cv2.putText(frame, "STOP", (x,y), font, 1, (0,0,0))

    if DEBUG:
        cv2.imshow("processed", img)
        cv2.imshow("mask", mask)
        cv2.imshow("color filter", maskfilter)
    if DEMO_MODE or DEBUG:
        cv2.imshow("live", frame)
    
    key = cv2.waitKey(WAIT_TIME)
    if key == ord('q'):
        break
    continue


    

cap.release()
cv2.destroyAllWindows()
sys.exit(0)
