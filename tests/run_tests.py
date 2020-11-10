'''
# Sign detection - Red Stop Sign

This works by OpenCV first detecting selecting a set area of the image (HSV).
 It then goes to detect a Region of Interest (ROI) through colour filtering.
 Post ROI analysis is completed and a result is then returned.

 1.  Read in an image (or capture)
 2.  Convert to Hue, Saturation, Value (HSV)
 3.  Extract features based on colour
 4.  Further feature extraction of ROI from step 3 for shape, location
      in image, size, etc.
 5.  Result returned.
'''

import cv2
import numpy as np
import platform
#import matplotlib.pyplot as plt

from camera import jetsoncam
import time
import sys

##
## VARIABLES
##
# Turn on the trackbar feature for extracting colours from images
TRACKBARS = False

# Turn on OpenCV Output
DEBUG = True
DEMO_MODE = True

# Select what data to use for analysis.
#  None - video camera
#  .mp4 - video file (one play)
#  .jpg - single image (on loop)
#  folder name - run through a tub
VIDEO_FILE =  "tub2007/138_cam-image_array_.jpg"
#VIDEO_FILE = "track-data/data6.mp4"
VIDEO_FILE = "track-data/data6/468_cam-image_array_.jpg"

# Use this option when providing 'folder name'
TUB = False

# Video frame rate.  33 = ~30 fps, 50 = ~20fps
WAIT_TIME = 50

# Directory when capturing data for analysis
OUTPUT_DIR = "capture"

#  Set the minimum and maximum areas for detection
AREA_SIZE = 30
MAX_AREA_SIZE = 1000

# set the size of the detection kernel for image morph operations.
KERNEL_SIZE = 5

## FILTERS
# red filter for stop sign
lower_red_stop = np.array([160, 75, 50])   #red filter
upper_red_stop = np.array([180, 255, 255])   #red filter

lower_red2_stop = np.array([0, 150, 100])   #red filter
upper_red2_stop = np.array([10, 255, 255])   #red filter

lower_white_stop = np.array([0, 0, 240])  #white octagon
upper_stop = np.array([180, 240, 255])

#kernels
kernel = np.ones((KERNEL_SIZE,KERNEL_SIZE), np.uint8)


## TESTING WINDOWS AND DEBUGGING
# set up the windows for testing
if DEBUG:
    cv2.namedWindow("live")
    cv2.namedWindow("processed")
    cv2.namedWindow("mask")
    cv2.namedWindow("color filter")

if TRACKBARS:
    def nothing(x):
        # any operation
        pass
    cv2.namedWindow("trackbars")
    cv2.namedWindow("trackbarview")
    cv2.createTrackbar("L-H", "trackbars", 0, 180, nothing)
    cv2.createTrackbar("L-S", "trackbars", 0, 255, nothing)
    cv2.createTrackbar("L-V", "trackbars", 0, 255, nothing)
    cv2.createTrackbar("U-H", "trackbars", 180, 180, nothing)
    cv2.createTrackbar("U-S", "trackbars", 255, 255, nothing)
    cv2.createTrackbar("U-V", "trackbars", 255, 255, nothing)

if DEBUG or DEMO_MODE:
    cv2.namedWindow("live")

# set some variables for testing output
font = cv2.FONT_HERSHEY_SIMPLEX
    

## Set the camera for the OS
cap = None
os = platform.system()
if VIDEO_FILE is None:
    if os == 'Linux':  # Jetson
        cap = cv2.VideoCapture(jetsoncam(flip_method=2), cv2.CAP_GSTREAMER)
    elif os == 'Windows':
        cap = cv2.VideoCapture(0)
    elif os == 'Darwin':
        cap = cv2.VideoCapture("/dev/video1")
else:
    cap = cv2.VideoCapture(VIDEO_FILE)


### FUNCTIONS ###
def valid_range(x, y, w, h, frame):
    '''
    This function returns if an roi is in a valid or acceptable part of the image.  The reason
     for having this is due to extra parts of the frame containing reflections.
    '''
    left_buf = 10
    right_buf = 40
    top_buf = 10
    centre_buf = 25

    height, width = frame.shape[:2]

    h0 = top_buf
    h1 = int(height / 5)
    h2 = int(height / 2)
    #horizon = int(((height / 2) + (height / 3))/2)
    horizon = int(height/2)

    v0 = left_buf       # furthest left width
    v1 = int(width/3)   # 1/3rd width
    v2 = v1*2           # 2/3rd width
    v3 = width - right_buf   # furthest right width

    #cv2.line(frame, (0, int(height/3) ), (width, int(height/3) ), (0,0,255))
    cv2.line(frame, (0, h1 ), (width, h1 ), (255,0,255))
    cv2.line(frame, (0, horizon ), (width, horizon ), (0,255,255))

    cv2.line(img, (0, h1 ), (width, h1 ), (255,0,255))
    #cv2.line(img, (0, int(height/2) ), (width, int(height/2) ), (0,0,255))
    cv2.line(img, (0, horizon ), (width, horizon ), (0,255,255))

    cw = True
    ch = False

    if ( (v0 < x < v1) or (v2 < x < v3) ) and ( (v0 < x+w < v1) or (v2 < x+w < v3) ):
        cw = True

    if (h1 < y < horizon) and (h1 < y+h < horizon): #h0 < y < h2:
        ch = True

    if ch and cw:
        return True
    else:
        return False


###  B E G I N  P R O G R A M  ###
roi_num = 0
im_num = 1
frame_num = 0
count = 0

while True:
    # grab a frame from the camera
    if not TUB:
        ret, frame = cap.read()
        frame_num += 1
        if ret == False:
            frame = cv2.imread(VIDEO_FILE)
            ret = True

    # grab a frame from file
    if TUB:
        ret = True
        frame = cv2.imread("{1}/{0}_cam-image_array_.jpg".format(im_num, VIDEO_FILE))
        im_num += 1

    # if frame does not exist... exit
    if not ret:
        print("ERROR - failed to grab frame")
        break

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

# EXAMPLES
mask = cv2.inRange(hsv, lower_red, upper_red)
kernel = np.ones((5, 5), np.uint8)
mask = cv2.erode(mask, kernel)

contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

area = cv2.contourArea(cnt)
approx = cv2.approxPolyDP(cnt, 0.02*cv2.arcLength(cnt, True), True)
x = approx.ravel()[0]
y = approx.ravel()[1]

if area > 400:
    cv2.drawContours(frame, [approx], 0, (0, 0, 0), 5)
    if len(approx) == 3:
        cv2.putText(frame, "Triangle", (x, y), font, 1, (0, 0, 0))
    elif len(approx) == 4:
        cv2.putText(frame, "Rectangle", (x, y), font, 1, (0, 0, 0))
    elif len(approx) == 8:
        cv2.putText(frame, "Octagon", (x, y), font, 1, (0, 0, 0))
    elif 10 < len(approx) < 20:
        cv2.putText(frame, "Circle", (x, y), font, 1, (0, 0, 0))
