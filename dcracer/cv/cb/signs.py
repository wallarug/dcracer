'''
# Sign detection
'''

import cv2
import numpy as np
import platform

import time
import sys

from rmracerlib import config as cfg
from rmracerlib.cv.func import sign_direction, direction_check, valid_range

## FILTERS
# red filter for stop sign
lower_red_stop = np.array([160, 75, 50])   #red filter
upper_red_stop = np.array([180, 255, 255])   #red filter

lower_red2_stop = np.array([0, 150, 100])   #red filter
upper_red2_stop = np.array([10, 255, 255])   #red filter

# green filter for park sign
lower_green_park = np.array([60, 55, 55])   #green filter
upper_green_park = np.array([90, 130, 150])   #green filter

# blue filter for turn arrow
lower_blue_sign = np.array([90, 80, 50])    #blue filter good
upper_blue_sign = np.array([140, 255, 255])   #blue filter good

lower_white_sign = np.array([0, 0, 150])     #hsv filter good
upper_white_sign = np.array([255, 55, 255])   #hsv filter good

lower_blue_sign_bgr =  np.array([50,0,0])
upper_blue_sign_bgr =  np.array([255,15,15])


# set kernel for operations
kernel = np.ones((cfg.KERNEL_SIZE,cfg.KERNEL_SIZE), np.uint8)
stop_kernel = np.ones((cfg.STOP_KERNEL_SIZE,cfg.STOP_KERNEL_SIZE), np.uint8)

###
###
###  SIGN DETECTIONS
###
###

def detect_stop(frame, hsv):
    """
     Expects: HSV image of any shape + current frame
     Returns: TBD
    """
    # convert to HSV colour space
    #hsv = cv2.cvtColor(frame, cfg.COLOUR_CONVERT) # convert to HSV CS 
    
    # filter
    mask_red = cv2.inRange(hsv, lower_red_stop, upper_red_stop)
    mask_red2 = cv2.inRange(hsv, lower_red2_stop, upper_red2_stop)
    mask = mask_red + mask_red2

    # operations
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, stop_kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, stop_kernel, iterations=1)
    mask = cv2.dilate(mask, stop_kernel, iterations=3)
    
    img = cv2.bitwise_and(frame, frame, mask = mask)

    # logic
    # contours detection
    height, width = mask.shape[:2] 
    contours, _ = cv2.findContours(mask[0:int(height/2), 0:width], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # find the largest contour first that fits all our conditions.
    largest_area = -1
    rect = None
    for cnt in contours:
        # calculate area
        area = cv2.contourArea(cnt)

        # determine if inside the valid range
        x,y,w,h = cv2.boundingRect(cnt)
        vr = valid_range(x,y,w,h,frame)
        if not vr:
            continue
        
        # determine if largest object
        if area > largest_area and area > cfg.AREA_SIZE_STOP:
            largest_area = area
            rect = cv2.boundingRect(cnt)

    # we found a rect that fits the conditions
    if largest_area > 0:
        x,y,w,h = rect
        #roi = frame[y:y+h, x:x+w]

        # check if the ROI is in allowed area
        vr = valid_range(x,y,w,h,frame)
        if not vr:
            return None

        # draw on the rectangle
        if cfg.DEMO_MODE:
            cv2.rectangle(frame, (x,y), (x+w, y+h), (255,255,0), 2)
            cv2.putText(frame, "stop sign", (x,y), cfg.FONT, 1, (255,255,0))
        return "stop"
    return None


def detect_turn(frame, hsv):
    """
     Expects: HSV image of any shape + current frame
     Returns: TBD
    """
    #hsv = cv2.cvtColor(frame, cfg.COLOUR_CONVERT) # convert to HSV CS

    # filter
    #mask = cv2.inRange(hsv, lower_blue_sign, upper_blue_sign)
    maskfilter = cv2.inRange(frame, lower_blue_sign_bgr, upper_blue_sign_bgr)
    maskfilter_white = cv2.inRange(hsv, lower_white_sign, upper_white_sign)

    mask = maskfilter + maskfilter_white

    # operations
    mask = cv2.dilate(maskfilter, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    img = cv2.bitwise_and(frame,frame,mask = mask)
    
    # logic
    # contours detection
    height, width = mask.shape[:2]
    contours, _ = cv2.findContours(mask[0:int(height/2), 0:width], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    result = None
    
    for cnt in contours:
        area = cv2.contourArea(cnt)

        # check the size
        if cfg.AREA_SIZE_TURN < area < cfg.MAX_AREA_SIZE:
            # select a region of interest (ROI)
            x,y,w,h = cv2.boundingRect(cnt)
            roi = frame[y:y+h, x:x+w]

            # check that the ROI falls within an area we want to check
            vr = valid_range(x,y,w,h,frame)
            if not vr:
                continue
            
            result = None
            result = sign_direction(roi)
            if result is not None:
                if cfg.DEMO_MODE:
                    cv2.rectangle(frame, (x,y), (x+w, y+h), (255,127,0), 2)
                    cv2.putText(frame, result, (x+w, y+h), cfg.FONT, 1, (255,127,0))
                return result
    return result
    
    

def detect_park(frame, hsv):
    """
     Expects: HSV image of any shape + current frame
     Returns: TBD
    """
    #hsv = cv2.cvtColor(frame, cfg.COLOUR_CONVERT) # convert to HSV CS

    # filter
    mask = cv2.inRange(hsv, lower_green_park, upper_green_park)

    # operations
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel,iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel,iterations=1)

    img = cv2.bitwise_and(frame,frame,mask = mask)

    # logic
    height, width = mask.shape[:2]
    contours, _ = cv2.findContours(mask[0:int(height/2), 0:width], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        area = cv2.contourArea(cnt)     # calculate area of the contour
        x,y,w,h = cv2.boundingRect(cnt) # create a rectangle around the contour
        #roi = frame[y:y+h, x:x+w]       # select an ROI out of the frame

        # check if the ROI is in allowed area
        vr = valid_range(x,y,w,h,frame)
        if not vr:
            continue

        # calculate ratio of sides - anything not square is not worth checking
        sr = is_squarish(h, w)
        if not sr:
            continue

        # check the area size (too small ignore, too big ignore)
        if cfg.AREA_SIZE_PARK < area < cfg.MAX_AREA_SIZE: #and ( w / h < 1.0):
            if cfg.DEMO_MODE:
                cv2.rectangle(frame, (x,y), (x+w, y+h), (127,255,127), 2)
                cv2.putText(frame, "PARK", (x,y), cfg.FONT, 2, (127,255,127))
            return "park"
    return None
