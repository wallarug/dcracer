#!/usr/bin/env python3
"""
Scripts for operating the OpenCV by Robotics Masters
 with the Donkeycar
author: @wallarug (Cian Byrne) 2020
contrib: @peterpanstechland 2020

Note: To be used with code.py bundled in this repo. See donkeycar/contrib/robohat/code.py
"""

import time
import donkeycar as dk
import cv2

from rmracerlib.cv.lights import detect_traffic
from rmracerlib.cv.signs import detect_stop, detect_turn, detect_park
from rmracerlib.datastructure.Queue10 import Queue10
from rmracerlib import config as rmcfg

class RMRacerCV:
    '''
    Documentation to be added later
    
    '''
    def __init__(self, cfg, debug=False):
        self.throttle = 0
        self.steering = 0
        self.debug = debug
        #self.show_bounding_box = cfg.show_bounding_box

        # actions for the car & error detection
        self.stop_sign = Queue10(rmcfg.DK_COUNTER_SIZE)
        self.right_sign = Queue10(rmcfg.DK_COUNTER_SIZE)
        self.left_sign = Queue10(rmcfg.DK_COUNTER_SIZE)
        self.park_sign = Queue10(rmcfg.DK_COUNTER_SIZE)
        self.red_traf = Queue10(rmcfg.DK_COUNTER_SIZE)
        self.grn_traf = Queue10(rmcfg.DK_COUNTER_SIZE)

        # if we are already running a sequence (like parking) do
        #  not do anything except finish that sequence
        self.running = False
        self.action = None
        self.wait = 0

    def update(self):
        pass

    def run_threaded(self, img_arr, throttle, steering, debug=False):
        return self.run(img_arr, throttle, steering)

    def shutdown(self):
        try:
            pass
        except:
            pass

    def run(self, img_arr, throttle, steering, debug=False):
        if img_arr is None:
            return img_arr, throttle, steering

        if self.running == False:
            ## pre-process the image to save time...
            hsv = cv2.cvtColor(img_arr, rmcfg.COLOUR_CONVERT) # convert to HSV CS 

            ##
            ##  DETECTION CODE
            ##
            # Stop Sign (Detect)
            stop = detect_stop(img_arr, hsv)
            if stop == "stop":
                # add one to counter, check if above threshold
                if rmcfg.DK_SHOW_TEXT_DEBUG: print("detection -stop")
                if self.stop_sign.put(1) > rmcfg.DK_COUNTER_THRESHOLD:
                    self.running = True
                    if rmcfg.DK_SHOW_TEXT_DEBUG: print("multiple!")
                    self.action = "stopping"
                    return img_arr, throttle, steering
            else:
                self.stop_sign.put()

            # Turn Sign (Detect)
            turn = None #detect_turn(img_arr, hsv)
            if turn == "right":
                if self.right_sign.put(1) > rmcfg.DK_COUNTER_THRESHOLD:
                    self.running = True
                    self.action = "right"
                    if rmcfg.DK_SHOW_TEXT_DEBUG: print("detection -right")
                    return img_arr, throttle, steering

            elif turn == "left":
                if self.left_sign.put(1) > rmcfg.DK_COUNTER_THRESHOLD:
                    self.running = True
                    self.action = "left"
                    if rmcfg.DK_SHOW_TEXT_DEBUG: print("detection -left")
                    return img_arr, throttle, steering
            else:
                self.right_sign.put()
                self.left_sign.put()

            # Traffic Light (Detect)
            traffic_light = None#detect_traffic(img_arr, hsv)
            if traffic_light == "stop":
                if self.red_traf.put(1) > rmcfg.DK_COUNTER_THRESHOLD:
                    self.running = True
                    self.action = "lstopping"
                    return img_arr, throttle, steering
            elif traffic_light == "go":
                self.grn_traf.put(1)
            else:
                self.red_traf.put()
                self.grn_traf.put()

            # Park Sign (Detect)
            park = None#detect_park(img_arr, hsv)
            if park == "park":
                if self.park_sign.put(1) > rmcfg.DK_COUNTER_THRESHOLD:
                    self.running = True
                    self.action = "park"
                    return img_arr, throttle, steering
            else:
                self.park_sign.put()

        ##
        ## Action Check Status
        ##

        if self.running == True:
            # stop sign & traffic light actions
            if (self.action == "stopping" or self.action == "lstopping") and self.wait <= rmcfg.DK_ACTION_DELAY:  # wait before stopping
                self.wait += 1
                if rmcfg.DK_SHOW_TEXT_DEBUG: print("action - stopping")
                return throttle, steering, img_arr
            elif ((self.action == "stopping" or self.action == "lstopping") and self.wait > rmcfg.DK_ACTION_DELAY): # execute
                if rmcfg.DK_SHOW_TEXT_DEBUG: print("action - stopping now")
                self.wait = 0
                if self.action == "lstopping":
                    self.action = "waiting"
                else:
                    self.action = "stopped"
                return img_arr, 0, steering

            # stop sign only
            elif self.action == "stopped" and self.wait <= rmcfg.DK_ACTION_RUNTIME:  # execute, action is running
                if rmcfg.DK_SHOW_TEXT_DEBUG: print("action - stopped {0}".format(self.wait))
                self.wait += 1
                cv2.putText(img_arr, "stopped", (5,5), rmcfg.FONT, 1, (255,255,0))
                return img_arr, 0, steering
            elif self.action == "stopped" and self.wait > rmcfg.DK_ACTION_RUNTIME:  # complete, leave action
                self.wait = 0
                self.action = None
                self.running = False
                self.stop_sign.clear()
                if rmcfg.DK_SHOW_TEXT_DEBUG: print("complete! control returned!")
                return img_arr, throttle, steering

            # traffic light only
            elif self.action == "waiting": # minimum wait time?: and self.wait < cfg.DK_ACTION_RUNTIME:
                hsv = cv2.cvtColor(frame, rmcfg.COLOUR_CONVERT) # convert to HSV CS
                traffic_light = detect_traffic(img_arr, hsv)

                if traffic_light == "go":
                    self.grn_traf.put(1)
                    if self.grn_traf.total > rmcfg.DK_COUNTER_THRESHOLD:
                        # we can move again
                        self.wait = 0
                        self.action = None
                        self.running = False
                        self.red_traf.clear()
                        return img_arr, throttle, steering
                else:
                    self.grn_traf.put()
                    self.wait += 1
                    return img_arr, 0, steering

            # turn only
            elif (self.action == "left" or self.action == "right") and self.wait <= rmcfg.DK_ACTION_DELAY:
                self.wait += 1
                if rmcfg.DK_SHOW_TEXT_DEBUG: print("action - turning")
                return img_arr, throttle, steering

            elif (self.action == "left" or self.action == "right") and self.wait > rmcfg.DK_ACTION_DELAY:
                if self.action == "left" and self.wait < (rmcfg.DK_ACTION_DELAY + rmcfg.DK_ACTION_RUNTIME):
                    self.wait += 1
                    if rmcfg.DK_SHOW_TEXT_DEBUG: print("action - left")
                    return img_arr, throttle, -1.0
                elif self.action == "right" and self.wait < (rmcfg.DK_ACTION_DELAY + rmcfg.DK_ACTION_RUNTIME):
                    self.wait += 1
                    if rmcfg.DK_SHOW_TEXT_DEBUG: print("action - right")
                    return img_arr, throttle, 1.0
                else:
                    self.wait = 0
                    self.action = None
                    self.running = False
                    self.left_sign.clear()
                    self.right_sign.clear()
                    if rmcfg.DK_SHOW_TEXT_DEBUG: print("complete! control returned!")
                    return img_arr, throttle, steering

            # park only
            elif self.action == "park" and self.wait <= rmcfg.DK_ACTION_DELAY:
                self.wait += 1
                return throttle, steering, img_arr
            elif self.action == "park" and self.wait > rmcfg.DK_ACTION_DELAY:
                self.wait = 0
                self.action = "parking"
                return img_arr, 0, steering
            elif self.action == "parking":
                # action for parking the car (over time)
                ##  To accomplish this, we have to send different values over various ticks
                ##   since using sleep freezes the Donkey Car.
                ##  We shall define ticks as 20 every second.  The whole process will take
                ##   120 ticks (approx) or 6 seconds
                ##
                if self.wait < 40:
                    # first we must steer the car into the parking spot, assuming we know
                    #  which side the park sign is on (TODO)
                    return img_arr, -0.5, 1.0 # lock left, reverse parking?

                elif self.wait < 60:
                    return img_arr, -0.5, 0.0 # straighten up, reverse parking?

                elif self.wait > 100:
                    return img_arr, 0.0, 0.0   # stop the car

                elif self.wait > 200:
                    self.wait = 0
                    self.action = None
                    self.running = False
                    self.park_sign.clear()
                    return img_arr, throttle, steering

            else:
                print("BAD BAD BAD!!!")

