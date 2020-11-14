#
# Script to rename and create data for TensorFlow
#  Can take in a whole folder and CSV to mass process
#

import os
from os import listdir, rename, listdir
from os.path import isfile, join
from pathlib import Path
import sys

import cv2

PATH = "/Users/cianb/Documents/repos/wallarug/dcracer/tests/data/shenzhen-stop-1"
FILE_NAME = "stop"
LIMIT = 100

"""
    Pre-Process Proceedure
    1.  Set up the path for the test you wish to change in PATH
    2.  Set up the sign that you are detecting (or result returned by NN) in FILE_NAME
    3.  Set how many images you are starting with

    When using the function frame_list_rename
    1. Path
    2. detection object
    3. First Frame to start test from
    4. First detection in data
    5. Last detection in data

"""


#train_dir = os.path.join(PATH, 'train')
#validation_dir = os.path.join(PATH, 'validation')


def folder_rename(path, name):
    # renames all files from 0 to x with format 'name{0}.jpg'
    for count, filename in enumerate(listdir(path)):
        # get extension
        ext = filename[-3:]
        
        # set up new file name
        dst = "{0}/{1}{2}.{3}".format(path, name, count, ext)

        # set up old file name
        src = "{0}/{1}".format(path, filename)

        if src == dst:
            print("file already exists")
            continue

        # rename files
        rename(src, dst)
        #print(src, dst)

def file_rename(files, name, index):
    """
        Takes a list of full path file names.
    """
    tracker = index
    print(tracker)
    for count, f in enumerate(files):
        # get extension
        ext = f[-3:]

        # get path
        path = '\\'.join(f.split('\\')[:-1])
  
        # set up new file name
        dst = "{0}\{1}_{2}.{3}".format(path, count+index, name, ext)

        # set up old file name
        src = "{0}".format(f)

        if src == dst:
            print("file already exists")
            continue

        # rename files
        rename(src, dst)
        tracker += 1

        if tracker > LIMIT:
            break

        print(src, dst)

    return tracker

def frame_list_rename(path, detection, start, first, last):
    frames = []

    # get all the files from the given path (assume it is set correctly)
    onlyfiles = [f for f in listdir(path) if isfile(join(path, f))]

    # rebuild the full path 
    for filename in onlyfiles:
        frames.append(os.path.join(path, filename))

    # put all the frames in order
    frames.sort()

    #print(frames)

    index = 0

    # adjust for case where we do not have indexed images from 0
    ch1 = frames[0].split('_')[0].split('\\')[-1]

    offset = int(frames[0].split('_')[0].split('\\')[-1])
    print(offset)

    real_start = start - offset
    real_first = first - offset
    real_last = last - offset

    print(real_start, real_first, real_last)

    # non-detections at the start
    print(frames[real_start:real_first])
    index = file_rename(frames[real_start:real_first], (detection + '_false'), index)
    
    # first detection, and detection range
    index = file_rename(frames[real_first:real_last], (detection + '_true'), index)
    # after the sign has been pasted to the end of the array (fill out rest)
    index = file_rename(frames[real_last:], (detection + '_false'), index)
    

#TODO - read in a CSV file with all the parameters
def read_csv():
    pass

        
#folder_rename(Path(PATH),FILE_NAME)
frame_list_rename(Path(PATH), FILE_NAME, 200, 269, 289) 
