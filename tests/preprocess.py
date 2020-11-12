#
# Script to rename and create data for TensorFlow
#

from os import listdir, rename, listdir
from os.path import isfile, join
from pathlib import Path

import cv2

PATH = "/Users/cianb/Documents/repos/wallarug/tensorflow-training-ground/turn_signs_color_filtered/train/left"
FILE_NAME = "left"

"""
    Pre-Process Proceedure
    1.  Rename files to correct format
    2.  Generate Extra Data:
        i.   Flip Horizontal  (remember, left is right, right is left ;) )
        ii.  Shift up/down/left/right 10%-15% (random)
        iii. 
    3.  Split into train and validation (75/25)

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
    for count, f in enumerate(files):
        # get extension
        ext = f[-3:]
        
        # set up new file name
        dst = "{1}_{0}.{2}".format(name, count+index, ext)

        # set up old file name
        src = "{0}".format(filename)

        if src == dst:
            print("file already exists")
            continue

        # rename files
        rename(src, dst)
        tracker += 1

    return tracker

def frame_list_rename(detection, start, first, last, path):
    frames = []

    # get all the files from the given path (assume it is set correctly)
    onlyfiles = [f for f in listdir(path) if isfile(join(path, f))]

    # rebuild the full path 
    for filename in onlyfiles:
        frames.append(os.path.join(path, filename))

    # put all the frames in order
    frames.sort()

    index = 0

    # non-detections at the start
    index = file_rename(frames[start:first], (detection + '_false'), index)
    
    # first detection, and detection range
    index = file_rename(frames[first:last], (detection + '_true'), index)
    # after the sign has been pasted to the end of the array (fill out rest)
    index = file_rename(frames[last:], (detection + '_false', index)
    
    
        
#folder_rename(Path(PATH),FILE_NAME)

