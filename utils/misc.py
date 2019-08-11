import numpy as np
from .imgproc import normalization
import json
from time import time
from datetime import datetime
import cv2

def printInPlace(text):
    print('\r'+text+'\t'*5,end='',sep = '')

def get_time_left(startTime,currStep,totalSteps):
    ##### assumed first step is step = 0
    elaspsedTime = time() - startTime
    estTimePerStep = elaspsedTime / (currStep + 1)
    remainingSteps = totalSteps - currStep
    leftTime = estTimePerStep * remainingSteps #### in seconds
    M,S = divmod(leftTime,60)
    H,M = divmod(M,60)
    return "%d:%02d:%02d" % (H, M, S) 

def log(logfile,string,printStr = True):
    logfile.write(string+'\n')
    logfile.flush()
    if printStr:
        print(string)
    
def get_current_time():
    return datetime.now().strftime('%Y-%m-%d_%H:%M:%S')

def load_json(jsonPath):
    with open(jsonPath,'r') as fp:
        j_dict = json.load(fp)
    return j_dict

def save_json(j_dict,outPath):
    with open(outPath,'w') as fp:
        json.dump(j_dict,fp,indent=4)

def derange(array,maxTry=1000):
    # shuffles a iterable ensuring that none of the elements
    # remains at its original position
    c = 0
    while True:
        c += 1
        d_array = np.random.permutation(array)
        if all(array != d_array):
            break
        elif c > maxTry:
                print('Maximum number of dearangement attempts reached ('+str(maxTry)+'). Aborting...')
                break

    return d_array

def imshow(img,winName = 'image'):
    disp = normalization(img,min_val=0,max_val=255).astype(np.uint8)
    cv2.imshow('winName',disp)
    cv2.waitKey()
    cv2.destroyWindow(winName)