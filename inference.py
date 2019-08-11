import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  ###### disable tensorflow warnings logging
import tensorflow as tf
import numpy as np
from network import UNet
import cv2
from utils.imgproc import normalization
from pynput.mouse import Button, Controller
from time import time
import tkinter

def get_screen_size():
    root = tkinter.Tk()
    width = root.winfo_screenwidth()
    height = root.winfo_screenheight()
    root.quit()
    return width,height

def relative_to_absolute_coordinates(x,y,target_shape):
    r,c = target_shape[0:2]
    ax = int(x*c)
    ay = int(y*r)
    return ax,ay

def absolute_to_relative_coordinates(x,y,input_shape):
    r,c = input_shape[0:2]
    rx = float(x) / c
    ry = float(y) / r
    return rx,ry

def get_pred_eyes_centers(pred_heatmap,prob_thr = 0.5):
    pred_heatmap = np.squeeze(pred_heatmap)
    segmentation = (pred_heatmap > prob_thr).astype(np.uint8)

    lx = -1
    ly = -1
    rx = -1
    ry = -1

    # Detect blobs.
    contoursLeft,_ = cv2.findContours(segmentation[:,:,0],cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    contoursRight,_ = cv2.findContours(segmentation[:,:,1],cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

    if len(contoursLeft) > 0:
        cL = sorted(contoursLeft, key = cv2.contourArea, reverse = True)[0]
        mL = cv2.moments(cL) # calculate x,y coordinate of centers
        if mL["m00"] != 0:
            lx = int(mL["m10"] / mL["m00"]) 
            ly = int(mL["m01"] / mL["m00"])

    if len(contoursRight) > 0:
        cR = sorted(contoursRight, key = cv2.contourArea, reverse = True)[0]
        mR = cv2.moments(cR) # calculate x,y coordinate of centers
        if mR["m00"] != 0:
            rx = int(mR["m10"] / mR["m00"]) 
            ry = int(mR["m01"] / mR["m00"])
    
    return lx,ly,rx,ry,

def predict_eyes_coordinates(frame,net_input_shape):
    
    img = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY) 
    img = cv2.resize(img,(net_input_shape[1],net_input_shape[0]))
    img = (img - np.mean(img)) / np.std(img)
    img = img[np.newaxis, ..., np.newaxis]
    pred_heatmap = model.predict_on_batch(img)

    eyeState = dict()
    lx,ly,rx,ry = get_pred_eyes_centers(pred_heatmap,prob_thr=prob_thr)
    if lx >= 0 and ly >= 0:
        lx,ly = absolute_to_relative_coordinates(lx,ly,net_input_shape)
        lx,ly = relative_to_absolute_coordinates(lx,ly,frame.shape)
        eyeState['left_open'] = True
        eyeState['left_pos'] = (lx,ly)
    else:
        eyeState['left_open'] = False
        eyeState['left_pos'] = None
        
    if rx >= 0 and ry>= 0:
        rx,ry = absolute_to_relative_coordinates(rx,ry,net_input_shape)
        rx,ry = relative_to_absolute_coordinates(rx,ry,frame.shape)
        eyeState['right_open'] = True
        eyeState['right_pos'] = (rx,ry)
    else:
        eyeState['right_open'] = False
        eyeState['right_pos'] = None
    
    return lx,ly,rx,ry,eyeState

def run_eyestate_actions(eyeState,mouse):
    if eyeState['right_open'] and not eyeState['left_open']: #####left eye closed, right eye open
        mouse.move(-10, 0)
    elif eyeState['left_open'] and not eyeState['right_open']: #####left eye closed, right eye open
        mouse.move(10, 0)
    # elif not eyeState['left_open'] and not eyeState['right_open']: ### both eyes closed
    #     mouse.position = (1024,540) ###move mouse to the center of the screen


if __name__ == '__main__':

    cwd = os.path.dirname(os.path.realpath(__file__))
    vcap = cv2.VideoCapture(0)
    vc = int(vcap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vr = int(vcap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    input_shape = (400,400,1)
    print(input_shape)
    outDir = cwd + '/models/exp_2'
    bestModelPath = os.path.join(outDir,'bestModel.h5')

    model = tf.keras.models.load_model(bestModelPath)
    model_input = tf.keras.layers.Input(input_shape)
    model_output = model(model_input)
    model = tf.keras.Model(model_input,model_output)

    prob_thr = 0.5

    w,h = get_screen_size()

    mouse = Controller()
    while 1:
        ret, frame = vcap.read()
        lx,ly,rx,ry,eyeState = predict_eyes_coordinates(frame,net_input_shape = input_shape)
        
        disp = frame.copy()
        disp = cv2.circle(disp,center=(lx,ly),radius = 2,color = (0,0,255),thickness=-1)
        disp = cv2.circle(disp,center=(rx,ry),radius = 2,color = (0,255,0),thickness=-1)
        print('\nLeft: %d %d -- right: %d %d' % (lx,ly,rx,ry))
        print(eyeState)

        # run_eyestate_actions(eyeState,mouse)

        cv2.imshow('Eye detection',np.fliplr(disp))
        cv2.waitKey(10)
        
    

          