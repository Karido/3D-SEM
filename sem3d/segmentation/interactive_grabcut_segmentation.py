# Copyright (c) 2023, Stefan Toeberg.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. 
#
# (http://opensource.org/licenses/BSD-3-Clause)
#
# __author__ = "Stefan Toeberg, LUH: IMR"
# __description__ = """
#                   interactive grabcut segmentation based on the
#                   openCV GrabCut implementation,
#                   manual segmentation example code was adjusted to use 
#                   the interactive segmentation in the reconstruction routine
#                   and the initially determined mask as
#                   initial segmentation state if desired,
#                   button controls were additionally extended
#                   """
                  
# '''
# ===============================================================================
# Interactive Image Segmentation using GrabCut algorithm.
# Two windows are used which display the input and output.
# Draw a rectangle around the object by using the right mouse button.
# Press 'n' to segment the object (once or a few times)
# For manual adjustments, press any of the keys below and draw mark the parts
# in the image. Afterwards, press 'n' for updating the output.
# Key '0'             - Mark regions of fixed background
# Key '1'             - Mark regions of fixed foreground
# Key '2'             - Mark regions of  probable background
# Key '3'             - Mark regions of  probable foreground
# Key 'n'             - Perform GrabCut iteration
# Key 'r'             - Reset to no inputs and no segmentation
# Key 'i'             - Reset to initially used input mask
# Key 's' or 'Esc'    - break and return the results
# ===============================================================================
# '''

# packages
from __future__ import print_function
import numpy as np
import cv2 as cv
import copy

# modules
import sem3d.segmentation.segmentation as segmentation

# define constants
BLUE = [255,0,0]        # rectangle color      
RED = [0,0,255]         # PR BG
GREEN = [0,255,0]       # PR FG
BLACK = [0,0,0]         # sure BG
WHITE = [255,255,255]   # sure FG

DRAW_BG = {'color' : BLACK, 'val' : 0}
DRAW_FG = {'color' : WHITE, 'val' : 1}
DRAW_PR_FG = {'color' : GREEN, 'val' : 3}
DRAW_PR_BG = {'color' : RED, 'val' : 2}

# setting up flags
rect = (0,0,1,1)
drawing = False         # flag for drawing curves
rectangle = False       # flag for drawing rect
rectOver = False        # flag to check if rect drawn
rectOrMask = 100        # flag for selecting rect or mask mode
value = DRAW_FG         # drawing initialized to FG
thickness = 3           # brush thickness


def onmouse(event,x,y,flags,param):
    global img,img2,drawing,value,mask,rectangle,rect,rectOrMask,ix,iy,rectOver

    # Draw Rectangle
    if event == cv.EVENT_RBUTTONDOWN:
        rectangle = True
        ix,iy = x,y

    elif event == cv.EVENT_MOUSEMOVE:
        if rectangle == True:
            img = img2.copy()
            cv.rectangle(img,(ix,iy),(x,y),BLUE,2)
            rect = (min(ix,x),min(iy,y),abs(ix-x),abs(iy-y))
            rectOrMask = 0

    elif event == cv.EVENT_RBUTTONUP:
        rectangle = False
        rectOver = True
        cv.rectangle(img,(ix,iy),(x,y),BLUE,2)
        rect = (min(ix,x),min(iy,y),abs(ix-x),abs(iy-y))
        rectOrMask = 0
        print(" Press the key 'n' to perform a GrabCut iteration based on the inputs.\n")

    # draw touchup curves

    if event == cv.EVENT_LBUTTONDOWN:
        if rectOver == False and initByMask == False:
            print(" Draw a rectangle with pressed right click. \n")
        else:
            drawing = True
            cv.circle(img,(x,y),thickness,value['color'],-1)
            cv.circle(mask,(x,y),thickness,value['val'],-1)

    elif event == cv.EVENT_MOUSEMOVE:
        if drawing == True:
            cv.circle(img,(x,y),thickness,value['color'],-1)
            cv.circle(mask,(x,y),thickness,value['val'],-1)

    elif event == cv.EVENT_LBUTTONUP:
        if drawing == True:
            drawing = False
            cv.circle(img,(x,y),thickness,value['color'],-1)
            cv.circle(mask,(x,y),thickness,value['val'],-1)


def grab_specimen(image, initMask, initByMaskInput):
    
    global img, img2, value, mask, rectangle, output, rectOrMask, rect, output, ix, iy, rectOver, initByMask
    
    # build image with 3 channels
    img = np.stack((image,)*3, axis=-1)
    img2 = img.copy()
    initByMask = copy.copy(initByMaskInput) 
    
    # input and output windows    
    cv.namedWindow('output', cv.WINDOW_NORMAL)
    cv.resizeWindow('output', 800,800)
    cv.namedWindow('input', cv.WINDOW_NORMAL)
    cv.resizeWindow('input', 800,800)
    cv.setMouseCallback('input',onmouse)
    # cv.moveWindow('input',img.shape[1]+10,90)
               
    if (initByMask == True): 
        mask = initMask.copy()
        img2, bgdmodel, fgdmodel, rect = segmentation.get_input_GC(image)
        # just use the initial mask as first output
        mask2 = np.where( (mask==1) + (mask==3), 255, 0).astype('uint8')
        output = cv.bitwise_and(img2,img2,mask=mask2)  
        rectOrMask = True
    else:
        print(" Instructions:")
        print(" Draw a rectangle with pressed right click including the object to be segmented.")
        # mask initialized to PR_BG
        mask = np.zeros(img.shape[:2],dtype = np.uint8) 
        # output image to be shown
        output = np.zeros(img.shape,np.uint8)
        rectOrMask = False
              
    print(" To refine the result, mark the foreground and background by pressing the keys:\n"
        " 0 - background \n 1 - foreground \n 2 - probable background \n 3 - probable foreground\n"
        " r - reset completly \n i - reset to initial input mask if available \n Esc/s - return output\n") 
    
    while(True):
        cv.imshow('output',output)
        cv.imshow('input',img)
        
        # wait for user input
        k = cv.waitKey(True)
        
        # key bindings
        if k == 27:
            # esc to exit
            print(" Result as output! \n")
            break
        
        elif k == ord('0'): 
            # draw background
            print(" Mark the background by clickling the left mouse button. \n")
            value = DRAW_BG
            
        elif k == ord('1'):
            # draw foreground
            print(" Mark the foreground by clickling the left mouse button. \n")
            value = DRAW_FG
            
        elif k == ord('2'):
            # draw background
            print(" Mark the probable background by clickling the left mouse button. \n")
            value = DRAW_PR_BG
            
        elif k == ord('3'): 
            # draw probable foreground
            print(" Mark the probable foreground by clickling the left mouse button. \n")
            value = DRAW_PR_FG
            
        elif k == ord('s'):
            # return/save image
            # breaks the loop like esc           
            print(" Result as output! \n")
            break;
            
        elif k == ord('r'): 
            # reset to unsegmented image and no inputs
            print("Reset image! \n")
            rect = (0,0,1,1)
            drawing = False
            rectangle = False
            rectOrMask = 100
            rectOver = False
            value = DRAW_FG
            img = img2.copy()
            mask = np.zeros(img.shape[:2],dtype = np.uint8)
            output = np.zeros(img.shape,np.uint8)           
            
        elif k == ord('i'):  
            # reset to initial input mask
            if (initByMask == True):         
                img2, bgdmodel, fgdmodel, rect = segmentation.get_input_GC(image)
                # perform intitial iteration
                gcOutput = cv.grabCut(img2,initMask,rect,bgdmodel,fgdmodel,1,cv.GC_INIT_WITH_MASK)
                mask = copy.deepcopy(gcOutput[0])
                mask2 = np.where((mask==1) + (mask==3),255,0).astype('uint8')
                output = cv.bitwise_and(img2,img2,mask=mask2)  
                rectOrMask = True
                drawing = False
                img = img2.copy()
            else:
                pass
            
        elif k == ord('n'): 
            # perform GrabCut iteration
            
            if rectOrMask == False:         
                # grabcut with rectangle input
                bgdmodel = np.zeros((1,65), np.float64)
                fgdmodel = np.zeros((1,65), np.float64)
                cv.grabCut(img, mask, rect, bgdmodel, fgdmodel, 1, cv.GC_INIT_WITH_RECT)
                rectOrMask = True
                
            elif rectOrMask == True:         
                # grabcut with mask input
                bgdmodel = np.zeros((1,65),np.float64)
                fgdmodel = np.zeros((1,65),np.float64)
                cv.grabCut(img2,mask,rect,bgdmodel,fgdmodel,1,cv.GC_INIT_WITH_MASK)

        mask2 = np.where((mask==1) + (mask==3),255,0).astype('uint8')
        output = cv.bitwise_and(img2,img2,mask=mask2)   

    cv.destroyAllWindows()
    
    return output[:,:,0]    
