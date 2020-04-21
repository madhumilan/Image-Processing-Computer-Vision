'''
Created on Apr 27, 2018

@author: Shreyas Manjunath
'''
import cv2
import numpy as np
import imutils
ratio =1 
from picamera.array import PiRGBArray
from picamera import PiCamera
import time
    
camera = PiCamera()
camera.resolution = (640, 480)
camera.framerate = 32
    
rawCapture = PiRGBArray(camera, size=(640, 480))
time.sleep(0.1)
imageL = cv2.imread("ArrowL.jpg",0)
imageR = cv2.imread("ArrowR.jpeg",0)
image1 = cv2.imread("room1.jpg",0)
image2 = cv2.imread("room2.jpg",0)

camera.capture(rawCapture, format="bgr")
time.sleep(1)
image = rawCapture.array
        
wid, hei, ch = image.shape
ratio = wid / 400.0
orig = image.copy()
image = imutils.resize(image, height = 400)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.bilateralFilter(gray, 11, 17, 17)
edged = cv2.Canny(gray, 30, 200)
cnts = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2]
cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:10]
screenCnt = None
flag =0
for c in cnts:
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.02 * peri, True)

    if len(approx) == 4:
        screenCnt = approx
        flag=1
        break
if flag ==1:
    cv2.drawContours(image, [screenCnt], -1, (0, 255, 0), 3)
    
    pts = screenCnt.reshape(4, 2)
    rect = np.zeros((4, 2), dtype = "float32")
    s = pts.sum(axis = 1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis = 1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    rect *= ratio
    
    (tl, tr, br, bl) = rect
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    
    maxWidth = max(int(widthA), int(widthB))
    maxHeight = max(int(heightA), int(heightB))
    
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype = "float32")
    
    M = cv2.getPerspectiveTransform(rect, dst)
    warp = cv2.warpPerspective(orig, M, (maxWidth, maxHeight))
    
    gray_warp = cv2.cvtColor(warp, cv2.COLOR_BGR2GRAY)
    cv2.imshow("1234",gray_warp)
    #gray_warp_resized = imutils.resize(gray_warp, height=410, width= 664)
    di = (410, 664)
    resized = cv2.resize(gray_warp, di, interpolation = cv2.INTER_AREA)
    resized1 = cv2.resize(image1, di, interpolation = cv2.INTER_AREA)
    resized2 = cv2.resize(image2, di, interpolation = cv2.INTER_AREA)
    resizedL = cv2.resize(imageL, di, interpolation = cv2.INTER_AREA)
    resizedR = cv2.resize(imageR, di, interpolation = cv2.INTER_AREA)
    h,w=resized.shape
    
    ret, mask = cv2.threshold(resized, 127, 255, cv2.THRESH_BINARY )
    
    ret1, mask1 = cv2.threshold(resized1, 127, 255, cv2.THRESH_BINARY )
    dst1 = cv2.bitwise_xor(mask,mask1)
    whitecount1 = cv2.countNonZero(dst1)
    
    if whitecount1 < 11000:
        cv2.imshow('dst1',dst1)
        print("room1")
        print (whitecount1)
        
        
    ret2, mask2 = cv2.threshold(resized2, 127, 255, cv2.THRESH_BINARY )
    dst2 = cv2.bitwise_xor(mask,mask2)
    whitecount2 = cv2.countNonZero(dst2)
    cv2.imshow("dst2",dst2)
    if whitecount2 < 11000:
        #cv2.imshow('dst2',dst2)
        print("room2")
        print (whitecount2)
        
        
    retL, maskL = cv2.threshold(resizedL, 127, 255, cv2.THRESH_BINARY )
    dstL = cv2.bitwise_xor(mask,maskL)
    whitecountL = cv2.countNonZero(dstL)
    
    if whitecountL < 11000:
        cv2.imshow('dstL',dstL)
        print("left")
        print (whitecountL)
        
    retR, maskR = cv2.threshold(resizedR, 127, 255, cv2.THRESH_BINARY )
    dstR = cv2.bitwise_xor(mask,maskR)
    whitecountR = cv2.countNonZero(dstR)
    #cv2.imshow('dstR',dstR)
    print (whitecountR)
    if whitecountR < 11000:
        cv2.imshow('dstR',dstR)
        print("right")
        
    
    
    cv2.imshow("Image", image)
    #cv2.imshow("Canny",edged)
    cv2.imshow("warp", imutils.resize(warp, height = 400))
    cv2.waitKey(0)
rawCapture.truncate(0)
cv2.destroyAllWindows()
