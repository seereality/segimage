import numpy as np
import cv2
import cv

im = cv2.imread('/home/swetha/Monsoon2014/Internsip/balls.png')
imgray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)

cv2.namedWindow('image', cv2.WINDOW_NORMAL)
cv2.imshow('image',im)
cv2.wawaitKey(0)

ret,thresh = cv2.threshold(imgray,127,255,0)
contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(im,contours,-1,(0,255,0),3)

print len(contours)
cnt = contours[0]
moments = cv2.moments(cnt)

area = cv2.contourArea(cnt)

print area

x,y,w,h = cv2.boundingRect(cnt)
cv2.rectangle(im,(x,y),(x+w,y+h),(0,255,0),2)
 
cv2.namedWindow('image1', cv2.WINDOW_NORMAL)
cv2.imshow('image1',im)
cv2.waitKey(0)
 
 
 
rect = cv2.minAreaRect(cnt)
box = cv2.cv.BoxPoints(rect)
box = np.int0(box)
cv2.drawContours(im,[box],0,(0,0,255),2)

cv2.namedWindow('image2', cv2.WINDOW_NORMAL)
cv2.imshow('image2',im)
cv2.waitKey(0)


(x,y),radius = cv2.minEnclosingCircle(cnt)
center = (int(x),int(y))
radius = int(radius)
cv2.circle(im,center,radius,(0,255,0),2)

ellipse = cv2.fitEllipse(cnt)
cv2.ellipse(im,ellipse,(0,255,0),2)
 
cv2.namedWindow('image3', cv2.WINDOW_NORMAL)
cv2.imshow('image3',im)
cv2.waitKey(0)
 
cv2.destroyAllWindows()
