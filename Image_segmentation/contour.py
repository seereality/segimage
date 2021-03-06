import numpy as np
import cv2
import cv
 
im = cv2.imread('/home/swetha/Monsoon2014/Internsip/balls.png')

imgray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)

cv2.namedWindow('image', cv2.WINDOW_NORMAL)
cv2.imshow('image',imgray)
cv2.waitKey(0)

ret,thresh = cv2.threshold(imgray,127,255,0)
contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(im,contours,-1,(0,255,0),3)

cv2.namedWindow('image1', cv2.WINDOW_NORMAL)
cv2.imshow('image1',im)
cv2.waitKey(0)

# Now use masks

cv2.namedWindow('image2', cv2.WINDOW_NORMAL)
for h,cnt in enumerate(contours):
    mask = np.zeros(imgray.shape,np.uint8)
    cv2.drawContours(mask,[cnt],0,255,-1)
    mean = cv2.mean(im,mask = mask)
    cv2.imshow('image2',mask)
    cv2.waitKey(0)

cv2.destroyAllWindows()
