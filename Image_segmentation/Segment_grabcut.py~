'''
Created on 05-Oct-2014

@author: swetha
'''
import numpy as np
import cv2
from matplotlib import pyplot as plt

img = cv2.imread('/home/swetha/Monsoon2014/Internship/Man_formal.jpg')
mask = np.zeros(img.shape[:2],np.uint8)
# plt.imshow(img),plt.show()

bgdModel = np.zeros((1,65),np.float64)
fgdModel = np.zeros((1,65),np.float64)
 
rect = (0,0,170,120)
cv2.grabCut(img,mask,rect,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_RECT)

mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
img = img*mask2[:,:,np.newaxis]
 
# plt.imshow(img),plt.colorbar(),plt.show()
print "done!"

imgray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

cv2.namedWindow('image', cv2.WINDOW_NORMAL)
cv2.imshow('image',img)
cv2.waitKey(0)

ret,thresh = cv2.threshold(imgray,127,255,0)
contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(img,contours,-1,(0,255,0),3)

cv2.namedWindow('image1', cv2.WINDOW_NORMAL)
cv2.imshow('image1',img)
cv2.waitKey(0)
