'''
Created on 28-Sep-2014

@author: swetha
'''

'''
Simple Threshold
'''
import cv2
import numpy as np
from matplotlib import pyplot as plt
  
cimg = cv2.imread('/home/swetha/Monsoon2014/Honors/Images/myntra_760.jpg')
img = cv2.cvtColor(cimg,cv2.COLOR_BGR2GRAY)
ret,thresh1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
ret,thresh2 = cv2.threshold(img,127,255,cv2.THRESH_BINARY_INV)
ret,thresh3 = cv2.threshold(img,127,255,cv2.THRESH_TRUNC)
ret,thresh4 = cv2.threshold(img,127,255,cv2.THRESH_TOZERO)
ret,thresh5 = cv2.threshold(img,127,255,cv2.THRESH_TOZERO_INV)
  
thresh = ['img','thresh1','thresh2','thresh3','thresh4','thresh5']
  
for i in xrange(6):
    plt.subplot(2,3,i+1),plt.imshow(eval(thresh[i]),'gray')
    plt.title(thresh[i])
  
plt.show()

'''
Adaptive Thresholding
'''
  
cimg = cv2.imread('/home/swetha/Monsoon2014/Honors/Images/myntra_760.jpg')
img = cv2.cvtColor(cimg,cv2.COLOR_BGR2GRAY)
img = cv2.medianBlur(img,5)
  
ret,th1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
th2 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
            cv2.THRESH_BINARY,11,2)
th3 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv2.THRESH_BINARY,11,2)
  
plt.subplot(2,2,1),plt.imshow(img,'gray')
plt.title('input image')
plt.subplot(2,2,2),plt.imshow(th1,'gray')
plt.title('Global Thresholding')
plt.subplot(2,2,3),plt.imshow(th2,'gray')
plt.title('Adaptive Mean Thresholding')
plt.subplot(2,2,4),plt.imshow(th3,'gray')
plt.title('Adaptive Gaussian Thresholding')
  
plt.show()

'''
Otsu Binarization
'''

cimg = cv2.imread('/home/swetha/Monsoon2014/Internship/Man_formal.jpg')
img = cv2.cvtColor(cimg,cv2.COLOR_BGR2GRAY)
 
# global thresholding
ret1,th1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
 
# Otsu's thresholding
ret2,th2 = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
 
# Otsu's thresholding after Gaussian filtering
blur = cv2.GaussianBlur(img,(5,5),0)
ret3,th3 = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
 
# plot all the images and their histograms
titles = ['img','histogram1','th1',
          'img','histogram2','th2',
          'blur','histogram3','th3']
 
for i in xrange(3):
    plt.subplot(3,3,i*3+1),plt.imshow(eval(titles[i*3]),'gray')
    plt.title(titles[i*3])
    plt.subplot(3,3,i*3+2),plt.hist(eval(titles[i*3]).ravel(),256)
    plt.title(titles[i*3+1])
    plt.subplot(3,3,i*3+3),plt.imshow(eval(titles[i*3+2]),'gray')
    plt.title(titles[i*3+2])

plt.show()


'''
Otsu
'''

cimg = cv2.imread('/home/swetha/Monsoon2014/Internship/Man_formal.jpg')
img = cv2.cvtColor(cimg,cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(img,(5,5),0)
  
# find normalized_histogram, and its cum_sum
hist = cv2.calcHist([blur],[0],None,[256],[0,256])
hist_norm = hist.ravel()/hist.max()
Q = hist_norm.cumsum()
  
bins = np.arange(256)
  
fn_min = np.inf
thresh = -1
  
for i in xrange(1,256):
    p1,p2 = np.hsplit(hist_norm,[i]) # probabilities
    q1,q2 = Q[i],Q[255]-Q[i] # cum sum of classes
    b1,b2 = np.hsplit(bins,[i]) # weights
      
    # finding means and variances
    m1,m2 = np.sum(p1*b1)/q1, np.sum(p2*b2)/q2 
    v1,v2 = np.sum(((b1-m1)**2)*p1)/q1,np.sum(((b2-m2)**2)*p2)/q2
      
    # calculates the minimization function
    fn = v1*q1 + v2*q2
    if fn < fn_min:
        fn_min = fn
        thresh = i
  
# # find otsu's threshold value with OpenCV function 
# ret, otsu = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
# print thresh,ret
