import numpy as np
import cv2
from PIL import Image

# Extracting the mask of the cloth
im = Image.open('saree5.JPG')
cloth =cv2.imread('saree5.JPG')

#cloth = cv2.GaussianBlur(cloth,(5,5),0)
#cloth = cv2.bilateralFilter(cloth,5,75,75)
#cloth = cv2.medianBlur(cloth,3)
#kernel = np.ones((3,3),np.float32)/9
#cloth = cv2.filter2D(cloth,-1,kernel)

cloth_gray=cv2.cvtColor(cloth, cv2.COLOR_BGR2GRAY)
res_c, thresh_c = cv2.threshold(cloth_gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

#thresh_c = cv2.adaptiveThreshold(cloth_gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
#            cv2.THRESH_BINARY,11,1)

fg = cv2.erode(thresh_c,None,iterations = 2)
bgt = cv2.dilate(thresh_c,None,iterations = 3)
ret,bg = cv2.threshold(bgt,1,128,1)
marker = cv2.add(fg,bg)
marker32 = np.int32(marker)
cv2.watershed(cloth,marker32)
m = cv2.convertScaleAbs(marker32)

#thresh_c = cv2.adaptiveThreshold(m ,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
#            cv2.THRESH_BINARY,11,1)
res_c, thresh_c = cv2.threshold(cloth_gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)


#cv2.imshow('win',cloth_gray)

#ret_c, thresh_c = cv2.adaptiveThreshold(cloth_gray,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,2)
#ret_c, thresh_c = cv2.threshold(cloth_gray, 230, 255, cv2.THRESH_BINARY_INV)

contours_c, heirarchy_c=cv2.findContours(thresh_c, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
maxm = 0
ind = 0
for i in range(len(contours_c)):
        if maxm<len(contours_c[i]):
                maxm = len(contours_c[i])
                ind = i
cnt_c= contours_c[ind]

mask = np.zeros(cloth_gray.shape, np.uint8)
cv2.drawContours(mask,[cnt_c],0,255,-1)
alpha = Image.fromarray(np.uint8(mask))
im.putalpha(alpha)
im.save('alpha_top45.png');
#cv2.destroyAllWindows()
