import numpy as np
import cv2
from PIL import Image

# Extracting the mask of the cloth
im = Image.open('saree.JPG')
cloth =cv2.imread('saree.JPG')

cloth_gray=cv2.cvtColor(cloth, cv2.COLOR_BGR2GRAY)
thresh_c = cv2.adaptiveThreshold(cloth_gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv2.THRESH_BINARY,11,2)
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
im.save('alpha_top4.png');
cv2.destroyAllWindows()
