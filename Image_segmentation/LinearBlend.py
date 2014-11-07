'''
Created on 08-Oct-2014

@author: swetha
'''

import cv2
  
alpha = 0.5
beta = ( 1.0 - alpha )
imgsrc1 = cv2.imread('/home/swetha/Monsoon2014/Internsip/balls.png')
  
imgsrc2 = cv2.imread('/home/swetha/Monsoon2014/Internsip/balls_1.png')
  
dst = imgsrc1
  
cv2.addWeighted( imgsrc1, alpha, imgsrc2, beta, 0.0, dst)
  
cv2.namedWindow('image', cv2.WINDOW_NORMAL)
cv2.imshow('image',dst)
cv2.waitKey(0)


""" Method 2 """

# s_img = cv2.imread("smaller_Image.png") # contour
l_img = cv2.imread("larger_Image.jpg") # Whole Image
x_offset = 50
y_offset = 0
 
s_img = cv2.imread("smaller_Image.png", -1)
for c in range(0,3):
    l_img[ y_offset : y_offset + s_img.shape[0], x_offset : x_offset + s_img.shape[1], c ] = s_img[ :, : , c ] * ( s_img[ :, :, 3 ]/255.0 ) + l_img[ y_offset : y_offset + s_img.shape[0], x_offset : x_offset + s_img.shape[1], c] * (1.0 - s_img[:,:,3]/255.0)
     
cv2.namedWindow('image update', cv2.WINDOW_NORMAL)
cv2.imshow('image update',l_img)
cv2.waitKey(0)
