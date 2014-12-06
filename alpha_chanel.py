import numpy as np
import cv2
from PIL import Image

# Extracting the mask of the cloth
im = Image.open('top4.jpg')
cloth =cv2.imread('top4.jpg')
cloth_gray=cv2.cvtColor(cloth, cv2.COLOR_BGR2GRAY)
ret_c, thresh_c = cv2.threshold(cloth_gray, 230, 255, cv2.THRESH_BINARY_INV)
contours_c, heirarchy_c=cv2.findContours(thresh_c, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
maxm = 0
ind = 0
for i in range(len(contours_c)):
        if maxm<len(contours_c[i]):
                maxm = len(contours_c[i])
                ind = i
#	print maxm
#	print ind
cnt_c= contours_c[ind]
#im = Image.fromarray(np.uint8(cloth))
#im.save('try.png')
mask = np.zeros(cloth_gray.shape, np.uint8)
cv2.drawContours(mask,[cnt_c],0,255,-1)
alpha = Image.fromarray(np.uint8(mask))
im.putalpha(alpha)
#im.save('alpha_top2.jpeg')

#topaste = Image.fromarray(np.uint8(mask))
#alpha.paste(mask, (0,0), mask)
#alpha.show()
#cv2.imshow('mask',mask)
#cv2.waitKey(0)


#res_c = cv2.bitwise_and(cloth,cloth, mask=mask)
#im = Image.new("RGB", (512, 512), "white")
#Image.composite(im, im, mask)
#im.putalpha(mask)






im.save('alpha_top4.png');
#cv2.imshow('top',im)
#cv2.waitKey(0)


'''

# Capturing the human
cap=cv2.VideoCapture(0)
reti, framei = cap.read()
framei_g = cv2.cvtColor(framei, cv2.COLOR_BGR2GRAY)

cv2.imshow('init', framei_g)
cv2.waitKey(0)
im = framei

while(1):
	ret,frame = cap.read()
	im =frame
	frame_g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	diff = frame_g-framei_g
#	diff = cv2.absdiff(frame_g, framei_g)
	(thresh, im_bw) = cv2.threshold(diff, 200, 255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)
	contours, hierarchy = cv2.findContours(im_bw,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
	
	maxm = 0
	ind = 0
	for i in range(len(contours)):
	        if maxm<len(contours[i]):
	                maxm = len(contours[i])
	                ind = i
#	print maxm
#	print ind
	cnt= contours[ind]
	hull = cv2.convexHull(cnt)


	#cnt = contours[0]
#i	print len(contours)
#	cv2.polylines(im,[hull], True, (255,0,0))
	cv2.drawContours(im,[cnt],-1,(255,255,255),-1)

	mask = np.zeros(frame_g.shape, np.uint8)
	cv2.drawContours(mask,[cnt],0,255,-1)
	res_c = cv2.bitwise_and(cloth,cloth, mask=mask)
	cv2.imshow('top',res_c)
	
	
	#cv2.drawContours(im_bw,contours,-1,(0,255,0),-1)
#	cv2.imshow('wind',im)
	k=cv2.waitKey(30) & 0xFF
	if k==27:
		break

cap.release()
'''
cv2.destroyAllWindows()
