import numpy as np
import cv2

from cv2 import *

from PIL import Image

# Extracting the mask of the cloth
cloth =cv2.imread('top.jpeg')
cloth_gray=cv2.cvtColor(cloth, cv2.COLOR_BGR2GRAY)
ret_c, thresh_c = cv2.threshold(cloth_gray, 230, 255, cv2.THRESH_BINARY_INV)
contours_c, heirarchy_c=cv2.findContours(thresh_c, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
maxm = 0
ind = 0
for i in range(len(contours_c)):
        if maxm<len(contours_c[i]):
                maxm = len(contours_c[i])
                ind = i

cnt_c= contours_c[ind]
cloth_im = Image.fromarray(np.uint8(cloth))
mask = np.zeros(cloth_gray.shape, np.uint8)
cv2.drawContours(mask,[cnt_c],0,255,-1)
alpha = Image.fromarray(np.uint8(mask))
cloth_im.putalpha(alpha)
cloth_im.save('alphacloth.png')

M_c=cv2.moments(cnt_c)
cc_x = int(M_c['m10']/M_c['m00'])
cc_y=int(M_c['m01']/M_c['m00'])

# Capturing the human
cap=cv2.VideoCapture(0)
reti, framei = cap.read()
framei_g = cv2.cvtColor(framei, cv2.COLOR_BGR2GRAY)

im = framei

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

while(1):
	ret,frame = cap.read()
	im =frame
	img = frame
	
	frame_g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	gray= frame_g
	faces = face_cascade.detectMultiScale(frame_g, 1.3, 5)
	
	for (x,y,w,h) in faces:
	    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
	    roi_gray = gray[y:y+h, x:x+w]
	    roi_color = img[y:y+h, x:x+w]
	    eyes = eye_cascade.detectMultiScale(roi_gray)
	    for (ex,ey,ew,eh) in eyes:
		cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

	    newx = (int)(x+w/2 - 1.5*h)
	    newy = (int)(y+1.25*h)
	    cv2.rectangle(img,(newx, newy),(newx+3*h, newy+3*h),(0,0,255),2)


	diff = frame_g-framei_g
	(thresh, im_bw) = cv2.threshold(diff, 200, 255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)
	contours, hierarchy = cv2.findContours(im_bw,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
	
	maxm = 0
	ind = 0
	for i in range(len(contours)):
	        if maxm<len(contours[i]):
	                maxm = len(contours[i])
	                ind = i

	cnt= contours[ind]
	hull = cv2.convexHull(cnt, returnPoints=False)
	M=cv2.moments(cnt)
	cv2.drawContours(im,[cnt],-1,(255,255,255),-1)
	posx = int(M['m10']/M['m00'])
	posy=int(M['m01']/M['m00'])
	cv2.circle(im, (posx,posy),6,(255,0,0),-1)

	defects = cv2.convexityDefects(cnt, hull)
	for i in range(defects.shape[0]):
		s,e,f,d=defects[i,0]
		start=tuple(cnt[s][0])
		end=tuple(cnt[e][0])
		if d > 10000:
			far = tuple(cnt[f][0])
			cv2.line(im,start,end,[0,255,0],2)
			cv2.circle(im,far,5,[0,0,255],-1)
	

	cv2.imshow('wind',img)
	k=cv2.waitKey(30) & 0xFF
	if k==27:
		break

cap.release()

cv2.destroyAllWindows()
