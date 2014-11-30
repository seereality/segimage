import numpy as np
import cv2

from cv2 import *

#src = cv.LoadImage("over.jpg")	# Load a source image
#overlay = cv.LoadImage("face.png",-1)	# Load an image to overlay
#posx = 50	# Define a point (posx, posy) on the source
#posy = 50	# image where the overlay will be placed
S = (0,0,0,0)	# Define blending coefficients S and D
D = (1,1,1,1)	

def OverlayImage(src, overlay, posx, posy, S, D):
	h, w, depth = overlay.shape
	hs,ws,ds = src.shape
	for x in range(-w/2,w/2,1):
		if x+posx < ws and x+posx>=0:
			for y in range(-h/2,h/2,1):
				if y+posy < hs and y+posy>=0:
					source = cv2.cv.Get2D(src, y+posy, x+posx)
					over = cv2.cv.Get2D(overlay, y+(h/2), x+(w/2))
					merger = [0, 0, 0, 0]
					for i in range(3):
						merger[i] = (S[i]*source[i]+D[i]*over[i])
						merged = tuple(merger)
						cv2.cv.Set2D(src, y+posy, x+posx, merged)

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
#	print maxm
#	print ind
cnt_c= contours_c[ind]

mask = np.zeros(cloth_gray.shape, np.uint8)
cv2.drawContours(mask,[cnt_c],0,255,-1)
res_c = cv2.bitwise_and(cloth,cloth, mask=mask)
M_c=cv2.moments(cnt_c)
cc_x = int(M_c['m10']/M_c['m00'])
cc_y=int(M_c['m01']/M_c['m00'])
#cv2.circle(res_c, (cc_x,cc_y),6,(255,0,0),-1)
#cv2.imshow('top',res_c)
#cv2.waitKey(0)
overlay = res_c


# Capturing the human
cap=cv2.VideoCapture(0)
reti, framei = cap.read()
framei_g = cv2.cvtColor(framei, cv2.COLOR_BGR2GRAY)

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
	hull = cv2.convexHull(cnt, returnPoints=False)
	M=cv2.moments(cnt)
	cv2.drawContours(im,[cnt],-1,(255,255,255),-1)
	posx = int(M['m10']/M['m00'])
	posy=int(M['m01']/M['m00'])
	cv2.circle(im, (posx,posy),6,(255,0,0),-1)

	#cnt = contours[0]
#i	print len(contours)
#	cv2.polylines(im,[hull], True, (255,0,0))
	defects = cv2.convexityDefects(cnt, hull)
	for i in range(defects.shape[0]):
		s,e,f,d=defects[i,0]
		start=tuple(cnt[s][0])
		end=tuple(cnt[e][0])
		if d > 10000:
			far = tuple(cnt[f][0])
#			print d
			cv2.line(im,start,end,[0,255,0],2)
			cv2.circle(im,far,5,[0,0,255],-1)

	#cv2.drawContours(im,[cnt],-1,(255,255,255),-1)

	#mask = np.zeros(frame_g.shape, np.uint8)
	#cv2.drawContours(mask,[cnt],0,255,-1)
	#res_c = cv2.bitwise_and(cloth,cloth, mask=mask)
	#cv2.imshow('top',res_c)
	

######## Image overlay

	OverlayImage(im, overlay, posx, posy, S, D)
	#cv2.imsh('src.png', src) #Saves the image
	#print "Done"

	
#	cv2.drawContours(im_bw,contours,-1,(0,255,0),-1)
	cv2.imshow('wind',im)
	k=cv2.waitKey(30) & 0xFF
	if k==27:
		break

cap.release()

cv2.destroyAllWindows()
