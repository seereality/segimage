#!/usr/bin/python

import Image
import cv2
import numpy as np
import cv2.cv as cv
import matplotlib
matplotlib.rcParams['backend'] = "Qt4Agg"
from matplotlib import pyplot
import random
from scipy.spatial import Delaunay
if __name__ == '__main__':
    win = "source";
    rect = ( 0, 0, 600, 600 );
    
    active_facet_color = cv.RGB( 255, 0, 0 );
    delaunay_color  = cv.RGB( 0,0,0);
    voronoi_color = cv.RGB(0, 180, 0);
    bkgnd_color = cv.RGB(255,255,255);


       
    cv.NamedWindow( win, 1 );



    img = cv2.imread('top.jpeg');
    m,n = img.shape[:2];

    storage = cv.CreateMemStorage(0);
    subdiv = cv.CreateSubdivDelaunay2D( rect, storage );
    imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(imgray, 150, 255, cv2.THRESH_BINARY)
    cv2.imshow('1',thresh);
    cv2.waitKey(0);
    contours, heirarchy=cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    maxm = 0
    ind = 0
    for i in range(len(contours)):
            if maxm<len(contours[i]):
		    maxm = len(contours[i])
		    ind = i
    cnt= contours[ind]

    area = cv2.contourArea(cnt)
    print area
    
       

    fp = [] 
    hull = cv2.convexHull(cnt,returnPoints = False)
   
    defects = cv2.convexityDefects(cnt,hull)
    size =0
  
    for i in range(defects.shape[0]):
        s,e,f,d = defects[i,0]
	far = tuple(cnt[f][0])
	fp.append(far)
	size=size+1




    M = cv2.moments(cnt)
    centroid_x = int(M['m10']/M['m00'])
    centroid_y = int(M['m01']/M['m00'])
    fp.append((centroid_x, centroid_y))
    size=size+1

    im = cv.CreateImageHeader((img.shape[1], img.shape[0]), cv.IPL_DEPTH_8U, 3)
    cv.SetData(im, img.tostring(), img.dtype.itemsize * 3 * img.shape[1])
    
    for i in range(0,m,int(area/1000)):
	    for j in range(0,n,int(area/1000)):
		    p = cv2.pointPolygonTest(cnt,(i,j),False)
		    if(p>=0):
			    fp.append((i,j))
			    size=size+1

    print "Delaunay triangulation will be build now interactively."
    print "To stop the process, press any key\n";


    x=[];
    y=[];
    for i in range(size):
	    x.append(fp[i][0]);
	    y.append(fp[i][1])

    tri = Delaunay(fp)
    print tri.simplices
    y[:] = [250-w for w in y]
    pyplot.triplot(x,y, tri.simplices.copy())
    pyplot.plot(x,y, 'o')

    pyplot.show()
    cv2.destroyAllWindows();

