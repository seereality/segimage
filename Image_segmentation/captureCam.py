'''
Created on 28-Sep-2014

@author: swetha
'''

import cv2.cv as cv
 
if __name__ == '__main__':
    capture = cv.CaptureFromCAM(0)
    cv.NamedWindow("camera", 0) 
     
    while True:
        img = cv.QueryFrame(capture)
        cv.ShowImage("camera", img)
        if cv.WaitKey(33) == 27:
            break
        elif cv.WaitKey(33) == ord('c'):
            place = '/home/swetha/test_opencv.png'
            cv.SaveImage(place, img)
            print 'Captured :D'
            
    cv.DestroyAllWindows()
