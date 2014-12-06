import PIL
from PIL import Image
from PIL import ImageChops # used for multiplying images
import numpy as np
import cv2

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')


def black_onto(img1, img2,newx,newy,h):  
    # create blank white canvas to put img2 onto
    resized = Image.new("RGB", img1.size, "white")

    # define where to paste mask onto canvas
    img1_w, img1_h = img1.size
    img2_w, img2_h = img2.size
    #box = (img1_w/2-img2_w/2, img1_h/2-img2_h/2, img1_w/2-img2_w/2+img2_w, img1_h/2-img2_h/2+img2_h)
    box = (newx, newy,newx+3*h, newy+3*h)
    # multiply new mask onto image
    resized.paste(img2, box)
    return ImageChops.multiply(img1, resized)

# open images
painting = Image.open("fem2.jpg")
mask     = Image.open("alpha_top4.png")
img = np.asarray(painting)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

faces = face_cascade.detectMultiScale(gray, 1.3, 5)
print faces

for (x,y,w,h) in faces:
    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = img[y:y+h, x:x+w]
    #eyes = eye_cascade.detectMultiScale(roi_gray)
    #for (ex,ey,ew,eh) in eyes:
      #  cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

    newx = (int)(x+w/2 - 1.5*h)
    newy = (int)(y+1.25*h)
    cv2.rectangle(img,(newx, newy),(newx+3*h, newy+3*h),(0,0,255),2)

    mask = mask.resize((3*h, 3*h), Image.ANTIALIAS)
    #size = (3*h,3*h)
    #mask.thumbnail(size, Image.ANTIALIAS)
    out = black_onto(painting, mask,newx,newy,h)
    out.save('try4.jpg') # this gives the output image shown above
