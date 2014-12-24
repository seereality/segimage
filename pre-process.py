import cv2
import numpy as np

def denoise(src)
    """ Denoising using gaussian blur """
    dest = cv2.GaussianBlur(src, (5,5), 0)
    return dest

def change_size(src, dsize)
    """ Change the size of the image for faster operations """
    dest = cv2.resize(src, dsize)
    return dest

def sharpen(src)
    """ Sharpen the image using the unsharp masking technique """
    temp = denoise(src)
    dest = cv2.addWeighted(temp, 1.5, im, -0.5, 0)
    return dest
