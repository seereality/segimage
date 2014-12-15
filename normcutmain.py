#Main code for normalised cut
import ncut1
from scipy.misc import imresize
import cv2
import numpy as np
from PIL import Image

#im = cv2.imread('saree.JPG')
im = np.asarray(Image.open('saree.JPG'))
m,n = im.shape[:2]

# resize image to (wid,wid)
wid = 50
rim = imresize(im,(wid,wid),interp='bilinear')
rim = np.asarray(rim,'f')
# create normalized cut matrix
A = ncut1.ncut_graph_matrix(rim,sigma_d=1,sigma_g=1e-2)
# cluster
code,V = ncut1.cluster(A,k=3,ndim=3)
# reshape to original image size
codeim = imresize(code.reshape(wid,wid),(m,n),interp='nearest')
# plot result
#figure()
cv2.imshow('tal',codeim)
cv2.waitKey(0)
cv2.destroyAllWindows()
#gray()
#show()

	
