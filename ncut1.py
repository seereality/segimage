#Contains functions that are called in normcutmain.py
import numpy as np
from scipy.cluster.vq import *

def cluster(S,k,ndim):
	""" Spectral clustering from a similarity matrix."""
	# check for symmetry
	if sum(sum(abs(S-S.T))) > 1e-10:
		print 'not symmetric'

	# create Laplacian matrix
	#rowsum = sum(abs(S),axis=0)
	rowsum = sum(abs(S))
	D =np.diag(1 / np.sqrt(rowsum + 1e-6))
	L = np.dot(D,np.dot(S,D))
	
	# compute eigenvectors of L
	U,sigma,V = np.linalg.svd(L)
	
	# create feature vector from ndim first eigenvectors
	#by stacking eigenvectors as columns
	features = np.array(V[:ndim]).T
	
	# k-means
	features = whiten(features)
	centroids,distortion = kmeans(features,k)
	code,distance = vq(features,centroids)
	return code,V


def ncut_graph_matrix(im,sigma_d=1e2,sigma_g=1e-2):
	""" Create matrix for normalized cut. The parameters are
	the weights for pixel distance and pixel similarity. """
	m,n = im.shape[:2]
	N = m*n
	
	# normalize and create feature vector of RGB or grayscale
	if len(im.shape)==3:
		for i in range(3):
			im[:,:,i] = im[:,:,i] / im[:,:,i].max()
	
		vim = im.reshape((-1,3))
	else:
		im = im / im.max()
		vim = im.flatten()

	# x,y coordinates for distance computation
	xx,yy = np.meshgrid(range(n),range(m))
	x,y = xx.flatten(),yy.flatten()
	
	# create matrix with edge weights
	W = np.zeros((N,N),'f')
	for i in range(N):
		for j in range(i,N):
			d = (x[i]-x[j])**2 + (y[i]-y[j])**2
			W[i,j] = W[j,i] = np.exp(-1.0*sum((vim[i]-vim[j])**2)/sigma_g) * np.exp(-d/sigma_d)
	
	return W

