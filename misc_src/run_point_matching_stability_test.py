__author__ = 'kroegert'


import sys
import matplotlib.pyplot as plt
import cv2
import os
import scipy
from glob import glob
import sys
import numpy as np
from scipy import signal
from func_OF_util import func_read_flo_file, func_extract_bil_patch, func_extract_NN_patch, func_get_pat_cosmask, gauss2Dfilter

sys.path.append("../DL_Architectures_Tricks")
from func_util import func_img_channel_write, func_img_channel_read


# Purpose: Test point matching stability using different matching metrics

# Global parameters
nopts = 200
scfct = 4
psize_all = 16 # patch size in px


# Load Images
imgpath = '/home/kroegert/local/Datasets/Sintel-Stereo/training/final_left/'
#imgaf = 'market_2/frame_0004.png'
#imgbf = 'market_2/frame_0005.png'
imgaf = 'market_5/frame_0030.png'
imgbf = 'market_5/frame_0031.png'
#imgaf = 'bamboo_2/frame_0036.png'
#imgbf = 'bamboo_2/frame_0037.png'


imga = plt.imread(imgpath+imgaf).astype(float)
imgb = plt.imread(imgpath+imgbf).astype(float)
imga = scipy.misc.imresize(imga, (imga.shape[0]/scfct, imga.shape[1]/scfct), interp='bilinear')
imgb = scipy.misc.imresize(imgb, (imgb.shape[0]/scfct, imgb.shape[1]/scfct), interp='bilinear')
width = imga.shape[1]
height = imga.shape[0]

imgflowGT = func_read_flo_file('/home/kroegert/local/Datasets/OF_Sintel/test/' + imgaf[:-3] + "flo")
#imgflowGT = np.zeros((height, width, 2), dtype=(np.float32))

imgflowGT = np.stack((scipy.misc.imresize(imgflowGT[:,:,0], (height, width), interp='bilinear', mode='F') * 1.0/float(scfct), scipy.misc.imresize(imgflowGT[:,:,1], (height, width), interp='bilinear', mode='F') * 1.0/float(scfct)), axis=2)

invalid = (plt.imread('/home/kroegert/local/Datasets/OF_Sintel/invalid/' + imgaf) + plt.imread('/home/kroegert/local/Datasets/OF_Sintel/occlusions/' + imgaf)) > 0.5
invalid = scipy.misc.imresize(invalid, (height, width), interp='nearest', mode='F')
  

# Create random keypoints
pts1 = np.stack((np.random.randint(psize_all/2+1, height-psize_all/2-1, nopts), np.random.randint(psize_all/2+1, width-psize_all/2-1, nopts)), axis=1)
pts2 = np.copy(pts1)
for i in xrange(nopts):
  pts2[i,0] += np.round(imgflowGT[pts1[i,0], pts1[i,1], 1]).astype(int)
  pts2[i,1] += np.round(imgflowGT[pts1[i,0], pts1[i,1], 0]).astype(int)

idxdel = np.where((pts2[:,0] <= psize_all) | (pts2[:,1] <= psize_all) | (pts2[:,0] >= height-psize_all-1) | (pts2[:,1] >= width-psize_all-1) | (invalid[pts1[:,0], pts1[:,1]] == 1) )
pts1 = np.delete(pts1, idxdel, axis=0)
pts2 = np.delete(pts2, idxdel, axis=0)
nopts = pts1.shape[0]

# keep only 25 points with strongest motion
nnorm = np.linalg.norm(pts1-pts2, axis=1)
idxkeep = np.argsort(nnorm)[-25:]
pts1 = pts1[idxkeep, :]
pts2 = pts2[idxkeep, :]
nopts = pts1.shape[0]



# Visualize randomly extracted patch here, check for correctness
plt.ion()
plt.subplot(2,1,1)
plt.imshow(imga, interpolation='nearest')
plt.hold('on')
plt.scatter(pts1[:,1], pts1[:,0], 15, 'g')
plt.subplot(2,1,2)
plt.imshow(imgb, interpolation='nearest')
plt.hold('on')
plt.scatter(pts2[:,1], pts2[:,0], 15, 'g')
plt.plot(np.stack((pts1[:,1], pts2[:,1])), np.stack((pts1[:,0], pts2[:,0])), 'r')
plt.show()

# compute scoring array of matching in local neighborhood of GT location

## Norm of channel difference, in: imga, imgb, pts1, pts2, parameter_struct
nopts = pts1.shape[0]
nochannels = imga.shape[2]
do_log = 0 # Take log of absolute channel values (sign(log(|x|))
do_zeromean = 1 # Zero-mean of channels values
do_unitnorm = 0 # Unit-normalize patches
do_normsel = 0; # 0: L2, 1: L1, 2: Huber, 3: L_Inf


def func_create_imgstack(imga):
  img_stack = []
  for x in xrange(-psize_all/2, psize_all/2):
    for y in xrange(-psize_all/2, psize_all/2):
      img_cut = imga[(psize_all/2 + y):(height-psize_all/2 + y), (psize_all/2 + x):(width-psize_all/2 + x) , :]
      #print img_cut.shape
      img_stack.append(img_cut)
  img_stack = np.stack(img_stack, axis=3).astype(float)    
  
  if (do_log==1):
    img_stack = np.sign(img_stack) * np.log(np.minimum(255,np.maximum(0.1,np.abs(img_stack))))

  if (do_zeromean==1):
    img_stack -= np.mean(img_stack, axis=(3))[:,:,:,None]
    
  if (do_unitnorm==1):
    img_stack /= np.maximum(np.linalg.norm(img_stack, axis=(3)), 1e-20)[:,:,:,None]
    
  return img_stack
    
imga_stack = func_create_imgstack(imga)
imgb_stack = func_create_imgstack(imgb)


pat_res = []
for i in xrange(nopts):
  pat_t  = imga_stack[pts1[i,0]-psize_all/2, pts1[i,1]-psize_all/2, :, :]
  diff = imgb_stack - pat_t[None,None,:,:]
  pts2_new = pts2[i,:]-psize_all/2
  diff = diff[(pts2_new[0]-psize_all/2):(pts2_new[0]+psize_all/2), (pts2_new[1]-psize_all/2):(pts2_new[1]+psize_all/2), :,:]
  if (do_normsel==0):
    diff = np.linalg.norm(diff, 2, axis=(3))
  if (do_normsel==1):
    diff = np.linalg.norm(diff, 1, axis=(3))
  #if (do_normsel==2):
    # TODO: Huber norm
  if (do_normsel==3):
    diff = np.linalg.norm(diff, np.Inf, axis=(3))
  diff = np.sum(diff, axis=(2))
  pat_res.append(np.copy(diff))  
  #plt.ion()
  #plt.imshow(diff, interpolation='nearest')
  #plt.show()
pat_res = np.stack(pat_res, axis=2)
pat_res = -(pat_res - np.max(pat_res, axis=(0,1))[None,None,:]) # convert to score
pat_res /= np.linalg.norm(pat_res, axis=(0,1))[None, None,:]






## Correlation, imga, imgb, pts1, pts2, parameter_struct
filtertype = 1; #0: NCC, 1: MOSSE
nopts = pts1.shape[0]
nochannels = imga.shape[2]
do_log = 0 # Take log of absolute channel values (sign(log(|x|))
do_zeromean = 1 # Zero-mean of channels values
do_unitnorm = 1 # Unit-normalize patches
do_circweight = 1
do_meanmax = 0 # 0: mean over channels, 1: max over channels
gsigma = 2
beta = 0.01

if (do_circweight):
  ptmask = func_get_pat_cosmask(psize_all) # Circular mask
else:
  ptmask = None

if (filtertype==1):
  gfilt = gauss2Dfilter(shape=(psize_all,psize_all),sigma=gsigma) # adapt this to cornerness/discriminativness of point in first image
  gfilt_fft = np.fft.fft2(gfilt, [psize_all, psize_all])

  

pat_res = []
for i in xrange(nopts):
  pat_t  = func_extract_NN_patch(pts1[i,[1, 0]], imga.astype(float), psize_all, use_mask=ptmask, do_log=do_log, do_zeromean=do_zeromean, do_unitnorm=do_unitnorm, do_flatten=0)  # target patch, with
  pat_q  = func_extract_NN_patch(pts2[i,[1, 0]], imgb.astype(float), psize_all, use_mask=ptmask, do_log=do_log, do_zeromean=do_zeromean, do_unitnorm=do_unitnorm, do_flatten=0)  # target patch, with
  
  pat_t_fft = np.stack([np.fft.fft2(pat_t[:,:,x], [psize_all, psize_all]) for x in xrange(pat_t.shape[2])], axis=2)
  pat_q_fft = np.stack([np.fft.fft2(pat_q[:,:,x], [psize_all, psize_all]) for x in xrange(pat_q.shape[2])], axis=2)

  # NAIVE, NCC between target and query patch, 
  if (filtertype==0):
    res = [np.fft.ifft2(pat_q_fft[:,:,x] * np.conj(pat_t_fft[:,:,x])) for x in xrange(pat_q.shape[2])]
    res = np.maximum(0,np.real(np.fft.fftshift(res)))
  if (filtertype==1):
    pat_t_fft_c = np.conj(pat_t_fft) # complex conjugate 
    hfilt_fft = np.stack([np.divide(gfilt_fft * pat_t_fft_c[:,:,x], pat_t_fft[:,:,x] * pat_t_fft_c[:,:,x] + beta) for x in xrange(pat_t_fft.shape[2])], axis=2)
    res = np.stack([np.maximum(0,np.real(np.fft.ifft2(pat_q_fft[:,:,x] * hfilt_fft[:,:,x]))) for x in xrange(pat_t_fft.shape[2])], axis=0)

  if (do_meanmax==0):
    res = np.mean(res, axis=0)
  else:
    res = np.max(res, axis=0)

  pat_res.append(np.copy(res))  
pat_res = np.stack(pat_res, axis=2)
pat_res /= np.linalg.norm(pat_res, axis=(0,1))[None, None,:]


def func_get_entropy(pat_in, minv = 1e-8):
  pat = np.copy(pat_in)
  pat = np.maximum(pat, minv)
  pat /= np.linalg.norm(pat, axis=(0,1))[None, None,:]
  return -np.sum(np.log(pat) * pat, axis=(0,1))




plt.figure()
for i in xrange(25):
  plt.subplot(5,5,i+1)
  plt.imshow(pat_res[:,:,i], interpolation='nearest')
plt.show()
print np.mean(func_get_entropy(pat_res))



# TODO: Initialize pts2 with OF, check stability of peaks with entropy
# TODO: Initialize pts2 with 0 , check stability of peaks would be reached by 1) maximum, 2)

