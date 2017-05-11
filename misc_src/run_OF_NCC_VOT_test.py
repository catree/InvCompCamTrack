__author__ = 'kroegert'


import sys
import matplotlib.pyplot as plt
import cv2
import os
from glob import glob
import sys
import numpy as np
from scipy import signal
from func_OF_util import func_extract_bil_patch, func_get_pat_cosmask, gauss2Dfilter

sys.path.append("../DL_Architectures_Tricks")
from func_util import func_img_channel_write, func_img_channel_read



# Parameters
psize = 128 # patch size in px
locquery = np.array([246, 201]) #center location of image patch
#locquery = np.array([469, 210]) #center location of image patch
#locquery = np.array([np.random.randint(1024-psize)+psize/2, np.random.randint(436-psize)+psize/2])
print locquery
do_log = 0 # Take log of intensity values
do_zeromean = 1 # Zero-mean for patches
do_unitnorm = 1 # Unit-normalize patches
beta = 1

use_input = 1 # 0: intensity, 1: rgb, 2: first layer VGG16-places, 3: use NCC-optimized representation

ptmask = func_get_pat_cosmask(psize) # Circular mask

# Load Images
imgpath = '/home/kroegert/local/Datasets/Sintel-Stereo/training/final_right/'
#imgaf = 'frame_0001.png'
#imgbf = 'frame_0002.png'
imgaf = 'ambush_2/frame_0001.png'
imgbf = 'ambush_2/frame_0002.png'
imga = plt.imread(imgpath+imgaf).astype(float)
imgb = plt.imread(imgpath+imgbf).astype(float)

# extract patches from both images
if (use_input==0): # Gray scale
  imga = np.mean(imga, axis=2)
  imgb = np.mean(imgb, axis=2)
  pat_t  = func_extract_bil_patch(locquery, imga[:,:,None], psize, use_mask=ptmask, do_log=do_log, do_zeromean=do_zeromean, do_unitnorm=do_unitnorm, do_flatten=0)  # target patch, with circular window
  pat_q = func_extract_bil_patch(locquery, imgb[:,:,None], psize, use_mask=ptmask,   do_log=do_log, do_zeromean=do_zeromean, do_unitnorm=do_unitnorm, do_flatten=0)   # query patch
if (use_input==1): # RGB scale
  pat_t  = func_extract_bil_patch(locquery, imga[:,:,:], psize, use_mask=ptmask, do_log=do_log, do_zeromean=do_zeromean, do_unitnorm=do_unitnorm, do_flatten=0)  # target patch, with circular window
  pat_q = func_extract_bil_patch(locquery, imgb[:,:,:], psize, use_mask=ptmask,   do_log=do_log, do_zeromean=do_zeromean, do_unitnorm=do_unitnorm, do_flatten=0)   # query patch
if (use_input==2): # first layer VGG160-places
  imga_trafo = func_img_channel_read ('/scratch_net/zinc/kroegert_depot/Datasets/Sintel_trafo/VGG16_L1/final/' + imgaf[:-4] + '.dat')
  imgb_trafo = func_img_channel_read ('/scratch_net/zinc/kroegert_depot/Datasets/Sintel_trafo/VGG16_L1/final/' + imgbf[:-4] + '.dat')
  pat_t = func_extract_bil_patch(locquery, imga_trafo[:,:,:], psize, use_mask=ptmask, do_log=do_log, do_zeromean=do_zeromean, do_unitnorm=do_unitnorm, do_flatten=0)  # target patch, with circular window
  pat_q = func_extract_bil_patch(locquery, imgb_trafo[:,:,:], psize, use_mask=ptmask, do_log=do_log, do_zeromean=do_zeromean, do_unitnorm=do_unitnorm, do_flatten=0)   # query patch
if (use_input==3):
  imga_trafo = func_img_channel_read ('/scratch_net/zinc/kroegert_depot/Datasets/Sintel_trafo/NCC_L1/final/' + imgaf[:-4] + '.dat')
  imgb_trafo = func_img_channel_read ('/scratch_net/zinc/kroegert_depot/Datasets/Sintel_trafo/NCC_L1/final/' + imgbf[:-4] + '.dat')
  pat_t = func_extract_bil_patch(locquery, imga_trafo[:,:,:], psize, use_mask=ptmask, do_log=do_log, do_zeromean=do_zeromean, do_unitnorm=do_unitnorm, do_flatten=0)  # target patch, with circular window
  pat_q = func_extract_bil_patch(locquery, imgb_trafo[:,:,:], psize, use_mask=ptmask, do_log=do_log, do_zeromean=do_zeromean, do_unitnorm=do_unitnorm, do_flatten=0)   # query patch

# FFT of target and query (reverse template filters here?)
pat_t_fft = np.stack([np.fft.fft2(pat_t[:,:,x], [psize, psize]) for x in xrange(pat_t.shape[2])], axis=2)
pat_q_fft = np.stack([np.fft.fft2(pat_q[:,:,x], [psize, psize]) for x in xrange(pat_q.shape[2])], axis=2)

# NAIVE, NCC between target and query patch, 
#res1 = np.fft.ifft2(pat_q_fft * np.conj(pat_t_fft))
res1 = [np.fft.ifft2(pat_q_fft[:,:,x] * np.conj(pat_t_fft[:,:,x])) for x in xrange(pat_q.shape[2])]
res1 = np.maximum(0,np.real(np.fft.fftshift(res1)))
#res1 = [np.maximum(0,np.real(np.fft.fftshift(res1))) for x in xrange(pat_q.shape[2])]
res1 = np.mean(res1, axis=0)
#res1 = np.max(res1, axis=0)



# Visualize patches
vvmin = np.minimum(np.min(pat_t), np.min(pat_q))
vvmax = np.maximum(np.max(pat_t), np.max(pat_q))
fig = plt.figure()
plt.ion()
fig.add_subplot(2,2,1)
plt.imshow((imga*255.0).astype(np.uint8), interpolation='nearest', cmap='Greys_r')
plt.title('Image1, target keypoint')
plt.hold(True)
plt.scatter(locquery[0], locquery[1], 25, 'r')
fig.add_subplot(2,2,2)
plt.imshow((imgb*255.0).astype(np.uint8), interpolation='nearest', cmap='Greys_r')
plt.title('Image2, keypoint at same location as Image1')
plt.scatter(locquery[0], locquery[1], 25, 'b')
fig.add_subplot(2,2,3)
plt.imshow(np.mean(pat_t, axis=2), vmin = vvmin, vmax = vvmax, interpolation='nearest')
#plt.imshow(ptmask, interpolation='nearest')
plt.title('Patch from Image1')
fig.add_subplot(2,2,4)
plt.imshow(np.mean(pat_q, axis=2), vmin = vvmin, vmax = vvmax, interpolation='nearest')
plt.title('Patch from Image2')
plt.show()


fig = plt.figure()
plt.imshow(res1, vmin=0, vmax=1, interpolation='nearest')
plt.title('NAIVE correlation between patch1 and patch2')
plt.show()



# MOSSE-filter (cvpr 2010). Learn MOSSE fiter 'hfilt' based on 2D gaussian filter 'gfilt'
def getMOSSEfiter(pat_t_fft, psize, gsigma):
    gfilt = gauss2Dfilter(shape=(psize,psize),sigma=gsigma) # adapt this to cornerness/discriminativness of point in first image
    #gfilt /= np.max(gfilt)
    gfilt_fft = np.fft.fft2(gfilt, [psize, psize])
    pat_t_fft_c = np.conj(pat_t_fft) # complex conjugate 
    hfilt_fft = np.stack([np.divide(gfilt_fft * pat_t_fft_c[:,:,x], pat_t_fft[:,:,x] * pat_t_fft_c[:,:,x] + beta) for x in xrange(pat_t_fft.shape[2])], axis=2)
    return hfilt_fft, gfilt


fig = plt.figure()
sigrange = [4,8,16,32,64]
for sigma, i in zip(sigrange, xrange(len(sigrange))):
    
    hfilt_fft, gfilt = getMOSSEfiter(pat_t_fft, psize, float(psize)/sigma)
    #res = np.maximum(0,np.real(np.fft.ifft2(pat_q_fft * hfilt_fft)))
    res = np.mean(np.stack([np.maximum(0,np.real(np.fft.ifft2(pat_q_fft[:,:,x] * hfilt_fft[:,:,x]))) for x in xrange(pat_t_fft.shape[2])], axis=2), axis=2)
    fig.add_subplot(3,len(sigrange),i+1)
    plt.imshow(gfilt, interpolation='nearest')
    plt.title('Gauss.activation, sig={}'.format(psize/sigma))   
    fig.add_subplot(3,len(sigrange),i+1+len(sigrange))
    plt.imshow(np.abs(np.fft.ifft2(hfilt_fft[:,:,0])), interpolation='nearest')
    plt.title('Filter H')    
    fig.add_subplot(3,len(sigrange),i+1+2*len(sigrange))
    plt.imshow(res, interpolation='nearest') #vmin = 0, vmax = 1, 
    plt.title('Filter resonse')    

plt.show()
raw_input("Press Enter to continue...")





