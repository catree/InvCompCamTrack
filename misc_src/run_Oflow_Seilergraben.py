__author__ = 'kroegert'

import sys
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import cv2
import os
from glob import glob
import sys
import numpy as np

def viz_flow(u,v,logscale=True,scaledown=6,output=False):
    """
    topleft is zero, u is horiz, v is vertical
    red is 3 o'clock, yellow is 6, light blue is 9, blue/purple is 12
    """
    colorwheel = makecolorwheel()
    ncols = colorwheel.shape[0]

    radius = np.sqrt(u**2 + v**2)
    if output:
        print("Maximum flow magnitude: %04f" % np.max(radius))
    if logscale:
        radius = np.log(radius + 1)
        if output:
            print("Maximum flow magnitude (after log): %0.4f" % np.max(radius))
    radius = radius / scaledown    
    if output:
        print("Maximum flow magnitude (after scaledown): %0.4f" % np.max(radius))
    rot = np.arctan2(-v, -u) / np.pi

    fk = (rot+1)/2 * (ncols-1)  # -1~1 maped to 0~ncols
    k0 = fk.astype(np.uint8)       # 0, 1, 2, ..., ncols

    k1 = k0+1
    k1[k1 == ncols] = 0

    f = fk - k0

    ncolors = colorwheel.shape[1]
    img = np.zeros(u.shape+(ncolors,))
    for i in range(ncolors):
        tmp = colorwheel[:,i]
        col0 = tmp[k0]
        col1 = tmp[k1]
        col = (1-f)*col0 + f*col1
       
        idx = radius <= 1
        # increase saturation with radius
        col[idx] = 1 - radius[idx]*(1-col[idx])
        # out of range    
        col[~idx] *= 0.75
        img[:,:,i] = np.floor(255*col).astype(np.uint8)
    
    return img.astype(np.uint8)
        
def makecolorwheel():
    # Create a colorwheel for visualization
    RY = 15
    YG = 6
    GC = 4
    CB = 11
    BM = 13
    MR = 6
    
    ncols = RY + YG + GC + CB + BM + MR
    
    colorwheel = np.zeros((ncols,3))
    
    col = 0
    # RY
    colorwheel[0:RY,0] = 1
    colorwheel[0:RY,1] = np.arange(0,1,1./RY)
    col += RY
    
    # YG
    colorwheel[col:col+YG,0] = np.arange(1,0,-1./YG)
    colorwheel[col:col+YG,1] = 1
    col += YG
    
    # GC
    colorwheel[col:col+GC,1] = 1
    colorwheel[col:col+GC,2] = np.arange(0,1,1./GC)
    col += GC
    
    # CB
    colorwheel[col:col+CB,1] = np.arange(1,0,-1./CB)
    colorwheel[col:col+CB,2] = 1
    col += CB
    
    # BM
    colorwheel[col:col+BM,2] = 1
    colorwheel[col:col+BM,0] = np.arange(0,1,1./BM)
    col += BM
    
    # MR
    colorwheel[col:col+MR,2] = np.arange(1,0,-1./MR)
    colorwheel[col:col+MR,0] = 1

    return colorwheel    

def func_read_flo_file(name, ch=2):
    # WARNING: this will work on little-endian architectures (eg Intel x86) only!
    f = open(name, 'rb')
    magic = np.fromfile(f, np.float32, count=1)
    print magic
    data2D = []
    if 202021.25 != magic:
        print 'Head incorrect. Invalid .flo file'
    else:
        w = np.fromfile(f, np.int32, count=1)
        h = np.fromfile(f, np.int32, count=1)
        print 'Reading %d x %d flo file' % (w, h)
        data = np.fromfile(f, np.float32, count=ch*w*h)
        # Reshape data into 3D array (columns, rows, bands)
        data2D = np.resize(data, (h, w, ch))
    f.close()

    return data2D
  
def func_read_pfm_file(name):
    # WARNING: this will work on little-endian architectures (eg Intel x86) only!
    f = open(name, 'rb')
    magic = f.readline()
    #magic = np.fromfile(f, np.uint8, count=3)
    #print magic
    data2D = []
    if magic=="Pf\n": #( np.array_equal(magic,np.array([ 80, 102,  10], dtype=np.uint8))):
        wh = f.readline()
        wh = wh[:-1].split()
        w  = int(wh[0])
        h  = int(wh[1])
        sc = f.readline()
        sc = float(sc[:-1])
        print 'Reading %d x %d pfm file with scaling %d' % (w, h, sc)
        data = np.fromfile(f, np.float32, count=w*h)
        # Reshape data into 3D array (columns, rows, bands)
        data2D = np.resize(data, (h, w))
        data2D = data2D[::-1,:]
    else:
        print 'Head incorrect. Invalid .pfm file'

    f.close()

    return data2D
  


datapath_l = '/scratch_net/biwisrv01_second/varcity/datasets/GeoAutomation_Zurich/geoto/eth/images_jpg/zurich/1/2013.11.13/13/00/' # left images
datapath_r = '/scratch_net/biwisrv01_second/varcity/datasets/GeoAutomation_Zurich/geoto/eth/images_jpg/zurich/1/2013.11.13/13/01/' # right images

# Save stuff here:
savepath = '/scratch_net/biwisrv01_second/varcity/user/kroegert/Seilergraben_Final/'

startno = 113008000 # first image
noimages = 1400 # number of images, consecutively numbered

# Flow parameters
paramstr = '5 2 16 16 0.05 0.95 0 8 0.60 0 1 0 1 16 13 4.5 2 3 1.6 2'

for i in xrange(startno, startno+noimages):
  
  imga = plt.imread(datapath_l + 'image.' + str(i  ) + '.jpg').astype(float)
  imgb = plt.imread(datapath_l + 'image.' + str(i+1) + '.jpg').astype(float)
  plt.imsave('/tmp/imga.jpg', imga[5:-5,5:-5,:]) # save images, remove black borders
  plt.imsave('/tmp/imgb.jpg', imgb[5:-5,5:-5,:])

  # Optical Flow, Log scaling of magnitude
  os.system("rm /tmp/outfile.flo; /home/kroegert/local/Code/OpticalFlow/Github/OF_DIS/build/run_OF_RGB /tmp/imga.jpg /tmp/imgb.jpg /tmp/outfile.flo " + paramstr)
  flowout = func_read_flo_file("/tmp/outfile.flo").astype(float)
  x0_plot = viz_flow(flowout[:,:,0], flowout[:,:,1])
  plt.imsave(savepath + 'Flow/' + '{:04d}'.format(i-startno+1) + '.jpg', x0_plot)
  # Comment, encode as video with: avconv  -f image2 -i %04d.jpg -r 12  -b:v 1000k test.mp4

  ## Depth from Stereo  
  #imgb = plt.imread(datapath_r + 'image.' + str(i) + '.jpg').astype(float)
  #plt.imsave('/tmp/imgb.jpg', imgb[5:-5,5:-5,:])
  #os.system("rm /tmp/outfile.flo; /home/kroegert/local/Code/OpticalFlow/Github/OF_DIS/build/run_DE_RGB /tmp/imga.jpg /tmp/imgb.jpg /tmp/outfile.flo " + paramstr)
  #flowf = func_read_pfm_file("/tmp/outfile.flo").astype(float)
  #plt.imsave(savepath + 'Depth/' + 'image.' + str(i  ) + '.jpg', flowf, cmap='gray')



