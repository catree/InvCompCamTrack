__author__ = 'kroegert'


import sys
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import scipy.io as scio
import math
#import cv2
import time
import random
from func_viz_flow import viz_flow
from func_util_geom import func_crossMatrix, func_plot_cameras, func_set_axes_equal, func_reproject, func_F_transfer_points
import os

def func_eval_flowgt(imgflow, flowout):
  gtmag = np.sqrt(np.sum(imgflow**2, axis=2))

  errmag = np.sqrt(np.sum((imgflow - flowout)**2,axis=2))

  bin_0 = np.ones_like(gtmag).astype(np.bool)

  bin_s10 = (gtmag < 10)

  bin_s1040 = ((gtmag >= 10) & (gtmag < 40))

  bin_s40 = (gtmag >= 40)

  def errcnt(errmag, bins):
    cnt = np.sum(bins)
    idx = np.where(np.ndarray.flatten(bins))[0]
    return np.sum(np.ndarray.flatten(errmag)[idx]) / cnt

  return [errcnt(errmag, bin_0), errcnt(errmag, bin_s10), errcnt(errmag, bin_s1040), errcnt(errmag, bin_s40)  ]


# load images, compute corners and features
def func_read_flo_file(name, ch=2):
    # WARNING: this will work on little-endian architectures (eg Intel x86) only!
    f = open(name, 'rb')
    magic = np.fromfile(f, np.float32, count=1)
    #print magic
    data2D = []
    if 202021.25 != magic:
        print 'Head incorrect. Invalid .flo file'
    else:
        w = np.fromfile(f, np.int32, count=1)
        h = np.fromfile(f, np.int32, count=1)
        #print 'Reading %d x %d flo file' % (w, h)
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


def func_extract_bil_patch(ptin, img, pz, do_zeromean=0, use_mask=None, do_log=0, do_unitnorm=0, do_flatten=1):

    ptfloor = np.floor(ptin).astype(int)
    ptceil = ptfloor+1
    ptf = ptin - ptfloor
    w = np.array([ptf[0]*ptf[1], (1-ptf[0])*ptf[1], ptf[0]*(1-ptf[1]), (1-ptf[0])*(1-ptf[1])])

    pz2 = pz/2
    pa = img[(ptceil[1]-pz2):(ptceil[1]+pz2), (ptceil[0]-pz2):(ptceil[0]+pz2), :]
    pb = img[(ptceil[1]-pz2):(ptceil[1]+pz2), (ptfloor[0]-pz2):(ptfloor[0]+pz2), :]
    pc = img[(ptfloor[1]-pz2):(ptfloor[1]+pz2), (ptceil[0]-pz2):(ptceil[0]+pz2), :]
    pd = img[(ptfloor[1]-pz2):(ptfloor[1]+pz2), (ptfloor[0]-pz2):(ptfloor[0]+pz2), :]

    pf = pa*w[0] + pb*w[1] + pc*w[2] + pd*w[3]

    if (do_log==1):
        pf = np.log(np.minimum(255,np.maximum(0.1,pf)))

    if (do_zeromean==1):
        for i in xrange(pf.shape[2]):
            pf[:,:,i] -= np.mean(pf[:,:,i])

    if (use_mask is not None):
        for i in xrange(pf.shape[2]):
            pf[:,:,i] *= use_mask        
            #print pf[:,:,i].shape
            #print use_mask.shape
            
    if (do_unitnorm==1):
        for i in xrange(pf.shape[2]):
          pnorm = np.linalg.norm(pf[:,:,i])
          #if (np.sum(pf[:,:,i]) < 1e-10):
          #  pf[:,:,i] += 1e-15
          if (pnorm < 1e-15):
            pnorm = 1e-15
          pf[:,:,i] /= pnorm

    if (do_flatten==1):
        ptout = []
        for i in xrange(pf.shape[2]):
            ptout.append(np.ndarray.flatten(pf[:,:,i]))
        pf = np.stack(ptout, axis=1)
    return pf


def func_extract_NN_patch(ptin, img, pz, do_zeromean=0, use_mask=None, do_log=0, do_unitnorm=0, do_flatten=1):

    pz2 = pz/2
    pf = img[(ptin[1]-pz2):(ptin[1]+pz2), (ptin[0]-pz2):(ptin[0]+pz2), :]
    

    if (do_log==1):
        pf = np.log(np.minimum(255,np.maximum(0.1,pf)))

    if (do_zeromean==1):
        for i in xrange(pf.shape[2]):
            pf[:,:,i] -= np.mean(pf[:,:,i])

    if (use_mask is not None):
        for i in xrange(pf.shape[2]):
            pf[:,:,i] *= use_mask        
            #print pf[:,:,i].shape
            #print use_mask.shape
            
    if (do_unitnorm==1):
        for i in xrange(pf.shape[2]):
          pnorm = np.linalg.norm(pf[:,:,i])
          #if (np.sum(pf[:,:,i]) < 1e-10):
          #  pf[:,:,i] += 1e-15
          if (pnorm < 1e-15):
            pnorm = 1e-15
          pf[:,:,i] /= pnorm

    if (do_flatten==1):
        ptout = []
        for i in xrange(pf.shape[2]):
            ptout.append(np.ndarray.flatten(pf[:,:,i]))
        pf = np.stack(ptout, axis=1)
    return pf



def func_get_pat_cosmask(psize):
    maskwindow = np.zeros((psize,psize), dtype=float)
    cent = psize/2
    for xi in xrange(psize):
        for yi in xrange(psize):
            maskwindow[xi,yi] = np.minimum(1,np.sqrt(float((xi-cent+0.5)**2 + (yi-cent+0.5)**2) / (psize/2)**2))
    return np.cos(maskwindow*np.pi/2)

def gauss2Dfilter(shape=(3,3),sigma=0.5):
    m,n = [np.ceil((ss-1.)/2.) for ss in shape]
    y,x = np.ogrid[0:shape[0],0:shape[1]]
    y = y.astype(float) - float(m)
    x = x.astype(float) - float(n)
    h = np.exp( -(x*x + y*y) / (2.*sigma*sigma) )
    h[ h < np.finfo(h.dtype).eps*h.max() ] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h
