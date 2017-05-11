__author__ = 'kroegert'


import sys
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import cv2
import os
from glob import glob
import sys
import numpy as np
sys.path.append("../DL_Architectures_Tricks")

from func_viz_flow import viz_flow



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
  
  

datapath = '/scratch_net/zinc/kroegert_depot/Datasets/OFFossati/V2/'
savepath = '/scratch_net/zinc/kroegert_share/flowout/V2/'

filelist = [y for x in os.walk(datapath) for y in glob(os.path.join(x[0], '*.png'))]
filelist.sort(key=lambda x:(not x.islower(), x))

for i in xrange(len(filelist)-1):
  paramstr = '5 3 16 16 0.05 0.95 0 8 0.60 0 1 0 1 16 13 4.5 2 3 1.6 2'
  os.system("rm /tmp/outfile.flo; /home/kroegert/local/Code/OpticalFlow/Github/OF_DIS/build/run_OF_RGB " + filelist[i] + " " + filelist[i+1] + " /tmp/outfile.flo " + paramstr)
  flowout = func_read_flo_file("/tmp/outfile.flo").astype(float)

  x0_plot = viz_flow(flowout[:,:,0], flowout[:,:,1])
  plt.imsave(savepath + 'img%04i.jpg' % i, x0_plot)



viz_flow

