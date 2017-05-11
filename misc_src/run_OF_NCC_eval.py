__author__ = 'kroegert'


import sys
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import scipy.io as scio
from scipy import ndimage
import math
import cv2
import time
import random
from func_viz_flow import viz_flow
from func_util_geom import func_crossMatrix, func_plot_cameras, func_set_axes_equal, func_reproject, func_F_transfer_points
from func_OF_util import func_read_flo_file, func_read_pfm_file, func_extract_bil_patch, func_get_pat_cosmask, gauss2Dfilter, func_eval_flowgt
import os


sys.path.append("../DL_Architectures_Tricks")
from func_util import func_get_filelist

#mmpp = '/home/till/zinc/'
mmpp = '/home/kroegert/'

maingtpath = mmpp + 'local/Datasets/Sintel/training_misc/flow/'

filelist_train = func_get_filelist(maingtpath, '', '*.flo');
nofiles = len(filelist_train)
random.seed(23) # fix randomization seed
filelist_train = random.sample(filelist_train, 2)

res_DIS = []
res_NCC = []
res_MOS = []

for i in xrange(len(filelist_train)):
  imgflowgt = maingtpath + filelist_train[i]
  imgflowGT = func_read_flo_file(imgflowgt)
  xGT_plot = viz_flow(imgflowGT[:,:,0], imgflowGT[:,:,1])
  
  
  imgaf = mmpp + 'local/Datasets/Sintel-Stereo/training/final_left/' + filelist_train[i][:-4] + '.png'
  imgbf = mmpp + 'local/Datasets/Sintel-Stereo/training/final_left/' + filelist_train[i][:-8] + '{:04}'.format((int(filelist_train[i][-8:-4])+1)) + '.png'
  
  def func_write_gradmag_img(imf, fout):
    im_dy = ndimage.sobel(imf, 0)
    im_dx = ndimage.sobel(imf, 1)
    mmin = np.minimum(np.min(im_dy), np.min(im_dx))
    im_dy -= mmin
    im_dx -= mmin
    mmax = np.maximum(np.max(im_dy), np.max(im_dx))
    im_dy = (im_dy / mmax)*255.0
    im_dx = (im_dx / mmax)*255.0
    im_dmag = np.sqrt(im_dx**2 + im_dy**2)
    im_dmag -= np.min(im_dmag)
    im_dmag = (im_dmag / np.max(im_dmag) )*255
    im_d = np.stack((im_dmag, im_dx, im_dy), axis=2)
    im_d = im_d.astype(np.uint8)
    
    plt.imsave(fout, im_d)

  #imf = plt.imread(imgaf).astype(float)
  #imgaf = '/tmp/imga.png'
  #imf = np.sqrt(np.sum(imf**2, axis=2))
  #func_write_gradmag_img(imf, imgaf)
  #imf = plt.imread(imgbf).astype(float)
  #imgbf = '/tmp/imgb.png'
  #imf = np.sqrt(np.sum(imf**2, axis=2))
  #func_write_gradmag_img(imf, imgbf)
  
  
  
  #imgaf = '/home/till/zinc/local/Datasets/Sintel-Stereo/training/final_left/market_2/frame_0003.png'
  #imgbf = '/home/till/zinc/local/Datasets/Sintel-Stereo/training/final_left/market_2/frame_0004.png'
  #imgflowgt = '/home/till/zinc/local/Datasets/Sintel/training_misc/flow/market_2/frame_0003.flo'

  #imgaf = '/home/till/zinc/local/Datasets/Sintel-Stereo/training/final_left/bamboo_2/frame_0003.png'
  #imgbf = '/home/till/zinc/local/Datasets/Sintel-Stereo/training/final_left/bamboo_2/frame_0004.png'
  #imgflowgt = '/home/till/zinc/local/Datasets/Sintel/training_misc/flow/bamboo_2/frame_0003.flo'

  #imgaf = '/home/till/zinc/local/Datasets/Sintel-Stereo/training/final_left/market_6/frame_0002.png'
  #imgbf = '/home/till/zinc/local/Datasets/Sintel-Stereo/training/final_left/market_6/frame_0003.png'
  #imgflowgt = '/home/till/zinc/local/Datasets/Sintel/training_misc/flow/market_6/frame_0002.flo'

  #imgaf = '/home/till/Downloads/frame_0018_0_0.png'
  #imgbf = '/home/till/Downloads/frame_0018_-9_-7.png'

paramstr = '5 1 16 12 0.05 0.95 0 8 0.50 0 1 0 1 16 13 4.5 2 3 1.6 2'
os.system("rm /tmp/outfileDIS.flo;" + mmpp + "local/Code/DIS_Flow_Github/OF_DIS/build/run_OF_RGB /scratch_net/biwisrv01_second/varcity/user/andrasb/4till/plane_warp_flow/im1.png /scratch_net/biwisrv01_second/varcity/user/andrasb/4till/plane_warp_flow/im2.png /tmp/outfileDIS.flo " + paramstr)
flowout = func_read_flo_file("/tmp/outfileDIS.flo").astype(float)
x0_DIS = viz_flow(flowout[:,:,0], flowout[:,:,1])
plt.figure();
plt.imshow(x0_DIS)
plt.title('DIS')
plt.show()

plt.imshow(np.sqrt(np.sum(flowout**2, axis=2)))
plt.show()

##res_DIS.append(func_eval_flowgt(imgflowGT, flowout))
##print func_eval_flowgt(imgflowGT, flowout)


  

  paramstr = '5 3 16 12 0.05 0.95 0 8 0.30 0 1 0 1 16 13 4.5 1 3 1.6 2'
  os.system("rm /tmp/outfileDIS.flo;" + mmpp + "local/Code/DIS_Flow_Github/OF_DIS/build/run_OF_INT " + imgaf + " " + imgbf + " /tmp/outfileDIS.flo " + paramstr)
  flowout = func_read_flo_file("/tmp/outfileDIS.flo").astype(float)
  x0_DIS = viz_flow(flowout[:,:,0], flowout[:,:,1])
  res_DIS.append(func_eval_flowgt(imgflowGT, flowout))
  #print func_eval_flowgt(imgflowGT, flowout)

  #scale first, scale last, iterations, relative convergence threshold, absolute convergence threshold, patch size, patch overlap
  #zero-mean, log input, circular masking, multi-channel mean/max, averaging, NCC/learned filter, beta regularizer, target filter parameter (sigma activation function)
  #use variational, 3x var weights, outer iterations, inner iterations, SOR value, sel_verbosity
  paramstr = '5 3 0 0 1.1 0.8 12 0.5 1 0 1 0 0.9 0 0.5 1.0 1 16 13 4.5 1 3 1.6 2'
  os.system("rm /tmp/outfileNCC.flo;" + mmpp + "local/Code/DIS_Flow_Github/OF_NCC/build/run_OF_INT " + imgaf + " " + imgbf + " /tmp/outfileNCC.flo " + paramstr)
  flowout = func_read_flo_file("/tmp/outfileNCC.flo").astype(float)
  x0_plot = viz_flow(flowout[:,:,0], flowout[:,:,1])
  res_NCC.append(func_eval_flowgt(imgflowGT, flowout))
  #print func_eval_flowgt(imgflowGT, flowout)
  
  paramstr = '5 3 0 0 1.1 0.8 12 0.5 1 0 1 0 0.9 1 0.5 1.0 1 16 13 4.5 1 3 1.6 2'
  os.system("rm /tmp/outfileNCC.flo;" + mmpp + "local/Code/DIS_Flow_Github/OF_NCC/build/run_OF_INT " + imgaf + " " + imgbf + " /tmp/outfileNCC.flo " + paramstr)
  flowout = func_read_flo_file("/tmp/outfileNCC.flo").astype(float)
  x0_plot = viz_flow(flowout[:,:,0], flowout[:,:,1])
  #res_MOS.append(func_eval_flowgt(imgflowGT, flowout))
  print func_eval_flowgt(imgflowGT, flowout)






                                      #paramstr = '4 4 16 12 0.05 0.95 0 8 0.30 0 1 0 1 16 13 4.5 1 3 1.6 2'
                                      #os.system("rm /tmp/outfileDIS.flo;" + mmpp + "local/Code/DIS_Flow_Github/OF_DIS/build/run_OF_RGB " + imgaf + " " + imgbf + " /tmp/outfileDIS.flo " + paramstr)
                                      #flowout = func_read_flo_file("/tmp/outfileDIS.flo").astype(float)
                                      #x0_DIS = viz_flow(flowout[:,:,0], flowout[:,:,1])
                                      #res_DIS.append(func_eval_flowgt(imgflowGT, flowout))
                                      #epedis = func_eval_flowgt(imgflowGT, flowout)[0]

                                      ##scale first, scale last, iterations, relative convergence threshold, absolute convergence threshold, patch size, patch overlap
                                      ##zero-mean, log input, circular masking, multi-channel mean/max, averaging, NCC/learned filter, beta regularizer, target filter parameter (sigma activation function)
                                      ##use variational, 3x var weights, outer iterations, inner iterations, SOR value, sel_verbosity
                                      #paramstr = '4 4 0 0 1.1 0.8 12 0.5 1 0 1 0 1.0 1 0.5 0.5 1 16 13 4.5 1 3 1.6 2'
                                      #os.system("rm /tmp/outfileNCC.flo;" + mmpp + "local/Code/DIS_Flow_Github/OF_NCC/build/run_OF_RGB " + imgaf + " " + imgbf + " /tmp/outfileNCC.flo " + paramstr)
                                      #flowout = func_read_flo_file("/tmp/outfileNCC.flo").astype(float)
                                      #xRGB_plot = viz_flow(flowout[:,:,0], flowout[:,:,1])
                                      #res_NCC.append(func_eval_flowgt(imgflowGT, flowout))
                                      #epeRGBcorr = func_eval_flowgt(imgflowGT, flowout)[0]

                                      #imgafdat = '/scratch_net/zinc/kroegert_depot/Datasets/Sintel_trafo/VGG16_L1/final' + imgaf[63:-3] + 'dat'
                                      #imgbfdat = '/scratch_net/zinc/kroegert_depot/Datasets/Sintel_trafo/VGG16_L1/final' + imgbf[63:-3] + 'dat'

                                      #os.system("rm /tmp/outfileNCC.flo;" + mmpp + "local/Code/DIS_Flow_Github/OF_NCC/build/run_OF_VAR67 " + imgafdat + " " + imgbfdat + " /tmp/outfileNCC.flo " + paramstr)
                                      #flowout = func_read_flo_file("/tmp/outfileNCC.flo").astype(float)
                                      #xCNN_plot = viz_flow(flowout[:,:,0], flowout[:,:,1])
                                      ##res_MOS.append(func_eval_flowgt(imgflowGT, flowout))
                                      #epeCNNcorr = func_eval_flowgt(imgflowGT, flowout)[0]


                                      
                                      #plt.figure();
                                      #plt.subplot(2,2,1)
                                      #plt.imshow(x0_DIS)
                                      #plt.title('DIS {}'.format(epedis))
                                      #plt.subplot(2,2,2)
                                      #plt.imshow(xRGB_plot)
                                      #plt.title('CorrRGB {}'.format(epeRGBcorr))
                                      #plt.subplot(2,2,3)
                                      #plt.imshow(xCNN_plot)
                                      #plt.title('CorrCNN {}'.format(epeCNNcorr))
                                      #plt.subplot(2,2,4)
                                      #plt.imshow(xGT_plot)
                                      #plt.title('GT')
                                      #plt.show()
  
  
  
  
  
  
  print res_DIS[-1]
  print res_NCC[-1]
  print res_MOS[-1]


print np.nanmean(np.stack(res_DIS, axis=0), axis=0)
print np.nanmean(np.stack(res_NCC, axis=0), axis=0)
print np.nanmean(np.stack(res_MOS, axis=0), axis=0)
print np.nanmean(np.stack(res_CNNNCC, axis=0), axis=0)

# RGB
#[  7.15597468   4.10496051   9.57151551  38.44436175]
#[  7.97637035   3.89890174  10.09861511  41.78316583]
#[  7.63699263   4.13127863  10.22798499  39.51631032]
# GRAD.MAG, DX, DY 
#[  9.6496172    5.13524866  10.45499096  48.91830766]
#[  7.70261581   4.57075119  10.62688392  41.22718468] - NCC
#[  8.10575375   4.89216309  11.60735488  40.7056866 ] - MOS
# CNN Imag.transform test
#[  9.01128332   5.19780955  10.18098727  41.65197115]
#[  8.67964464   4.19708823  10.85376427  44.22736657]
#[  8.47043828   4.31039542  10.5691848   42.27274317]


[  5.1537278    2.51065045  10.50325105  30.80849581]
[  5.45359837   2.55735167   9.86102527  32.33511113]
[  5.20491307   2.52805867   9.59535774  30.59281656]



# TODO: Dense aggregation
# TODO: Evaluation


plt.ion();
plt.figure();
plt.imshow(x0_DIS)
plt.title('DIS')
plt.show()

plt.ion();
plt.figure();
plt.imshow(x0_plot)
plt.title('Corr')
plt.show()


#raw_input('press key')








