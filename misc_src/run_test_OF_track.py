__author__ = 'kroegert'


import sys
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import scipy.io as scio
import math
import cv2
import time
import random
from func_viz_flow import viz_flow
from func_util_geom import func_crossMatrix, func_plot_cameras, func_set_axes_equal, func_reproject, func_F_transfer_points
from func_OF_util import func_read_flo_file, func_read_pfm_file
import os


bsize = 14
pathl = '/scratch_net/zinc/kroegert_depot/Datasets/Kitti_MVFlow/training/image_0/'
pathr = '/scratch_net/zinc/kroegert_depot/Datasets/Kitti_MVFlow/training/image_1/'
#pathl = '/home/kroegert/local/Datasets/Sintel-Stereo/training/final_left/market_2/'
#pathr = '/home/kroegert/local/Datasets/Sintel-Stereo/training/final_right/market_2/'
movid = 4

file = open("/scratch_net/zinc/kroegert_depot/Datasets/Kitti_FlowCalib/training/calib/" + "%06i.txt" % (movid), 'r')
calibfile = file.readlines()
file.close()
P0 = np.reshape(np.fromstring(calibfile[0][4:-1], dtype=float, sep=" "), (3,4))
P1 = np.reshape(np.fromstring(calibfile[1][4:-1], dtype=float, sep=" "), (3,4))
K01 = P0[0:3,0:3]
E01 = func_crossMatrix(P1[0:3,3])



def func_get_transf_position(xy, disp_u, disp_v=np.zeros((0,0))):
    xy_floor = np.floor(xy)
    xy_ceil = xy_floor+1
    xy_f =  xy.astype(float) - xy_floor.astype(float)
    xy_floor = xy_floor.astype(int)
    xy_ceil = xy_ceil.astype(int)

    xy_res = np.ones_like(xy).astype(float)
    xy_res[:] = np.NaN
    idxvalid = ~((xy_floor[:,0] < 0) | (xy_ceil[:,0] < 0) | (xy_floor[:,1] < 0) | (xy_ceil[:,1] < 0) | (xy_floor[:,0] >= disp_u.shape[1]) | (xy_ceil[:,0] >= disp_u.shape[1]) | (xy_floor[:,1] >= disp_u.shape[0]) | (xy_ceil[:,1] >= disp_u.shape[0]))
    idxvalid = np.where(idxvalid)[0]
    xy_res[idxvalid,:] = xy[idxvalid,:]

    w = np.array([xy_f[:,0]*xy_f[:,1], (1-xy_f[:,0])*xy_f[:,1], xy_f[:,0]*(1-xy_f[:,1]), (1-xy_f[:,0])*(1-xy_f[:,1])])

    du = disp_u[xy_ceil [idxvalid,1],xy_ceil [idxvalid,0]] * w[0,idxvalid] + \
         disp_u[xy_ceil [idxvalid,1],xy_floor[idxvalid,0]] * w[1,idxvalid] + \
         disp_u[xy_floor[idxvalid,1],xy_ceil [idxvalid,0]] * w[2,idxvalid] + \
         disp_u[xy_floor[idxvalid,1],xy_floor[idxvalid,0]] * w[3,idxvalid]
    
    xy_res[idxvalid,0] += du
    
    if (disp_v.size>0):
        dv = disp_v[xy_ceil [idxvalid,1],xy_ceil [idxvalid,0]] * w[0,idxvalid] + \
             disp_v[xy_ceil [idxvalid,1],xy_floor[idxvalid,0]] * w[1,idxvalid] + \
             disp_v[xy_floor[idxvalid,1],xy_ceil [idxvalid,0]] * w[2,idxvalid] + \
             disp_v[xy_floor[idxvalid,1],xy_floor[idxvalid,0]] * w[3,idxvalid]
        
        xy_res[idxvalid,1] += dv
        
    return xy_res


imgs = [] # list of tuples of images
imgs_d = [] # list of left->right and right->back depth estimates
imgs_fl = [] # list of left  flow forward and backward
imgs_fr = [] # list of right flow forward and backward
for i in xrange(bsize):
    imgname1 = pathl + "%06i_%02i.png" % (movid,i)  # KITTI
    imgname2 = pathr + "%06i_%02i.png" % (movid,i)
    imgname1f = pathl + "%06i_%02i.png" % (movid,i+1)
    imgname2f = pathr + "%06i_%02i.png" % (movid,i+1)
    paramstr = '6 1 64 32 0.05 0.95 0 8 0.80 0 1 1 1 16 13 4.5 8 5 1.6 2 0'
    
    #imgname1 = pathl + "frame_%04i.png" % (i+1) # SINTEL
    #imgname2 = pathr + "frame_%04i.png" % (i+1)
    #imgname1f = pathl + "frame_%04i.png" % (i+2)
    #imgname2f = pathr + "frame_%04i.png" % (i+2)
    #paramstr = '5 1 64 32 0.05 0.95 0 8 0.80 0 1 1 1 16 13 4.5 8 5 1.6 2 0'
    
    img_tuple = (cv2.imread(imgname1).astype(float), cv2.imread(imgname2).astype(float))
    imgs.append(img_tuple)

    os.system("rm /tmp/outfile.flo; /home/kroegert/local/Code/OpticalFlow/build/run_DE_RGB " + imgname1 + " " + imgname2 + " /tmp/outfile.flo " + paramstr)
    flowf = func_read_pfm_file("/tmp/outfile.flo").astype(float)

    cv2.imwrite('/tmp/imgl.png',img_tuple[1][:,::-1])
    cv2.imwrite('/tmp/imgr.png',img_tuple[0][:,::-1])
    os.system("rm /tmp/outfile.flo; /home/kroegert/local/Code/OpticalFlow/build/run_DE_RGB /tmp/imgl.png /tmp/imgr.png /tmp/outfile.flo " + paramstr)
    flowb = func_read_pfm_file("/tmp/outfile.flo").astype(float)

    imgs_d.append((flowf, flowb[:,::-1]))

    os.system("rm /tmp/outfile.flo; /home/kroegert/local/Code/OpticalFlow/build/run_OF_RGB " + imgname1 + " " + imgname1f + " /tmp/outfile.flo " + paramstr)
    flowf = func_read_flo_file("/tmp/outfile.flo").astype(float)
    os.system("rm /tmp/outfile.flo; /home/kroegert/local/Code/OpticalFlow/build/run_OF_RGB " + imgname1f + " " + imgname1 + " /tmp/outfile.flo " + paramstr)
    flowb = func_read_flo_file("/tmp/outfile.flo").astype(float)
    imgs_fl.append((flowf, flowb))

    os.system("rm /tmp/outfile.flo; /home/kroegert/local/Code/OpticalFlow/build/run_OF_RGB " + imgname2 + " " + imgname2f + " /tmp/outfile.flo " + paramstr)
    flowf = func_read_flo_file("/tmp/outfile.flo").astype(float)
    os.system("rm /tmp/outfile.flo; /home/kroegert/local/Code/OpticalFlow/build/run_OF_RGB " + imgname2f + " " + imgname2 + " /tmp/outfile.flo " + paramstr)
    flowb = func_read_flo_file("/tmp/outfile.flo").astype(float)
    imgs_fr.append((flowf, flowb))

imheight = imgs[0][0].shape[0]
imwidth = imgs[0][0].shape[1]


# # disparity range is tuned for 'aloe' image pair
# window_size = 11
# min_disp = 0
# num_disp = 64-min_disp
# stereo = cv2.StereoSGBM_create(minDisparity = min_disp,
#     numDisparities = num_disp,
#     blockSize = window_size,
#     P1 = 600, #4*window_size**2,
#     P2 = 2400, #16*window_size**2,
#     speckleWindowSize = 3,
#     speckleRange = 3,
#     uniquenessRatio = 1,
#     disp12MaxDiff = 50,
#     mode = 1
# )
# 
# print('computing disparity...')
# disp = stereo.compute(img_tuple[0], img_tuple[1]).astype(np.float32)
# disp /= 16
# plt.imshow(disp)
# plt.show()

# fig = plt.figure()
# fig.add_subplot(2,1,1)
# plt.imshow(flowf)
# plt.hold(True)
# fig.add_subplot(2,1,2)
# plt.imshow(flowb)
# plt.show()


   
    # 
    # #imgname1 = "/home/kroegert/local/Datasets/Sintel-Stereo/training/final_left/market_2/frame_0004.png"
    # #imgname2 = "/home/kroegert/local/Datasets/Sintel-Stereo/training/final_left/market_2/frame_0005.png"
    # os.system("rm /tmp/outfile.flo; /home/kroegert/local/Code/OpticalFlow/build/run_OF_RGB " + (imgname1) + " " + (imgname2) + " /tmp/outfile.flo " + paramstr)
    # flow = func_read_flo_file("/tmp/outfile.flo").astype(float)
    # #flow = func_read_flo_file("/home/kroegert/local/Datasets/OF_Sintel/test/market_2/frame_0004.flo").astype(float)    
    # #fig = plt.figure()
    # #plt.imshow(viz_flow(flow[:,:,0], flow[:,:,1]))
    # #plt.show()
    # 
    # #os.system("rm /tmp/outfile.flo; /home/kroegert/local/Code/OpticalFlow/build/run_DE_RGB " + (pathl + imgname1) + " " + (pathl + imgname2) + " /tmp/outfile.flo " + paramstr)
    # #flow = func_read_pfm_file("/tmp/outfile.flo").astype(float)
    # 
    # imga = cv2.imread(imgname1)    
    # imgb = cv2.imread(imgname2)
    # 
    # 


threshold_flowvalid_ratio=.2 # max ratio: error vecor magnitude / displacement vector magnitude
threshold_flowvalid_abs=1 # max absolute error in pixels

xy = np.meshgrid(np.arange(imwidth), np.arange(imheight))
xy_l1 = np.vstack((np.ndarray.flatten(xy[0]), np.ndarray.flatten(xy[1]))).transpose()


xy_l = xy_l1 
######

xy_t = np.NaN*np.ones((xy_l.shape[0],4,bsize), dtype=float)

#initial left point transfer
xy_r = func_get_transf_position(xy_l, -imgs_d[0][0])

# left-right consistency check
xy_l_verif_d = func_get_transf_position(xy_r, imgs_d[0][1])

idxvalid=np.where((np.linalg.norm(xy_l-xy_l_verif_d, axis=1)/np.linalg.norm(xy_r-xy_l, axis=1) < threshold_flowvalid_ratio) & 
                  (np.linalg.norm(xy_l-xy_l_verif_d, axis=1)                                   < threshold_flowvalid_abs))[0]
xy_t[idxvalid,:,0] = np.hstack((xy_l, xy_r))[idxvalid,:]

for fr in xrange(1,bsize):
    xy_l = xy_t[:,0:2,fr-1]
    xy_r = xy_t[:,2:4,fr-1]
    
    
    
    xy_lf = func_get_transf_position(xy_l, imgs_fl[fr-1][0][:,:,0], imgs_fl[fr-1][0][:,:,1])
    xy_rf = func_get_transf_position(xy_r, imgs_fr[fr-1][0][:,:,0], imgs_fr[fr-1][0][:,:,1])

    #f2 left-right consistency check
    xy_rf_verif_d = func_get_transf_position(xy_lf, -imgs_d[fr][0])
    xy_lf_verif_d = func_get_transf_position(xy_rf,  imgs_d[fr][1])
     
    # L forward backward check
    xy_l_verif_f = func_get_transf_position(xy_lf, imgs_fl[fr-1][1][:,:,0], imgs_fl[fr-1][1][:,:,1])
    
    # R forward backward check
    xy_r_verif_f = func_get_transf_position(xy_rf, imgs_fr[fr-1][1][:,:,0], imgs_fr[fr-1][1][:,:,1])
     
    idxvalid = (np.linalg.norm(xy_rf-xy_rf_verif_d, axis=1)/np.linalg.norm(xy_rf-xy_lf, axis=1) < threshold_flowvalid_ratio) & \
               (np.linalg.norm(xy_lf-xy_lf_verif_d, axis=1)/np.linalg.norm(xy_rf-xy_lf, axis=1) < threshold_flowvalid_ratio) & \
               (np.linalg.norm(xy_l -xy_l_verif_f , axis=1)/np.linalg.norm(xy_l -xy_lf, axis=1) < threshold_flowvalid_ratio) & \
               (np.linalg.norm(xy_r -xy_r_verif_f , axis=1)/np.linalg.norm(xy_r -xy_rf, axis=1) < threshold_flowvalid_ratio) & \
               (np.linalg.norm(xy_rf-xy_rf_verif_d, axis=1) < threshold_flowvalid_abs) & \
               (np.linalg.norm(xy_lf-xy_lf_verif_d, axis=1) < threshold_flowvalid_abs) & \
               (np.linalg.norm(xy_l -xy_l_verif_f , axis=1) < threshold_flowvalid_abs) & \
               (np.linalg.norm(xy_r -xy_r_verif_f , axis=1) < threshold_flowvalid_abs)
    idxinvalid  = np.where(~idxvalid)[0]
    idxvalid = np.where(idxvalid)[0]
    
    xy_t[idxvalid ,: ,fr] = np.hstack((xy_lf, xy_rf))[idxvalid,:]
    xy_t[idxinvalid,:,: ] = np.NaN

idxvalid = np.where(~np.isnan(xy_t[:,0,0]))[0]
xy_t = xy_t[idxvalid,:,:]


# # left-right consistency check
# xy_l1_verif_d = func_get_transf_position(xy_r1, imgs_d[0][1])
# 
# idxinvalid=(np.linalg.norm(xy_l1-xy_l1_verif_d, axis=1) < threshold_flowvalid)
# idxinvalid=np.where(~idxinvalid)[0] 
# # 
# xy_l1 = np.delete(xy_l1, idxinvalid, 0)
# xy_r1 = np.delete(xy_r1, idxinvalid, 0)
# 
# 
# 
# 
# xy_l2 = func_get_transf_position(xy_l1, imgs_fl[0][0][:,:,0], imgs_fl[0][0][:,:,1])
# xy_r2 = func_get_transf_position(xy_r1, imgs_fr[0][0][:,:,0], imgs_fr[0][0][:,:,1])
#  
#  
#  
# #f2 left-right consistency check
# xy_r2_verif_d = func_get_transf_position(xy_l2, -imgs_d[1][0])
# xy_l2_verif_d = func_get_transf_position(xy_r2,  imgs_d[1][1])
#  
# # L forward backward check
# xy_l1_verif_f = func_get_transf_position(xy_l2, imgs_fl[0][1][:,:,0], imgs_fl[0][1][:,:,1])
# 
# # R forward backward check
# xy_r1_verif_f = func_get_transf_position(xy_r2, imgs_fr[0][1][:,:,0], imgs_fr[0][1][:,:,1])
#  
# idxinvalid=(np.linalg.norm(xy_l1-xy_l1_verif_d, axis=1) < threshold_flowvalid) & \
#            (np.linalg.norm(xy_r2-xy_r2_verif_d, axis=1) < threshold_flowvalid) & \
#            (np.linalg.norm(xy_l2-xy_l2_verif_d, axis=1) < threshold_flowvalid) & \
#            (np.linalg.norm(xy_l1-xy_l1_verif_f, axis=1) < threshold_flowvalid) & \
#            (np.linalg.norm(xy_r1-xy_r1_verif_f, axis=1) < threshold_flowvalid) 
# idxinvalid=np.where(~idxinvalid)[0] 
# # 
# xy_l1 = np.delete(xy_l1, idxinvalid, 0)
# xy_r1 = np.delete(xy_r1, idxinvalid, 0)
# xy_l2 = np.delete(xy_l2, idxinvalid, 0)
# xy_r2 = np.delete(xy_r2, idxinvalid, 0)


# fig = plt.figure()
# fig.add_subplot(2,2,1)
# plt.imshow(imgs[0][0].astype(np.uint8))
# plt.hold(True)
# plt.scatter(xy_t[0:-1:500,0,0], xy_t[0:-1:500,1,0], 5, color='red')
# fig.add_subplot(2,2,2)
# plt.imshow(imgs[0][1].astype(np.uint8))    
# plt.hold(True)
# plt.scatter(xy_t[0:-1:500,2,0], xy_t[0:-1:500,3,0], 5, color='red')
# fig.add_subplot(2,2,3)
# plt.imshow(imgs[1][0].astype(np.uint8))
# plt.hold(True)
# plt.scatter(xy_t[0:-1:500,0,1], xy_t[0:-1:500,1,1], 5, color='red')
# fig.add_subplot(2,2,4)
# plt.imshow(imgs[1][1].astype(np.uint8))
# plt.hold(True)
# plt.scatter(xy_t[0:-1:500,2,1], xy_t[0:-1:500,3,1], 5, color='red')
# plt.show()


subsampleno = 100
plt.ion()
for fr in xrange(bsize):
    fig = plt.figure(figsize=(16, 12))
    plt.imshow(np.vstack((imgs[fr][0], imgs[fr][1])).astype(np.uint8))
    plt.hold(True)
    plt.scatter(xy_t[0:-1:subsampleno,0,fr], xy_t[0:-1:subsampleno,1,fr], 5, color='red')
    plt.scatter(xy_t[0:-1:subsampleno,2,fr], xy_t[0:-1:subsampleno,3,fr]+imheight, 5, color='red')
    if (fr>0):
        a = np.array([xy_t[0:-1:subsampleno,0,fr], xy_t[0:-1:subsampleno,0,fr-1]])
        b = np.array([xy_t[0:-1:subsampleno,1,fr], xy_t[0:-1:subsampleno,1,fr-1]])
        plt.plot(a,b, '-r')
        a = np.array([xy_t[0:-1:subsampleno,2,fr], xy_t[0:-1:subsampleno,2,fr-1]])
        b = np.array([xy_t[0:-1:subsampleno,3,fr], xy_t[0:-1:subsampleno,3,fr-1]])
        plt.plot(a,b+imheight, '-r')
    plt.axis([0, imwidth, imheight*2, 0])
    plt.show()
    plt.savefig('/home/kroegert/local/Results/AccGPSFuse/imgseq%02i.png' % fr)
    
plt.close('all')



## Divide points in static and dynamic using the fundamental fundamental matrix
nopt = xy_t.shape[0]
thresh_F_inlier = 2.0 
no_ransac_samplernds = 100

# fit F over sequence left and right. Count how many inliers are present
best_inlno = 0
best_inliers = []
for rr in xrange(no_ransac_samplernds):
  # select minimum sampling set
  randi =  np.random.choice(xrange(nopt), size=8, replace=False)

  dd = np.zeros((nopt,bsize/2), dtype=float)
  for fr in xrange(bsize/2):
    fr_f = fr+np.ceil(bsize/2.0).astype(int) # compare fr0 to fr0+halfseqlength
    
    # LEFT compare fr0 to fr0+halfseqlength
    retval, mask = cv2.findFundamentalMat(xy_t[randi,0:2,fr], xy_t[randi,2:4,fr_f], method=cv2.FM_8POINT)
    dr = func_F_transfer_points(retval,  xy_t[:,0:2,fr], xy_t[:,2:4,fr_f])

    # RIGHT compare fr0 to fr0+halfseqlength
    retval, mask = cv2.findFundamentalMat(xy_t[randi,2:4,fr], xy_t[randi,0:2,fr_f], method=cv2.FM_8POINT)
    dl = func_F_transfer_points(retval,  xy_t[:,2:4,fr], xy_t[:,0:2,fr_f])

    dd[:,fr] = np.max(np.array([dr, dl]), axis=0)


    inliers = np.where(np.all(dd<thresh_F_inlier, axis=1))[0]
    inlno = inliers.shape[0]
    
    
  if (inlno > best_inlno):
    best_inlno = inlno
    best_inliers = inliers.copy()
    
    
##  Display test: show only static keypoints
fr=bsize-1
fig = plt.figure(figsize=(16, 12))
plt.imshow(np.vstack((imgs[fr][0], imgs[fr][1])).astype(np.uint8))
plt.hold(True)
plt.scatter(xy_t[best_inliers[0:-1:100],0,fr], xy_t[best_inliers[0:-1:100],1,fr], 5, color='red')
plt.scatter(xy_t[best_inliers[0:-1:100],2,fr], xy_t[best_inliers[0:-1:100],3,fr]+imheight, 5, color='red')
#idxbesttest=best_inliers[::-1][0:100]
#plt.scatter(xy_t[idxbesttest,0,fr], xy_t[idxbesttest,1,fr], 5, color='red')
#plt.scatter(xy_t[idxbesttest,2,fr], xy_t[idxbesttest,3,fr]+imheight, 5, color='red')
plt.axis([0, imwidth, imheight*2, 0])
plt.show()



## Triangulate points based on baseline in first frame
b = xy_t[:,2,0]-xy_t[:,0,0]
img_depth = P1[0,3] * K01[0,0] / b


K01inv = np.linalg.inv(K01)
ptnorm = (xy_t[:,0:2,0] - K01[0:2,2][None,:]) / np.array([K01[0,0], K01[1,1]])[None,:] # Normalized coordinates
ptnorm = np.concatenate((ptnorm, np.ones((ptnorm.shape[0],1), dtype=float)), axis=1) # points in positive Z direction
#ptnorm = np.dot(K01inv,ptnorm.transpose()).transpose()
#ptnorm /= np.linalg.norm(ptnorm, axis=1)[:,None] 
#ptnorm[:,2] = -ptnorm[:,2] # points are in negative Z direction
ptnorm *= img_depth[:,None] 


## Fit cameras
#Cameras in first frame
#R_l = np.identity(3,dtype=float)
#t_l = np.zeros((3,), dtype=float)
#R_r = np.identity(3,dtype=float)
t_r = P1[:,3]
wh = np.array([imwidth, imheight], dtype=float)
fc = np.array([K01[0,0], K01[1,1]])
cc = np.array([K01[0,2], K01[1,2]]) - wh / 2.0
kc = np.zeros((0,0), dtype=float)

CamRt_list = []
for fr in xrange(bsize): 
    retval, R_l_vec, t_l = cv2.solvePnP(ptnorm[best_inliers,:], xy_t[best_inliers,0:2,fr], K01, np.zeros((0,),dtype=float)) 
    R_l = cv2.Rodrigues(R_l_vec)[0]
    t_l = t_l[:,0]
    
    #retval, R_r_vec, t_r = cv2.solvePnP(ptnorm[best_inliers,:], xy_t[best_inliers,2:4,fr], K01, np.zeros((0,),dtype=float)) 
    #R_r = cv2.Rodrigues(R_r_vec)[0]
    #t_r = t_r[:,0]
    #print [np.degrees(np.arccos(np.sum(R_l[:,2]*R_r[:,2]))),     np.degrees(np.arccos(np.sum(R_l[:,1]*R_r[:,1])))]
    #print np.linalg.norm((t_l-P1[:,3]) -t_r)
    
    #R_l = cv2.Rodrigues(R_l_vec)[0].transpose()
    #t_l = -np.dot(R_l, t_l[:,0])
    CamRt_list.append((R_l, t_l))
    
    
# Reprojection test into left and right camera
for fr in xrange(bsize): 
    R = CamRt_list[fr][0]
    t = CamRt_list[fr][1]
    #pt, jec = cv2.projectPoints(ptnorm[best_inliers,:], R_l_vec, t_l, K01, kc) 
    xy_l_re = func_reproject(ptnorm[best_inliers,:], R, -np.dot(R.transpose(),t)        , fc, cc+wh/2.0, kc, camcenter=0)
    #xy_l_re = func_reproject(ptnorm[best_inliers,:], R, t        , fc, cc+wh/2.0, kc, camcenter=0)
    xy_r_re = func_reproject(ptnorm[best_inliers,:], R, -np.dot(R.transpose(),t+P1[:,3]), fc, cc+wh/2.0, kc, camcenter=0)
    print [np.mean(np.linalg.norm((xy_t[best_inliers,0:2,fr]) - xy_l_re, axis=1)),
           np.mean(np.linalg.norm((xy_t[best_inliers,2:4,fr]) - xy_r_re, axis=1))] # Uses known left-base baseline from P1
    #fig = plt.figure(figsize=(16, 12))
    #plt.imshow(np.vstack((imgs[fr][0], imgs[fr][1])).astype(np.uint8))
    #plt.hold(True)
    #plt.scatter(xy_l_re[0:-1:100,0], xy_l_re[0:-1:100,1], 5, color='red')
    #plt.scatter(xy_r_re[0:-1:100,0], xy_r_re[0:-1:100,1]+imheight, 5, color='red')
    #plt.scatter(xy_t[best_inliers[0:-1:100],0,fr], xy_t[best_inliers[0:-1:100],1,fr], 5, color='blue')
    #plt.scatter(xy_t[best_inliers[0:-1:100],2,fr], xy_t[best_inliers[0:-1:100],3,fr]+imheight, 5, color='blue')
    #plt.axis([0, imwidth, imheight*2, 0])
    #plt.show()



fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
for fr in xrange(bsize): 
    R = CamRt_list[fr][0]
    t = CamRt_list[fr][1]
    func_plot_cameras(ax, fc, cc,wh,R,-np.dot(R.transpose(), t        ),rgbface=np.array([1.0, 0.0, 0.0]),camerascaling=500.0,lw=2.0)
    #func_plot_cameras(ax, fc,cc,wh,R, t,rgbface=np.array([1.0, 0.0, 0.0]),camerascaling=100.0,lw=2.0)
    func_plot_cameras(ax, fc, cc,wh,R,-np.dot(R.transpose(), t+P1[:,3]),rgbface=np.array([0.0, 1.0, 0.0]),camerascaling=500.0,lw=2.0)
    #func_plot_cameras(ax, fc,cc,wh,R, t-P1[:,3],rgbface=np.array([1.0, 0.0, 0.0]),camerascaling=100.0,lw=2.0)
    #tc = -np.dot(R.transpose(), t        )
    #ax.scatter(tc[0], tc[1], tc[2], color=np.array([0.0, 1.0, 0.0]), linewidth=20.0)
  
ax.scatter(ptnorm[best_inliers[0:-1:subsampleno],0], ptnorm[best_inliers[0:-1:subsampleno],1], ptnorm[best_inliers[0:-1:subsampleno],2], color=np.array([0.0, 0.0, 1.0]), linewidth=2.0)
#ax.scatter(ptnorm[idxbesttest,0], ptnorm[idxbesttest,1], ptnorm[idxbesttest,2], color=np.array([0.0, 0.0, 1.0]), linewidth=2.0)
func_set_axes_equal(ax)
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
plt.show()
    





## TODO: 1) RANSAC camera estimation left for whole sequence, check right for consistency
# 2) Split in static and moving points
# 3) BA Adjust cameras and static points
