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
from func_util_geom import func_crossMatrix

def func_dist_kc(ttmp, kc):
    if (kc.shape[0] > 0):

        r2 = np.sum(ttmp[:,0:2]**2,axis=1)

        rc = 1.0 + kc[0]*r2

        if (kc.shape[0]>1):
            rc = rc + kc[1]*r2*r2

            if (kc.shape[0]==5):
                rc = rc + kc[4]*r2*r2*r2

            if (kc.shape[0]>=3):
                dx = np.array([(2*kc[2]*ttmp[:,0]*ttmp[:,1] + kc[3]*(r2 + 2*ttmp[:,0]**2)),
                               (2*kc[3]*ttmp[:,0]*ttmp[:,1] + kc[2]*(r2 + 2*ttmp[:,1]**2))]).transpose() # tangential distortion


        ptout = ttmp[:,0:2] * rc[:,None]

        if (kc.shape[0]>=3):
            ptout += dx

    else:
        ptout = ttmp[:,0:2]

    return ptout

def func_reproject(X, R, t, fc, cc, kc,camcenter=0):
    if (camcenter==0):
        ttmp = X-T_w
        ttmp = np.dot(R,ttmp.transpose()).transpose()
    else:
        ttmp = np.dot(R,X.transpose()).transpose()
        ttmp = ttmp-T_w

    ttmp[:,0] /= ttmp[:,2]
    ttmp[:,1] /= ttmp[:,2]

    ttmp = func_dist_kc(ttmp, kc)

    ttmp[:,0] = (ttmp[:,0] * fc[0]) + cc[0]
    ttmp[:,1] = (ttmp[:,1] * fc[1]) + cc[1]

    return ttmp

def func_get_mask_for_potential_matches_from_F(F, F_pts_a, F_pts_b, F_treshpx):
    lines_a_in_b = np.squeeze(cv2.computeCorrespondEpilines(F_pts_a, 1, F))
    lines_a_in_b = np.reshape(lines_a_in_b, (lines_a_in_b.shape[0],1,3))
    lines_a_in_b = np.repeat(lines_a_in_b, F_pts_b.shape[0], axis=1)

    mask = np.abs(lines_a_in_b[:,:,0] * F_pts_b[:,0] + lines_a_in_b[:,:,1] * F_pts_b[:,1] + lines_a_in_b[:,:,2])

    # lines_a_in_b = np.squeeze(cv2.computeCorrespondEpilines(pts1, 1, F))
    # lines_a_in_b[:,0]*pts2[:,0] + lines_a_in_b[:,1]*pts2[:,1] + lines_a_in_b[:,2]


    mask = (mask > F_treshpx)

    return mask

def func_get_first_second_neighbor_match(similcos, tr=1.1):
    idmax = np.argmax(similcos, axis=0)
    ptrange = xrange(similcos.shape[1])
    val = similcos[idmax, ptrange]
    comb = np.array([ptrange, idmax])

    similcos[idmax, ptrange] = -1
    idmax2 = np.argmax(similcos, axis=0)
    val2 = similcos[idmax2, ptrange]

    idxdel = np.where(val / val2 < tr)[0]

    comb = np.delete(comb, idxdel, axis=1)
    val = np.delete(val, idxdel, axis=0)

    return comb, val


def func_match_cossimil_2view(plist_a, plist_b, tr=1.1, abstr = 0.7, F=np.zeros((0,0)), F_treshpx=3.0, F_pts_a = [], F_pts_b = []):

    similcos = np.dot(plist_a,plist_b.transpose())

    # Filter by fundamental matrix
    if (F.size>0): #

        mask = func_get_mask_for_potential_matches_from_F(F, F_pts_a, F_pts_b, F_treshpx)

        similcos[mask] = 1e-10

    comb_a, val_a = func_get_first_second_neighbor_match(similcos.copy(), tr)
    comb_b, val_b = func_get_first_second_neighbor_match(similcos.copy().transpose(), tr)


    idxdel = []
    for i in xrange(comb_a.shape[1]):
        id = np.where(comb_b[1,:]==comb_a[0,i])[0]
        if (id.size==0):
            idxdel.append(i)
        else:
            if ((comb_a[1,i]!=comb_b[0,id[0]]) | (similcos[comb_a[1,i],comb_a[0,i]] < abstr)):
                idxdel.append(i)

    return np.delete(comb_a, idxdel, axis=1), np.delete(val_a, idxdel, axis=0)







def func_extract_bil_patch_list(pt_all, imgin, pz, do_zeromean=0, do_unitnorm=0):
    padval = pz/2+1
    img = np.pad(imgin, [(padval,padval),(padval,padval),(0,0)], 'edge')
    pt_all_pad = pt_all + padval

    patch_list = []

    for i in xrange(pt_all.shape[0]):
        patch_list.append(func_extract_bil_patch(pt_all_pad[i,:], img, pz, do_zeromean, do_unitnorm))

    patch_list = np.array(patch_list)

    return patch_list


def func_extract_bil_patch(ptin, img, pz, do_zeromean=0, do_unitnorm=0):

    ptfloor = np.floor(ptin).astype(int)
    ptceil = ptfloor+1
    ptf = ptin - ptfloor
    w = np.array([ptf[0]*ptf[1], (1-ptf[0])*ptf[1], ptf[0]*(1-ptf[1]), (1-ptf[0])*(1-ptf[1])])

    pz2 = pz/2
    pa = img[(ptceil[1]-pz2):(ptceil[1]+pz2+1), (ptceil[0]-pz2):(ptceil[0]+pz2+1), :]
    pb = img[(ptceil[1]-pz2):(ptceil[1]+pz2+1), (ptfloor[0]-pz2):(ptfloor[0]+pz2+1), :]
    pc = img[(ptfloor[1]-pz2):(ptfloor[1]+pz2+1), (ptceil[0]-pz2):(ptceil[0]+pz2+1), :]
    pd = img[(ptfloor[1]-pz2):(ptfloor[1]+pz2+1), (ptfloor[0]-pz2):(ptfloor[0]+pz2+1), :]

    pf = pa*w[0] + pb*w[1] + pc*w[2] + pd*w[3]

    if (do_zeromean==1):
        for i in xrange(pf.shape[2]):
            pf[:,:,i] -= np.mean(pf[:,:,i])

    if (do_unitnorm==1):
        pf /= np.linalg.norm(pf)

    pf = np.ndarray.flatten(pf)


    # # function test
    # img = np.zeros((10,20,3), dtype=float)
    # img[0:6,0:11,0] = 255
    # img[6:10, 0:11,1] = 255
    # img[6:10,11:20,2] = 255
    # pt2d_extract[0,0] = 10.5
    # pt2d_extract[0,1] = 5.5
    #
    # fig = plt.figure()
    # a=fig.add_subplot(1,2,1)
    # plt.imshow(img.astype(np.uint8), interpolation='None')
    # plt.hold(True)
    # plt.scatter(pt2d_extract[:,0], pt2d_extract[:,1],24, color='green')
    #
    # a=fig.add_subplot(1,2,2)
    # plt.imshow(func_extract_bil_patch(pt2d_extract[0,:], img, 3).astype(np.uint8), interpolation='None')
    # print func_extract_bil_patch(pt2d_extract[0,:], img, 3).shape
    # plt.show()

    return pf


data = scio.loadmat('/home/kroegert/local/Results/AccGPSFuse/FlorenceLionTest.mat')

noimg = data['cam']['F'][0].shape[0]

fc = np.mean(data['cam']['F'][0])[0][0]
fc = np.array([fc, fc])
kc = np.array([np.mean(data['cam']['radial'][0])[0][0]])
cc = np.array([0, 0], dtype=float)
pathprefix = '/home/kroegert/local/Code/VidReg_CodePackage/ToyDataset_LionFlorence/'
FileStr = []
R_all = []
T_all_w = []
pts3D_id = []
pts2D_id = []
for i in xrange(noimg):
    FileStr.append(pathprefix + data['cam']['FileStr'][0][i][0])
    R_all.append(data['cam']['R'][0][i])
    T_all_w.append(data['cam']['T'][0][i])
    pts3D_id.append(data['cam']['pts3D'][0][i][0,:].astype(int)-1)
    pts2D_id.append(data['cam']['pts2D'][0][i][0,:].astype(int)-1)


im_full = plt.imread(FileStr[0]).astype(float)
imwidth = im_full.shape[1]
imheight= im_full.shape[0]


nopt = len(data['pt']['XYZ'][0])

pt3D = []
pt3D_views=[]
for i in xrange(nopt):
    pt3D.append(data['pt']['XYZ'][0][i][0])

    noviews = data['pt']['V'][0][i]['CamIdx'][0].shape[0]

    tmpa = []
    tmpb = []
    tmpc = []
    for j in xrange(noviews):
        tmpa.append(data['pt']['V'][0][i]['CamIdx'][0][j][0][0].astype(int)-1)
        tmpb.append(data['pt']['V'][0][i]['FeatIdx'][0][j][0][0].astype(int)-1)
        tmpc.append(data['pt']['V'][0][i]['XY'][0][j][0,:].astype(float))
    tmpa = np.array(tmpa)
    tmpb = np.array(tmpb)
    tmpc = np.array(tmpc)

    data['pt']['V'][0][i]['CamIdx'][0] - 1
    data['pt']['V'][0][i]['FeatIdx'][0] - 1

    pt3D_views.append([tmpa, tmpb, tmpc])

pt3D = np.array(pt3D)

def func_return_2d3d_corresp(fr, pts3D_id, pt3D_views):
    nopt_re = pts3D_id[fr].shape[0]
    pt3d_valid = pt3D[pts3D_id[fr],:]
    pt2d_valid = np.zeros((nopt_re,2), dtype=float)
    for i in xrange(nopt_re):
        j=pts3D_id[fr][i]
        id = np.where(pt3D_views[j][0]==fr)[0][0]
        pt2d_valid[i,:] = pt3D_views[j][2][id]

    return nopt_re, pt3d_valid, pt2d_valid


#reprojection test for all frames
for fr in xrange(noimg):

    R = R_all[fr]
    T_w = T_all_w[fr]

    nopt_re, pt3d_valid, pt2d_valid = func_return_2d3d_corresp(fr, pts3D_id, pt3D_views)

    print np.mean(np.linalg.norm(func_reproject(pt3d_valid, R, T_w, fc, cc,kc, 0) - pt2d_valid, axis=1)) # mean reprojection error


#nopt_re, pt3d_valid, pt2d_valid_a = func_return_2d3d_corresp(fr  , pts3D_id, pt3D_views)
#pt2d_extract_a = pt2d_valid_a + np.array([imwidth/2.0, imheight/2.0])
#nopt_re, pt3d_valid, pt2d_valid_b = func_return_2d3d_corresp(fr+1, pts3D_id, pt3D_views)
#pt2d_extract_b = pt2d_valid_b + np.array([imwidth/2.0, imheight/2.0])


fr= 5
imga = plt.imread(FileStr[fr  ]).astype(float)
imgb = plt.imread(FileStr[fr+1]).astype(float)

# Feature locations: good features to track
gray = cv2.cvtColor(imga.astype(np.uint8),cv2.COLOR_BGR2GRAY).astype(np.float32)
cnr= cv2.goodFeaturesToTrack(gray, 10000, 0.001, 0)
pt2d_extract_a = np.squeeze(cnr)
gray = cv2.cvtColor(imgb.astype(np.uint8),cv2.COLOR_BGR2GRAY).astype(np.float32)
cnr= cv2.goodFeaturesToTrack(gray, 10000, 0.001, 0)
pt2d_extract_b = np.squeeze(cnr)

# Get features, zero-mean, unit normalized intensity patches
plist_a = func_extract_bil_patch_list(pt2d_extract_a, imga, 11, 1, 1)
plist_b = func_extract_bil_patch_list(pt2d_extract_b, imgb, 11, 1, 1)

# match, cosine similarity, first-second neighbor ratio
comb, val = func_match_cossimil_2view(plist_a, plist_b, 1.1, 0.6)


# display
imgc = np.concatenate((imga, imgb), axis=0)
pta = np.array([pt2d_extract_a[comb[1,:],0], pt2d_extract_b[comb[0,:],0]])
ptb = np.array([pt2d_extract_a[comb[1,:],1], pt2d_extract_b[comb[0,:],1]+ imheight])
fig = plt.figure()
plt.imshow(imgc.astype(np.uint8), interpolation='None')
plt.hold(True)
plt.plot(pta, ptb, '-r')
plt.show()


# Epipolar geometry test
# Selecting only the inliers
pts1 = pt2d_extract_a[comb[1,:],:]
pts2 = pt2d_extract_b[comb[0,:],:]
F, mask = cv2.findFundamentalMat(pts1,pts2,cv2.FM_RANSAC, 3.0, 0.99)
pts1 = pts1[mask[:,0]==1,:]
pts2 = pts2[mask[:,0]==1,:]


# display
imgc = np.concatenate((imga, imgb), axis=0)
pta = np.array([pts1[:,0], pts2[:,0]])
ptb = np.array([pts1[:,1], pts2[:,1]+ imheight])
fig = plt.figure()
plt.imshow(imgc.astype(np.uint8), interpolation='None')
plt.hold(True)
plt.plot(pta, ptb, '-r')
plt.show()


comb, val = func_match_cossimil_2view(plist_a, plist_b, 1.01, 0.5, F, 5.0, pt2d_extract_a, pt2d_extract_b)


# EPIPOLE Test
# Get epipole in 2nd image
U,S,V = np.linalg.svd(F.transpose())
epip = V[2,0:2] / V[2,2]

#id = random.randint(0,comb[1,:].shape[0])
print id
pts1 = pt2d_extract_a[comb[1,[id]],:]
pts2 = pt2d_extract_b[comb[0,[id]],:]
lines_a_in_b = np.squeeze(cv2.computeCorrespondEpilines(pts1, 1, F))
print (lines_a_in_b[0]*pts2[:,0] + lines_a_in_b[1]*pts2[:,1] + lines_a_in_b[2])
print (lines_a_in_b[0]*epip[0] + lines_a_in_b[1]*epip[1] + lines_a_in_b[2])


# display
imgc = np.concatenate((imga, imgb), axis=0)
#pta = np.array([pt2d_extract_a[comb[1,:],0], pt2d_extract_b[comb[0,:],0]])
#ptb = np.array([pt2d_extract_a[comb[1,:],1], pt2d_extract_b[comb[0,:],1]+ imheight])
pta = np.array([pts1[:,0], pts2[:,0]])
ptb = np.array([pts1[:,1], pts2[:,1]+ imheight])

fig = plt.figure()
plt.imshow(imgc.astype(np.uint8), interpolation='None')
plt.hold(True)
plt.plot(pta, ptb, '-r')
plt.plot(np.array([res[0], pts2[:,0]]),   np.array([res[1]+imheight, pts2[:,1]+imheight]), '-b') # draw line from epipole to matched feature location
plt.show()






#im_l = plt.imread('/home/kroegert/local/Datasets/Sintel-Stereo/training/final_left/alley_1/frame_0001.png').astype(float)*255.0
#im_r = plt.imread('/home/kroegert/local/Datasets/Sintel-Stereo/training/final_right/alley_1/frame_0001.png').astype(float)*255.0


def func_get_keypoints_and_features(img_in, maxnofeat=10000, cnrquality=0.001, psize=11, do_zeromean=1, do_unitnorm=1):
    # Feature locations: good features to track
    if (img_in.ndim>2):
        img = img_in
        gray = cv2.cvtColor(img.astype(np.uint8),cv2.COLOR_BGR2GRAY).astype(np.float32)
    else:
        img = img_in[:,:,None]
        gray = img.astype(np.float32)


    cnr= cv2.goodFeaturesToTrack(gray, maxnofeat, cnrquality, 0)
    pt2d_extract = np.squeeze(cnr)

    # Get features, zero-mean, unit normalized intensity patches
    plist = func_extract_bil_patch_list(pt2d_extract, img, psize, do_zeromean, do_unitnorm)

    return pt2d_extract, plist



bsize = 5
pathl = '/scratch_net/zinc/kroegert_depot/Datasets/Kitti_MVFlow/training/image_0/'
pathr = '/scratch_net/zinc/kroegert_depot/Datasets/Kitti_MVFlow/training/image_1/'
movid = 5

file = open("/scratch_net/zinc/kroegert_depot/Datasets/Kitti_FlowCalib/training/calib/" + "%06i.txt" % (movid), 'r')
calibfile = file.readlines()
file.close()
P0 = np.reshape(np.fromstring(calibfile[0][4:-1], dtype=float, sep=" "), (3,4))
P1 = np.reshape(np.fromstring(calibfile[1][4:-1], dtype=float, sep=" "), (3,4))
K01 = P0[0:3,0:3]
E01 = func_crossMatrix(P1[0:3,3])


# load images, compute corners and features
imgs = [] # list of tuples of images
pts = [] # list of tuples of 2d points
feats = [] # list of tuples of feature points
for i in  xrange(bsize):
    imgname = "%06i_%02i.png" % (movid,i)
    img_tuple = (plt.imread(pathl + imgname).astype(float)*255.0, plt.imread(pathr + imgname).astype(float)*255.0)
    imgs.append(img_tuple)

    pt2d_extract_l, plist_l = func_get_keypoints_and_features(img_tuple[0], maxnofeat=10000, cnrquality=0.01)
    pt2d_extract_r, plist_r = func_get_keypoints_and_features(img_tuple[1], maxnofeat=10000, cnrquality=0.01)
    pts_tuple = (pt2d_extract_l, pt2d_extract_r)
    feat_tuple = (plist_l, plist_r)

    pts.append(pts_tuple)
    feats.append(feat_tuple)

imheight = imgs[0][0].shape[0]
imwidth = imgs[0][0].shape[1]

# compute all stereo matches
comb_all = []
for fr in  xrange(bsize):
    comb, val = func_match_cossimil_2view(feats[fr][0], feats[fr][1], 1.000, 0.5, E01, 1.0, pts[fr][0], pts[fr][1])
    comb_all.append((comb, val))


# 4-view matching
fr=1
plist_l1 = feats[fr][0]
plist_r1 = feats[fr][1]
plist_l2 = feats[fr+1][0]
plist_r2 = feats[fr+1][1]
pts_l1 = pts[fr][0]
pts_r1 = pts[fr][1]
pts_l2 = pts[fr+1][0]
pts_r2 = pts[fr+1][1]
cc_lr1 = comb_all[fr]
cc_lr2 = comb_all[fr+1]

def func_match_cossimil_4view(plist_l1, plist_r1, plist_l2, plist_r2, pts_l1, pts_r1, pts_l2, pts_r2, cc_lr1, cc_lr2, tr_abs = 0.7):

    sc_ll  = np.dot(plist_l1,plist_l2.transpose())
    sc_rr  = np.dot(plist_r1,plist_r2.transpose())

    cc_lr2_list = list(cc_lr2[0].transpose()) #ids
    cc_lr2v_list =list(cc_lr2[1].transpose()) #values

    matches = []
    matches_val = []

    for i in xrange(cc_lr1[0].shape[1]):
        #print i
        id_l = cc_lr1[0][1,i] # left right matches
        id_r = cc_lr1[0][0,i]
        val_lr1 = cc_lr1[1][i] # left right score for i in lr1

        bestid = -1
        bestval = 0
        for j in xrange(len(cc_lr2_list)):
            val_ll = sc_ll[id_l,cc_lr2_list[j][1]]
            val_rr = sc_rr[id_r,cc_lr2_list[j][0]]
            val_lr2 = sc_rr[cc_lr2_list[j][0],cc_lr2_list[j][1]]
            val_final = val_lr1 * val_lr2 * val_ll * val_rr

            if ((val_final > bestval) & (val_ll > tr_abs) & (val_rr > tr_abs) & (val_lr1 > tr_abs) & (val_lr2 > tr_abs)):
                bestval = val_final
                bestid = j

        if (bestid != -1):
            matches.append((id_l,id_r, cc_lr2_list[bestid][1], cc_lr2_list[bestid][0]))
            matches_val.append(bestval)
            del cc_lr2_list[bestid]
            del cc_lr2v_list[bestid]

    return matches, matches_val

matches, matches_val = func_match_cossimil_4view(plist_l1, plist_r1, plist_l2, plist_r2, pts_l1, pts_r1, pts_l2, pts_r2, cc_lr1, cc_lr2, tr_abs = 0.7)

    # ### PSEUDO CODE , LinProg Feat.Select
    # # for every point x1 in l1
    #     # for matches y1 of x1 in r1 with score(y1,x1) > tr_abs
    #         # for matches x2 of x1 in l2 with score(x2,x1) > tr_abs
    #             # for matches y2 of x2 in r2 with min(score(y2,x2),score(y2,y1)) > tr_abs and px_diff_in_stereo_baseline < threshold
    #                 # save as match pair, with product of 4 scores
    #
    # # Binear Linear Program fx, Ax==0
    # # f = [x1c1,x1c2,...x1cN,x2c1,x2c2,...xMcN] of K candidates, x: selection which candidates are chosen,
    # # A sparse KxK matrix with 1 for all candidates which cannot be active together
    #
    # for i_x1 in xrange(plist_l1.shape[0]):
    #     listaa=[]
    #     for i_y1 in xrange(plist_r1.shape[0]):
    #         if (sc_lr1[i_x1,i_y1] > tr_abs):
    #             for i_x2 in xrange(plist_l2.shape[0]):
    #                 if (sc_ll[i_x1,i_x2] > tr_abs):
    #                     for i_y2 in xrange(plist_r2.shape[0]):
    #                         if ((sc_lr2[i_x2,i_y2] > tr_abs) & (sc_rr[i_y1,i_y2] > tr_abs)):
    #                             a =  sc_lr1[i_x1,i_y1] * sc_ll[i_x1,i_x2] * sc_lr2[i_x2,i_y2] * sc_rr[i_y1,i_y2]
    #                             listaa.append(a)
    #     print len(listaa)
    #
    #


# display 4-view
imgc = np.concatenate((np.concatenate((imgs[fr][0], imgs[fr+1][0]), axis=0), np.concatenate((imgs[fr][1], imgs[fr+1][1]), axis=0)), axis=1)
imgc = np.concatenate((imgc[:,:,None],imgc[:,:,None],imgc[:,:,None]), axis=2)
fig = plt.figure()
plt.imshow(imgc.astype(np.uint8), interpolation='None')
plt.hold(True)
plt.scatter(pts[fr][0][comb_all[fr][0][1,:],0]            , pts[fr][0][comb_all[fr][0][1,:],1], 10, color='blue')
plt.scatter(pts[fr][1][comb_all[fr][0][0,:],0]+imwidth    , pts[fr][1][comb_all[fr][0][0,:],1], 10, color='blue')
plt.scatter(pts[fr+1][0][comb_all[fr+1][0][1,:],0]        , pts[fr+1][0][comb_all[fr+1][0][1,:],1]+imheight, 10, color='blue')
plt.scatter(pts[fr+1][1][comb_all[fr+1][0][0,:],0]+imwidth, pts[fr+1][1][comb_all[fr+1][0][0,:],1]+imheight, 10, color='blue')
id = random.randint(0,len(matches))
plt.scatter(pts[fr][0][matches[id][0],0]          , pts[fr][0][matches[id][0],1], 20, color='red')
plt.scatter(pts[fr][1][matches[id][1],0]+imwidth  , pts[fr][1][matches[id][1],1], 20, color='red')
plt.scatter(pts[fr+1][0][matches[id][2],0]        , pts[fr+1][0][matches[id][2],1]+imheight, 20, color='red')
plt.scatter(pts[fr+1][1][matches[id][3],0]+imwidth, pts[fr+1][1][matches[id][3],1]+imheight, 20, color='red')
plt.show()


# Get OF from both frames over window size
# Get Stereo Depth
# 1. Initialize grid in L1, transfer to R1
# 2. Propagate to L2,R2 (backward consistency check, stereo baseline consistency check)
# 3. Extract Flow and depth from new positions, repeat 2,3 until window end
# 4. Compute camera change (RANSAC)



# display 2-view
imgc = np.concatenate((imgs[fr][0], imgs[fr][1]), axis=0)
imgc = np.concatenate((imgc[:,:,None],imgc[:,:,None],imgc[:,:,None]), axis=2)
pta = np.array([pts[fr][0][comb[1,:],0], pts[fr][1][comb[0,:],0]])
ptb = np.array([pts[fr][0][comb[1,:],1], pts[fr][1][comb[0,:],1]+ imheight])
fig = plt.figure()
plt.imshow(imgc.astype(np.uint8), interpolation='None')
plt.hold(True)
#plt.plot(pta, ptb, '-r')
plt.scatter(pts[fr][0][comb[1,:],0], pts[fr][0][comb[1,:],1], 10, color='red')
plt.scatter(pts[fr][1][comb[0,:],0], pts[fr][1][comb[0,:],1]+ imheight, 10, color='red')
plt.show()




