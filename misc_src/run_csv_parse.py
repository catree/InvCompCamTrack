import sys
import csv
import numpy as np
import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from func_util_geom import func_crossMatrix, func_rodr_to_rotM, func_android_rotM_from_rotvec, func_android_rotM_from_gyroscope, func_set_axes_equal, func_plot_cameras, func_smoothing_spline_crossvalp, func_smoothing_spline, func_smoothing_spline_batch, func_comp_rot, func_dcm2quat, func_spline_orientation, func_spline_orientation_interpolate
import random
import matplotlib.pyplot as plt
import scipy.interpolate as scpint
import math


def func_parse_imugps_cvs(filename):
    gps_t = []
    gps_val = []
    accel_t = []
    accel_val = []
    gyro_t = []
    gyro_val = []
    orient_t = []
    orient_val = []

    linacc_t = []
    linacc_val = []
    rotvec_t = []
    rotvec_val = []
    grav_t = []
    grav_val = []


    with open(filename, 'rb') as csvfile:
        filereader = csv.reader(csvfile, delimiter=',')
        for row in filereader:
            time_t = float(row[0])
            pt = 1
            while pt < len(row):
                sensid = int(row[pt])
                if (np.in1d(sensid, np.array([1,3,4, 5, 6, 7, 81, 82, 83, 84]))[0]):
                    val = np.array([float(row[pt+1]), float(row[pt+2]), float(row[pt+3])])
                    pt += 4
                elif (np.in1d(sensid, np.array([8, 85, 86]))[0]):
                    val = np.array([float(row[pt+1])])
                    pt += 2

                #print sensid
                if (sensid==1):
                    gps_t.append(time_t)
                    gps_val.append(val)
                elif (sensid==3):
                    accel_t.append(time_t)
                    accel_val.append(val)

                    #print ', '.join(row)
                    #time.sleep(5)
                elif (sensid==4):
                    gyro_t.append(time_t)
                    gyro_val.append(val)
                #elif (sensid==5): # Magnetism UNUSED
                #elif (sensid==6): # _XYZ_WGS84 UNUSED
                #elif (sensid==7): # VELOCITY_WGS84 UNUSED
                #elif (sensid==8): # GPS_UTC_TIME UNUSED
                elif (sensid==81):
                    orient_t.append(time_t)
                    orient_val.append(val)
                elif (sensid==82): # Lin_Acc UNUSED
                    linacc_t.append(time_t)
                    linacc_val.append(val)
                elif (sensid==83): # Gra UNUSED
                    grav_t.append(time_t)
                    grav_val.append(val)
                elif (sensid==84): # Rot_Vec UNUSED
                    rotvec_t.append(time_t)
                    rotvec_val.append(val)

                #elif (sensid==85): # Pre UNUSED
                #elif (sensid==86): # Bat tmp  UNUSED


            #print ', '.join(row)


    data_gps = (np.array(gps_t), np.array(gps_val))
    data_accel = (np.array(accel_t), np.array(accel_val))
    data_gyro = (np.array(gyro_t), np.array(gyro_val))
    data_orient = (np.array(orient_t), np.array(orient_val))
    data_linacc = (np.array(linacc_t), np.array(linacc_val))
    data_rotvec = (np.array(rotvec_t), np.array(rotvec_val))
    data_grav = (np.array(grav_t), np.array(grav_val))


    def func_del_invalid_rows(input):
        time, val = input
        if (time.size>0):
            idxdel = np.where(np.prod(val==0.0, axis=1) | np.prod(np.isinf(val), axis=1) | np.prod(np.isnan(val), axis=1))[0]
            val = np.delete(val,idxdel, axis=0)
            time = np.delete(time,idxdel, axis=0)
        return (time, val)

    data_gps = func_del_invalid_rows(data_gps)
    data_accel = func_del_invalid_rows(data_accel)
    data_gyro = func_del_invalid_rows(data_gyro)
    data_orient = func_del_invalid_rows(data_orient)
    data_linacc = func_del_invalid_rows(data_linacc)
    data_rotvec = func_del_invalid_rows(data_rotvec)
    data_grav = func_del_invalid_rows(data_grav)

    return data_gps, data_accel, data_gyro, data_orient, data_linacc, data_rotvec, data_grav

  



def run_main():

    np.set_printoptions(linewidth=150)
  
    filename = '/home/kroegert/local/Results/AccGPSFuse/mystream_6_10_12_54_5.csv'
    #filename = '/home/till/zinc/local/Results/AccGPSFuse/mystream_6_10_12_54_5.csv'
    #filename = '/home/kroegert/local/Results/AccGPSFuse/mystream_6_11_10_42_32.csv'
    #filename = '/home/till/zinc/local/Results/AccGPSFuse/mystream_6_11_10_55_10_phoneonback.csv'
    #filename = '/home/till/zinc/local/Results/AccGPSFuse/mystream_6_11_10_57_31_phoneonfront.csv'
    #filename = '/home/kroegert/local/Results/AccGPSFuse/mystream_6_11_11_29_28_phoneturn.csv'
    #filename = '/home/kroegert/local/Results/AccGPSFuse/mystream_6_15_10_59_5_walkcircle.csv'
    #filename = '/home/kroegert/local/Results/AccGPSFuse/mystream_6_15_11_24_41_phonevertcircle.csv'
    #filename = '/home/kroegert/local/Results/AccGPSFuse/mystream_6_15_16_12_46_phonesidewaysslide.csv'


    
    data_gps, data_accel, data_gyro, data_orient, data_linacc, data_rotvec, data_grav = func_parse_imugps_cvs(filename)

    # Android coordinate system: When a device is held in its default orientation, the X axis is horizontal and points to the right, the Y axis is vertical and points up,
    # and the Z axis points toward the outside of the screen face. In this system, coordinates behind the screen have negative Z values.


    # *** Determine start and end of valid lin.accel and gyroscope data
    t_start_track = np.maximum(data_linacc[0][0], data_gyro[0][0])
    t_end_track = np.minimum(data_linacc[0][0], data_gyro[0][0])

    # *** GPS to cartesian
    # TODO: Note, Android world system spanned by vectors: gravity+direction to magnetic north, GPScartesian: world axis and arbitrary x/y plane
    testpt = data_gps[1][:,0:3]
    # testpt[:,1] : longitude, should be x axis in ref. system,
    # testpt[:,0] : latitude, should be y axis in ref. system,
    GPS_xyz = np.vstack((np.sin(np.radians(testpt[:,0])) * np.sin(np.radians(testpt[:,1])) * (testpt[:,2] + 6371000), np.sin(np.radians(testpt[:,0])) * (testpt[:,2] + 6371000), np.sin(np.radians(testpt[:,0])) * np.cos(np.radians(testpt[:,1])) * (testpt[:,2] + 6371000))).transpose()
    GPS_xyz -= GPS_xyz[testpt.shape[0]/2,:][None,:]

    # *** normalize gravity vector to unit norm
    data_grav = (data_grav[0], data_grav[1] / np.linalg.norm(data_grav[1], axis=1)[:, None])


    # *** prepare camera orientation from rotvec data
    rotgt_org = np.zeros((len(data_rotvec[0]),3,3), dtype=float )
    for i in xrange(len(data_rotvec[0])): # Plot cameras from absolute orientation estimation (directly from sensor)
        rotgt_org[i,:,:] = func_android_rotM_from_rotvec(data_rotvec[1][i,:], True)


    # *** estimate camera orientation from rotational acceleration data
    rotestim = func_android_rotM_from_gyroscope(data_gyro[0], data_gyro[1], dosvd=True)
    #rotestim,p = func_spline_orientation_smooth(data_gyro[0],rotestim) # smooth orientation

    #rotestim_ = func_spline_orientation_interpolate(data_gyro[0][0:-1:40], rotestim[0:-1:40,:,:], data_gyro[0])
    #[func_comp_rot(rotestim[i,:,:], rotestim_new[i,:,:])   for i in xrange(rotestim_new.shape[0])]



    # *** compute camera trajectory from lin.accel and gyroscopic data
    rotestim_linacc = func_spline_orientation_interpolate(data_gyro[0], rotestim, data_linacc[0]) # fit gyroscopic orientation data to lin.accel data

    # fit GT camera orientation to lin.acc. data
    rotgt = func_spline_orientation_interpolate(data_rotvec[0], rotgt_org, data_linacc[0])



    linaccabs = np.zeros((data_linacc[0].shape[0],3), dtype=float ) # linear acceleration in world frame
    for i in xrange(data_linacc[0].shape[0]):
        linaccabs[i,:] = np.dot(np.linalg.inv(rotgt[i,:,:]), data_linacc[1][i,:]) # use fused rotation data

    tck = scpint.splrep(data_linacc[0], linaccabs[:,0], s=0.0) # x acceleration spline
    tck = scpint.splantider(tck,2) # double integral -> x displacement
    x = scpint.splev(data_linacc[0], tck)
    tck = scpint.splrep(data_linacc[0], linaccabs[:,1], s=0.0) # y acceleration spline
    tck = scpint.splantider(tck,2) # double integral -> y displacement
    y = scpint.splev(data_linacc[0], tck)
    tck = scpint.splrep(data_linacc[0], linaccabs[:,2], s=0.0) # z acceleration spline
    tck = scpint.splantider(tck,2) # double integral -> z displacement
    z = scpint.splev(data_linacc[0], tck)
    posaccum = np.stack((x,y,z), axis=1)

    # posaccum = np.zeros((rotestim_linacc.shape[0],3), dtype=float )
    # R0 = func_android_rotM_from_rotvec(data_rotvec[1][0,:], True)
    # for i in xrange(data_linacc[0].shape[0]-2):
    #     #R = np.dot(R0,rotestim_linacc[i,:,:]) # use GT location from first frame to align initial camera
    #     R = func_android_rotM_from_rotvec(data_rotvec[1][i,:], True) # use absolute orientation data to add linear acceleration
    #     #posaccum[i+1,:] = posaccum[i,:] + np.dot(np.linalg.inv(R), data_linacc[1][i,:])




    # *** estimate camera path from linear acceleration data
    # tck,u=scpint.splprep(data_linacc[1].transpose().tolist(),u=data_linacc[0], s=0.0) # interpolation spline over actual sensor data
    #
    # #p = func_smoothing_spline_crossvalp(data_linacc[0][1000:1500],data_linacc[1][1000:1500,:], crossvalperc = 0.1, crrounds = 100, verbosity=1)
    # #linaccsm, LL, p = func_smoothing_spline(data_linacc[0],data_linacc[1],p) # smooth acceleration
    # #tck,u=scpint.splprep(linaccsm.transpose().tolist(),u=data_linacc[0], s=0.0) # interpolation spline over smoothed sensor data
    # #linaccsm = np.array(scpint.splev(data_gps[0],tck)).transpose()



    ### Plot camera path
    wh = np.array([1280.0,960.0])
    fc = np.array([900.0,900.0])
    cc = np.array([0.0, 0.0])

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.plot(posaccum[5:,0], posaccum[5:,1], posaccum[5:,2], color=np.array([0.0, 1.0, 0.0]), linewidth=2.0)

    func_set_axes_equal(ax)
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    plt.show()


    ### Plot camera orientation path, and gravity direction
    wh = np.array([1280.0,960.0])
    fc = np.array([900.0,900.0])
    cc = np.array([0.0, 0.0])

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')


    t = np.array([1.0, 10.0, 100.0])
    for i in xrange(0,len(data_rotvec[0]),50): # Plot cameras from absolute orientation estimation (directly from sensor)
        R = rotgt_org[i,:,:] #func_android_rotM_from_rotvec(data_rotvec[1][i,:], True)
        tc = -np.dot(R, t ) # position in camera centric coordinate frame
        func_plot_cameras(ax, fc,cc,wh,R,tc,rgbface=np.array([1.0, 0.0, 0.0]),camerascaling=2.0,lw=2.0)
    #
    # for i in xrange(0,len(data_grav[0]),40): # Plot gravity vectors
    #     R = func_android_rotM_from_rotvec(data_rotvec[1][i,:], True)
    #     g_vec = data_grav[1][i,:] # gravity vector (in camera reference frame)
    #     g_vec_rot = np.dot(R,g_vec[:]) # gravity vector (in world ref. frame, uses estimated camera orientation)
    #
    #     ax.plot([t[0], t[0]+g_vec[0]], [t[1], t[1]+g_vec[1]], [t[2], t[2]+g_vec[2]], color=np.array([0.0, 0.0, 1.0]), linewidth=2.0)
    #     ax.plot([t[0], t[0]+g_vec_rot[0]], [t[1], t[1]+g_vec_rot[1]], [t[2], t[2]+g_vec_rot[2]], color=np.array([0.0, 1.0, 0.0]), linewidth=2.0)

    # for i in xrange(0,data_gyro[0].shape[0],10):  # Plot cameras from tracked orientation (accumulated gyroscope data)
    #     R0 = func_android_rotM_from_rotvec(data_rotvec[1][0,:], True)
    #     R = np.dot(R0,rotestim[i,:,:]) # use GT location from first frame to align initial camera
    #     tc = -np.dot(R, t ) # position in camera centric coordinate frame
    #     func_plot_cameras(ax, fc,cc,wh,R,tc,rgbface=np.array([0.0, 0.0, 1.0]),camerascaling=2.0,lw=4.0)

    #g_vec = data_grav[1][0,:]
    #ax.plot([t[0], t[0]+g_vec[0]], [t[1], t[1]+g_vec[1]], [t[2], t[2]+g_vec[2]], color=np.array([0.0, 1.0, 0.0]), linewidth=4.0)

    #g_vec = data_grav[1][-1,:]
    #ax.plot([t[0], t[0]+g_vec[0]], [t[1], t[1]+g_vec[1]], [t[2], t[2]+g_vec[2]], color=np.array([1.0, 0.0, 1.0]), linewidth=4.0)

    func_set_axes_equal(ax)
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    plt.show()


    ### Plot GPS path
    m = data_gps[0].shape[0]
    testpt = data_gps[1][:,0:3]
    #p = func_smoothing_spline_crossvalp(data_gps[0],data_gps[1][:,0:2], crossvalperc = 0.1, crrounds = 10, verbosity=1)
    testptsm, LL, p = func_smoothing_spline(data_gps[0],data_gps[1][:,0:3],p)
    tck,u=scpint.splprep(testptsm.transpose().tolist(),u=data_gps[0], s=0.0) # new homogeneous samples
    t_new = np.linspace(data_gps[0][0],data_gps[0][-1],500)
    testptsm = np.array(scpint.splev(t_new,tck)).transpose()
    plt.scatter(testpt[:,1], testpt[:,0],24, color='blue')
    plt.scatter(testpt[0,1], testpt[0,0],24, color='green')
    plt.plot(testpt[:,1], testpt[:,0],24, '-', color='b')
    plt.plot(testptsm[:,1], testptsm[:,0], '-', color='r', markersize=24)
    plt.hold(True)
    plt.axis([np.min(testpt[:,1]),  np.max(testpt[:,1]), np.min(testpt[:,0]),  np.max(testpt[:,0])])
    plt.show()

    # 3D, display cartesian
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(GPS_xyz[:,0], GPS_xyz[:,1], GPS_xyz[:,2], color=np.array([0.0, 0.0, 1.0]), linewidth=2.0)
    ax.scatter(GPS_xyz[0,0], GPS_xyz[0,1], GPS_xyz[0,2], color=np.array([0.0, 1.0, 0.0]), linewidth=2.0)

    func_set_axes_equal(ax)
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    plt.show()


