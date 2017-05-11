import numpy as np
import scipy.interpolate as scpint

def func_crossMatrix(x):
    return np.array([[0.0, -x[2], x[1]],[x[2], 0.0, -x[0]],[-x[1], x[0], 0.0]])

def func_comp_rot(R1,R2):
    func_dcm2quat(R1) * func_dcm2quat(R2)

    res = np.clip(np.abs(np.sum(func_dcm2quat(R1) * func_dcm2quat(R2))),0,1)
    return np.degrees(np.arccos(res))

def func_spline_orientation_smooth(x,y,p=-1):
    y_trafo = np.concatenate((y[:,1,:], y[:,2,:]), axis=1)
    if (p==-1):
        p = func_smoothing_spline_crossvalp(x,y_trafo, crossvalperc = 0.1, crrounds = 100, verbosity=1)
    yhat, LL, p = func_smoothing_spline(x,y_trafo,p) # smooth acceleration
    rotestim = np.zeros_like(y)
    rotestim[:,2,:] = yhat[:,3:6] / np.linalg.norm(yhat[:,3:6], axis=1)[:,None]
    rotestim[:,1,:] = yhat[:,0:3] / np.linalg.norm(yhat[:,0:3], axis=1)[:,None]
    rotestim[:,0,:] = np.cross(rotestim[:,1,:], rotestim[:,2,:])
    return rotestim, p

def func_spline_orientation_interpolate(x, y_in, x_new):
    y = np.concatenate((y_in[:,1,:], y_in[:,2,:]), axis=1)

    tck,u=scpint.splprep(y.transpose().tolist(),u=x, s=0.0) # interpolation spline over actual sensor data
    yhat = np.array(scpint.splev(x_new,tck)).transpose()

    res = np.zeros((x_new.shape[0], y_in.shape[1], y_in.shape[2]), dtype=float)
    res[:,2,:] = yhat[:,3:6] / np.linalg.norm(yhat[:,3:6], axis=1)[:,None]
    res[:,1,:] = yhat[:,0:3] / np.linalg.norm(yhat[:,0:3], axis=1)[:,None]
    res[:,0,:] = np.cross(res[:,1,:], res[:,2,:])
    return res


def func_dcm2quat(rotin): # compute quaternion from dcm
    # TODO, Note: UNTESTED !
    m00 = rotin[0, 0]
    m01 = rotin[0, 1]
    m02 = rotin[0, 2]
    m10 = rotin[1, 0]
    m11 = rotin[1, 1]
    m12 = rotin[1, 2]
    m20 = rotin[2, 0]
    m21 = rotin[2, 1]
    m22 = rotin[2, 2]

    tmp = np.array(  [[m00-m11-m22,  0.0,          0.0,          0.0],
                     [m01+m10,      m11-m00-m22,  0.0,          0.0],
                     [m02+m20,      m12+m21,      m22-m00-m11,  0.0],
                     [m21-m12,      m02-m20,      m10-m01,      m00+m11+m22]])
    tmp /= 3.0
    # quaternion is eigenvector of K that corresponds to largest eigenvalue
    w, V = np.linalg.eigh(tmp)
    res = V[[3, 0, 1, 2], np.argmax(w)]

    if res[0] < 0.0:
        np.negative(res, res)
    return res


def func_rodr_to_rotM(x):
  # TODO: THIS FUNCTION IS UNTESTED!
    a = x
    alpha = np.linalg.norm(a)
    C = func_crossMatrix(a)
    D = np.dot(C,C)
    if (alpha==0):
        y = np.identity(3, dtype=np.float)
    else:
        c = np.sin(alpha)/alpha
        d = (1-np.cos(alpha))/alpha**2
        y = np.identity(3, dtype=np.float) + c*C + d*D
    return y

def func_android_rotM_from_gyroscope(timestamps, gyrodata, dosvd=True):
    rotestim = np.zeros((timestamps.shape[0],3,3), dtype=float)
    rotestim[0,:,:] = np.identity(3,dtype=float)
    for i in xrange(timestamps.shape[0]-1):
        dT = timestamps[i+1] - timestamps[i]
        #estimrot  = scpint.splint(data_rotvec[0][i],data_rotvec[0][i+1],tck)
        estimrot = gyrodata[i,:].copy()

        omegaMagnitude = np.linalg.norm(estimrot)
        estimrot /= omegaMagnitude

        thetaOverTwo = omegaMagnitude * dT / 2.0

        sinThetaOverTwo = np.sin(thetaOverTwo)
        cosThetaOverTwo = np.cos(thetaOverTwo)

        deltaRotationVector = np.zeros((4,), dtype=float)
        deltaRotationVector[0] = sinThetaOverTwo * estimrot[0]
        deltaRotationVector[1] = sinThetaOverTwo * estimrot[1]
        deltaRotationVector[2] = sinThetaOverTwo * estimrot[2]
        deltaRotationVector[3] = cosThetaOverTwo

        R01test = func_android_rotM_from_rotvec(deltaRotationVector, dosvd)

        rotestim[i+1,:,:] = np.dot(rotestim[i,:,:], R01test)

    return rotestim

def func_android_rotM_from_rotvec(rv, dosvd=False):
  
  R = np.identity(3, dtype=np.float)
  q1 = rv[0]
  q2 = rv[1]
  q3 = rv[2]
  
  if (rv.shape[0]==4):
    q0 = rv[3]
  else:
    q0 = 1.0 - q1*q1 - q2*q2 - q3*q3;
    if (q0 < 0.0):
      q0 = 0.0
      
  sq_q1 = 2 * q1 * q1
  sq_q2 = 2 * q2 * q2
  sq_q3 = 2 * q3 * q3
  q1_q2 = 2 * q1 * q2
  q3_q0 = 2 * q3 * q0
  q1_q3 = 2 * q1 * q3
  q2_q0 = 2 * q2 * q0
  q2_q3 = 2 * q2 * q3
  q1_q0 = 2 * q1 * q0

  R[0,0] = 1 - sq_q2 - sq_q3
  R[0,1] = q1_q2 - q3_q0
  R[0,2] = q1_q3 + q2_q0

  R[1,0] = q1_q2 + q3_q0
  R[1,1] = 1 - sq_q1 - sq_q3
  R[1,2] = q2_q3 - q1_q0

  R[2,0] = q1_q3 - q2_q0
  R[2,1] = q2_q3 + q1_q0
  R[2,2] = 1 - sq_q1 - sq_q2
  
  if (dosvd==True): # if not perfectly orthonormal, renormalize with SVD
    U,S,V = np.linalg.svd(R)
    R = np.dot(U,V)
  
  return R


def func_smoothing_spline_batch(x,y,p, batchsize = 2000, overlap = .49):
    noverlap = int(batchsize*overlap)


    n = x.shape[0]

    if (batchsize > n):
        batchsize=n

    startpos = np.array(range(0,n-noverlap,batchsize-noverlap))
    endpos = np.minimum(startpos+batchsize, n)
    nblocks = startpos.shape[0]
    lb = np.floor(noverlap/2.0).astype(int)
    ub =np.ceil(noverlap/2.0).astype(int)


    yhat_list = []
    yhat_final = np.zeros_like(y)
    for nb in xrange(nblocks):
        yhat, LL, p = func_smoothing_spline(x[startpos[nb]:endpos[nb]],y[startpos[nb]:endpos[nb],:],p)
        yhat_list.append(yhat)

        #plt.plot(yhat[:,0], yhat[:,1], color='red', label='desired', linewidth=2.0)

        if (nb==0):
            yhat_final[:(endpos[nb]-ub),:] = yhat[:(batchsize-ub),:]

        if (nb==nblocks-1):
            yhat_final[(startpos[nb]+lb):,:] = yhat[lb:,:]

        if ((nb > 0) & (nb < (nblocks-1))):
            yhat_final[(startpos[nb]+lb):(endpos[nb]-ub),:] = yhat[lb:(batchsize-ub),:]

    return yhat_final


def func_smoothing_spline_crossvalp(xin,yin, crossvalperc = 0.1, crrounds = 1000, depthiter = 10, treespread = 5, verbosity=0):

    n = yin.shape[0]
    nremove = np.ceil(n*crossvalperc).astype(np.int)

    # 1: percent of randomly selected datapoints to remove (at least one)
    # 2: do x crossvalidation rounds before proceeding to parameter refinement
    # tree depth for binary search
    # breadth of tree

    parr = np.linspace(1e-10,1,treespread)
    resp = np.zeros_like(parr)

    def func_sp_compute_residual(xin, yin, p, crrounds, n, nremove):
        crresidual = []
        for cri in xrange(crrounds):
            # randomly remove elements
            idxdel = np.random.choice(n, nremove, replace=False)
            xcr = np.delete(xin,idxdel,0)
            ycr = np.delete(yin,idxdel,0)

            yhat, LL, p = func_smoothing_spline(xcr,ycr,p)

            tck,u=scpint.splprep(yhat.transpose().tolist(),u=xcr, s=0.0)

            res = scpint.splev(xin,tck)
            yhat_intp = np.array(res).transpose()

            residual = np.linalg.norm(yhat_intp[idxdel,:]-yin[idxdel,:], axis=1)**2
            residual = np.mean(residual)

            crresidual.append(residual)

        crresidual = np.mean(np.array(crresidual))

        return crresidual


    for i in xrange(treespread):
        resp[i] = func_sp_compute_residual(xin, yin, parr[i], crrounds, n, nremove)

    idxmin = np.argmin(resp)
    bestp = parr[idxmin]
    if (verbosity>0):
        print resp

    for i in xrange(depthiter):
        idxleft = np.maximum(0,idxmin-1)
        idxright = np.minimum(treespread-1,idxmin+1)

        resp[0] = resp[idxleft]
        resp[treespread-1] = resp[idxright]

        parr = np.linspace(parr[idxleft],parr[idxright],treespread)

        for i in xrange(1,treespread-1):
            resp[i] = func_sp_compute_residual(xin, yin, parr[i], crrounds, n, nremove)


        if (verbosity>0):
            print resp
        idxmin = np.argmin(resp)

        bestp = parr[idxmin]

    return bestp


def func_smoothing_spline(x,y,p):
  n = x.shape[0];

  hi = np.diff(x)

  delta = np.zeros((n-2,n), dtype=float)

  for i  in xrange(n-2):
      delta[i,i] =  1.0 / hi[i]
      delta[i,i+1] =  - 1.0 / hi[i] - 1 / hi[i+1]
      delta[i,i+2] =  1.0 / hi[i+1]


  W = np.zeros((n-2,n-2), dtype=float)
  for i  in xrange(n-2):
      W[i,i] =  (hi[i]+hi[i+1])/3.0;
      if (i>0):
          W[i-1,i] =  hi[i]/6.0;
          W[i,i-1] =  hi[i]/6.0;


  Winv = np.linalg.inv(W)
  K = np.dot(delta.transpose(), np.dot(Winv,delta))

  #fig = plt.figure()
  #ax = fig.add_subplot(1, 1, 1)
  #ax.imshow(LL[0:10,0:10], extent=[0,1,0,1], aspect='auto', interpolation='None')
  #plt.show()


  # Fit with smoother matrix
  LL = np.linalg.inv(np.identity(n,dtype=float) +  (1/p)*K - K)
  yhat = np.dot(LL , y)

  return yhat, LL, p


def func_set_axes_equal(ax):
    '''Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    '''

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5*max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])



def func_plot_cameras(ax, fc,cc,wh,R,t,rgbface=np.array([1.0, 0.0, 0.0]),camerascaling=2.0,lw=2.0):
    K = np.identity(3, dtype=np.float)
    K[0,0] = fc[0]
    K[1,1] = fc[1]
    K[0,2] = cc[0]
    K[1,2] = cc[1]

    Rinv = np.linalg.inv(R)
    camcent = -np.dot(Rinv,t)  # correct camera center in world coordinates, -R't = X, because x = RX+t -> R'x - R't = X and R'x is 0,0,0 (camera center)

    # plot new cameras
    # compute corner points
    x = np.array([[-wh[0]/2,-wh[1]/2,1.0], [-wh[0]/2,wh[1]/2,1.0], [wh[0]/2,wh[1]/2,1.0], [wh[0]/2,-wh[1]/2,1.0]]) # 2d homog. point in image plane
    X = np.dot(np.linalg.inv(K), x.transpose()) # inverse projection
    X /= np.linalg.norm(X, axis=0)[None, :] / camerascaling # normalization and  camera scaling
    X = np.dot(Rinv, (X - t[:,None])) # apply rotation and translation -> world coordinates

    ax.scatter(X[0,:], X[1,:], X[2,:], zdir='z', s=20, c=rgbface, depthshade=True)
    ax.scatter(camcent[0], camcent[1], camcent[2], zdir='z', s=20, c=rgbface, depthshade=True)

    for k in xrange(4):
        ax.plot([camcent[0], X[0,k]],[camcent[1], X[1,k]],[camcent[2], X[2,k]], color=rgbface, linewidth=lw)

    ax.plot(X[0,:],X[1,:],X[2,:], color=rgbface, linewidth=lw)
    ax.plot([X[0,-1], X[0,0]], [X[1,-1], X[1,0]], [X[2,-1], X[2,0]], color=rgbface, linewidth=lw)

    #ax.plot_surface(reshape(X(1,[1 2 4 3]),2,2),reshape(X(2,[1 2 4 3]),2,2),reshape(X(3,[1 2 4 3]),2,2),'FaceAlpha',.4,'FaceColor',rgbface,'EdgeColor',rgbface,'EdgeAlpha',.5);

    ax.plot_surface(np.reshape(X[0,[0,1,3,2]],(2,2)), np.reshape(X[1,[0,1,3,2]],(2,2)), np.reshape(X[2,[0,1,3,2]],(2,2)), color=rgbface, alpha=0.4)
    


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
        ttmp = X-t
        ttmp = np.dot(R,ttmp.transpose()).transpose()
    else:
        ttmp = np.dot(R,X.transpose()).transpose()
        ttmp = ttmp-t

    ttmp[:,0] /= ttmp[:,2]
    ttmp[:,1] /= ttmp[:,2]

    ttmp = func_dist_kc(ttmp, kc)

    ttmp[:,0] = (ttmp[:,0] * fc[0]) + cc[0]
    ttmp[:,1] = (ttmp[:,1] * fc[1]) + cc[1]

    return ttmp


