import numpy as np
import scipy.interpolate as scpint
from scipy import linalg
import ctypes

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

def func_inv_sym3x3(m):
    minv = np.zeros_like(m)
    minv[0,0] = m[2,2] * m[1,1] - m[1,2]**2
    minv[0,1] = m[0,2] * m[1,2] - m[2,2]*m[0,1]
    minv[0,2] = m[0,1] * m[1,2] - m[0,2]*m[1,1]
    minv[1,1] = m[2,2] * m[0,0] - m[0,2]**2
    minv[1,2] = m[0,1] * m[0,2] - m[0,0]*m[1,2]
    minv[2,2] = m[0,0] * m[1,1] - m[0,1]**2
    minv[1,0] = minv[0,1]
    minv[2,1] = minv[1,2]
    minv[2,0] = minv[0,2]
    det = m[0,0] * minv[0,0] + m[1,0] * minv[1,0] + m[0,2] * minv[0,2]
    return minv / det

def func_spline_orientation_interpolate(x, y_in, x_new):
    y = np.concatenate((y_in[:,1,:], y_in[:,2,:]), axis=1)

    tck,u=scpint.splprep(y.transpose().tolist(),u=x, s=0.0) # interpolation spline over actual sensor data
    yhat = np.array(scpint.splev(x_new,tck)).transpose()

    res = np.zeros((x_new.shape[0], y_in.shape[1], y_in.shape[2]), dtype=float)
    res[:,2,:] = yhat[:,3:6] / np.linalg.norm(yhat[:,3:6], axis=1)[:,None]
    res[:,1,:] = yhat[:,0:3] / np.linalg.norm(yhat[:,0:3], axis=1)[:,None]
    res[:,0,:] = np.cross(res[:,1,:], res[:,2,:])
    return res

def func_F_transfer_points(F, ptsa, ptsb):
  import cv2
  lines_a_in_b = np.squeeze(cv2.computeCorrespondEpilines(ptsa, 1, F))
  return np.abs(lines_a_in_b[:,0] * ptsb[:,0] + lines_a_in_b[:,1] * ptsb[:,1] + lines_a_in_b[:,2])


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


def func_quat2dcm(qin): # compute dcm from quaternions

  dcm = np.zeros((3,3))
  
  qin /= np.linalg.norm(qin)

  dcm[0,0] = qin[0]**2 + qin[1]**2 - qin[2]**2 - qin[3]**2;
  dcm[1,1] = qin[0]**2 - qin[1]**2 + qin[2]**2 - qin[3]**2;
  dcm[2,2] = qin[0]**2 - qin[1]**2 - qin[2]**2 + qin[3]**2;
  dcm[1,0] = 2*(qin[1]*qin[2] + qin[0]*qin[3]);
  dcm[0,1] = 2*(qin[1]*qin[2] - qin[0]*qin[3]);
  dcm[2,0] = 2*(qin[1]*qin[3] - qin[0]*qin[2]);
  dcm[0,2] = 2*(qin[1]*qin[3] + qin[0]*qin[2]);
  dcm[2,1] = 2*(qin[2]*qin[3] + qin[0]*qin[1]);
  dcm[1,2] = 2*(qin[2]*qin[3] - qin[0]*qin[1]);

  return dcm 



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



def func_plot_cameras(ax, fc, cc, wh, R, camcent, rgbface=np.array([1.0, 0.0, 0.0]),camerascaling=2.0,lw=2.0):
    K = np.identity(3, dtype=np.float)
    K[0,0] = fc[0]
    K[1,1] = fc[1]
    K[0,2] = cc[0]
    K[1,2] = cc[1]

    Rinv = np.linalg.inv(R)
    #camcent = -np.dot(Rinv,t)  # correct camera center in world coordinates, -R't = X, because x = RX+t -> R'x - R't = X and R'x is 0,0,0 (camera center)

    # plot new cameras
    # compute corner points
    x = np.array([[-wh[0]/2,-wh[1]/2,1.0], [-wh[0]/2,wh[1]/2,1.0], [wh[0]/2,wh[1]/2,1.0], [wh[0]/2,-wh[1]/2,1.0]]) # 2d homog. point in image plane
    X = np.dot(np.linalg.inv(K), x.transpose()) # inverse projection
    
    
    X /= np.linalg.norm(X, axis=0)[None, :] / camerascaling # normalization and  camera scaling
    #X[2,:] = -X[2,:] # camera looks into negative Z direction
    #X = np.dot(Rinv, (X - t[:,None])) # apply rotation and translation -> world coordinates
    X = np.dot(Rinv, X) + camcent[:,None] # apply rotation and translation -> world coordinates

    ax.scatter(X[0,:], X[1,:], X[2,:], zdir='z', s=20, c=rgbface, depthshade=True)
    ax.scatter(camcent[0], camcent[1], camcent[2], zdir='z', s=20, c=rgbface, depthshade=True)

    for k in xrange(4):
        ax.plot([camcent[0], X[0,k]],[camcent[1], X[1,k]],[camcent[2], X[2,k]], color=rgbface, linewidth=lw)
    ax.scatter(X[0,0], X[1,0], X[2,0], zdir='z', s=50, c=rgbface, depthshade=True)
        

    ax.plot(X[0,:],X[1,:],X[2,:], color=rgbface, linewidth=lw)
    ax.plot([X[0,-1], X[0,0]], [X[1,-1], X[1,0]], [X[2,-1], X[2,0]], color=rgbface, linewidth=lw)

    #ax.plot_surface(reshape(X(1,[1 2 4 3]),2,2),reshape(X(2,[1 2 4 3]),2,2),reshape(X(3,[1 2 4 3]),2,2),'FaceAlpha',.4,'FaceColor',rgbface,'EdgeColor',rgbface,'EdgeAlpha',.5);

    ax.plot_surface(np.reshape(X[0,[0,1,3,2]],(2,2)), np.reshape(X[1,[0,1,3,2]],(2,2)), np.reshape(X[2,[0,1,3,2]],(2,2)), color=rgbface, alpha=0.4)

def func_get_cov_ellipsoid(cov, Nstd=1, nobins=50):
    evals, evecs = linalg.eigh(cov)
    
    # For N standard deviations spread of data, the radii of the eliipsoid will
    # be given by N*SQRT(eigenvalues).
    rx,ry,rz = Nstd*np.sqrt(evals);

    # Set of all spherical angles:
    u = np.linspace(0, 2 * np.pi, nobins)
    v = np.linspace(0, np.pi, nobins)

    # Cartesian coordinates that correspond to the spherical angles:
    # (this is the equation of an ellipsoid):
    ellsurf = [rx * np.outer(np.cos(u), np.sin(v)), 
               ry * np.outer(np.sin(u), np.sin(v)), 
               rz * np.outer(np.ones_like(u), np.cos(v))]

    return [ellsurf[0] * evecs[i,0] + ellsurf[1] * evecs[i,1] + ellsurf[2] * evecs[i,2] for i in xrange(len(ellsurf))]

def func_get_cov_ellipe(cov2d, pt2d, nstd):
    from matplotlib.patches import Ellipse
    vals, vecs = np.linalg.eigh(cov2d)
    order = vals.argsort()[::-1]
    vals = vals[order]
    vecs = vecs[:,order]
    theta = np.degrees(np.arctan2(*vecs[:,0][::-1]))
    w, h = 2 * nstd * np.sqrt(vals)
    ell = Ellipse(xy=pt2d,
                  width=w, height=h,
                  angle=theta, color='black')
    ell.set_facecolor('none')
    return ell, w, h, theta

def func_propagate_project_3dcov_2dcov(cov, ptx, fc,cc,R,tw):
    P = func_get_P_from_KRt(fc, cc, R,tw)

    # Reprojection jacobian at pt
    denom = np.dot(P[2,:], np.concatenate((ptx,np.array([1.0])), axis=0))
    denom = denom ** 2

    c1 =                 P[2,1]*ptx[1] + P[2,2]*ptx[2] + P[2,3]
    c2 = P[2,0]*ptx[0] +                 P[2,2]*ptx[2] + P[2,3]
    c3 = P[2,0]*ptx[0] + P[2,1]*ptx[1]                 + P[2,3]

    Jac = np.array([[P[0,0] * c1 / denom, P[0,1] * c2 / denom, P[0,2] * c3 / denom],
                    [P[1,0] * c1 / denom, P[1,1] * c2 / denom, P[1,2] * c3 / denom]])

    cov2d = np.dot(np.dot(Jac, cov), Jac.T)
    
    return cov2d

def func_dist_kc(pt, kc, fc=None, cc=None):
    # NOTE: takes ownership of pt! Pass as ttmp.copy() if needed!
    if (cc is not None):
        pt -= cc
        pt /= fc

        
    if (kc.shape[0] > 0):

        r2 = np.sum(pt[:,0:2]**2,axis=1)

        rc = 1.0 + kc[0]*r2

        if (kc.shape[0]>1):
            rc = rc + kc[1]*r2*r2

            if (kc.shape[0]==5):
                rc = rc + kc[4]*r2*r2*r2

            if (kc.shape[0]>=3):
                dx = np.array([(2*kc[2]*pt[:,0]*pt[:,1] + kc[3]*(r2 + 2*pt[:,0]**2)),
                               (2*kc[3]*pt[:,0]*pt[:,1] + kc[2]*(r2 + 2*pt[:,1]**2))]).transpose() # tangential distortion


        pt = pt[:,0:2] * rc[:,None]
        if (kc.shape[0]>=3):
            pt += dx

        
    if (cc is not None):
        pt *= fc
        pt += cc

    return pt


def func_undist_kc(pt, kc, fc=None, cc=None):
    # NOTE: takes ownership of pt! Pass as ttmp.copy() if needed!
    if (cc is not None):
        pt -= cc
        pt /= fc

        
    if (kc.shape[0] > 0):

        r2 = np.sum(pt[:,0:2]**2,axis=1)

        rc = 1.0 + kc[0]*r2

        if (kc.shape[0]>1):
            rc = rc + kc[1]*r2*r2

            if (kc.shape[0]==5):
                rc = rc + kc[4]*r2*r2*r2

            if (kc.shape[0]>=3):
                dx = np.array([(2*kc[2]*pt[:,0]*pt[:,1] + kc[3]*(r2 + 2*pt[:,0]**2)),
                               (2*kc[3]*pt[:,0]*pt[:,1] + kc[2]*(r2 + 2*pt[:,1]**2))]).transpose() # tangential distortion


        if (kc.shape[0]>=3):
            pt -= dx
        pt = pt[:,0:2] / rc[:,None]            
        
    if (cc is not None):
        pt *= fc
        pt += cc

    return pt

    

def func_reproject(X, R, t, fc, cc, kc = None,camcenter=0):
    if (camcenter==0):
        ttmp = X-t
        ttmp = np.dot(R,ttmp.transpose()).transpose()
    else:
        ttmp = np.dot(R,X.transpose()).transpose()
        ttmp = ttmp-t

    ttmp[:,0] /= ttmp[:,2]
    ttmp[:,1] /= ttmp[:,2]

    if (kc is not None):
        ttmp = func_dist_kc(ttmp, kc)

    ttmp[:,0] = (ttmp[:,0] * fc[0]) + cc[0]
    ttmp[:,1] = (ttmp[:,1] * fc[1]) + cc[1]

    return ttmp[:,[0,1]]

def func_get_P_from_KRt(fc, cc, R, tw):
    t_c = np.dot(R, tw)[None,:].T
    K = np.identity(3)
    K[0,0] = fc[0]
    K[1,1] = fc[1]
    K[0,2] = cc[0]
    K[1,2] = cc[1]
    return np.dot(K, np.concatenate((-R, t_c), axis=1))

def func_get_Amatrix_for_triangulation(fc, cc, R_l, tw_l, x_l):
    nocams = len(x_l)
    A = np.zeros((2*nocams,4), dtype=float)
    for i in xrange(nocams):
        P = func_get_P_from_KRt(fc, cc,R_l[i],tw_l[i])
        A[2*i  , :] = x_l[i][0]*P[2,:] - P[0,:]
        A[2*i+1, :] = x_l[i][1]*P[2,:] - P[1,:]
    return A
    
    
def func_pt_triangulate_from_P_linear_eigen(fc, cc, R_l, tw_l, x_l):
    # Solve as in homogeneous case, constrain solution to |x|=1 by SVD.
    # See Hartley&Zisserman MV Geometry book
    A = func_get_Amatrix_for_triangulation(fc, cc, R_l, tw_l, x_l);
    u, s, vh = np.linalg.svd(A)
    return vh[3,0:3] / vh[3,3]


def func_pt_triangulate_from_P_linear_sq(fc, cc, R_l, tw_l, x_l, use_c_interf=False):
    if (use_c_interf==False):
        # sets W of X as 1 -> inhomogeneous case. Cannot deal with solution at inf.
        # See Hartley&Zisserman MV Geometry book
        A = func_get_Amatrix_for_triangulation(fc, cc, R_l, tw_l, x_l);
        # Assume uncorrelated observation noise
        XtX = np.linalg.inv(np.dot(A[:,0:3].T,A[:,0:3])) # Use this as covariance estimate at solution    

        ret = np.dot(np.dot(XtX, A[:,0:3].T), -A[:,3])  # Solves (X`T X)^-1 X`T y

        if (np.dot(R_l[0],(ret - tw_l[0].T))[2] < 0):
            ret *= np.nan
            XtX *= np.nan

        return ret, XtX
    else:
        # use C interface
        lib = '/home/kroegert/local/Code/AccGPSFuse/libtriang.so'
        dll = ctypes.cdll.LoadLibrary(lib)

        nopts = len(x_l)
        pt2d = np.stack(x_l, axis=1).astype(np.float32).copy()
        
        P_l = [func_get_P_from_KRt(fc, cc, R_l[x], tw_l[x]) for x in xrange(len(R_l))]
        Plin = [P_l[x].reshape(-1)  for x in xrange(len(R_l))]
        Plin = np.stack(Plin).astype(np.float32).transpose().copy() 
        
        dll.triangulate_DLT.argtypes=[ctypes.POINTER(ctypes.c_float), 
                             ctypes.POINTER(ctypes.c_float), 
                             ctypes.POINTER(ctypes.c_float), 
                             ctypes.POINTER(ctypes.c_float),
                             ctypes.c_longlong]

        pt3d_out = np.zeros((3,)).astype(np.float32);
        pt3d_cov_out = np.zeros((3,3)).astype(np.float32);
        dll.triangulate_DLT(pt3d_out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                   pt3d_cov_out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                   pt2d.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                   Plin.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                   nopts)        
        
        return pt3d_out, pt3d_cov_out

    
def func_pt_triangulate_from_P_nonlin_LM(pt3dinit, fc, cc, R_l, tw_l, x_l, noiter=10, mswitch=1, 
                            lamb_damp_init = 2.0, lamp_damp_fact = 10.0, minres = 1e-5, verbose=0, use_c_interf=False):
    if (use_c_interf==False):
        # LM Minimization of 3D point position given views

        #mswitch=1; # 0: full 3D, 1: depth-only from first frame
        #lamb_damp_init = 1e0 # initial dampening factor
        #lamp_damp_fact = 1.5 # dampening change factor
        pt3d = pt3dinit.copy()

        def func_eval_res(ptin):
            pt2dreproj = [func_reproject(ptin[:,None].T, R_l[x], tw_l[x], fc, cc) for x in xrange(len(R_l))]
            pt2dreproj = np.vstack(pt2dreproj)
            residual = (pt2d_obs - pt2dreproj)
            #print "res: ", np.mean(residual**2)
            return residual

        def func_create_point(ptin, deltap=None):
            if (mswitch==0):
                if (deltap is not None):
                    return ptin+deltap
                return ptin
            if (mswitch==1):
                ppdepthnew = np.linalg.norm(ptin-tw_l[0])
                if (deltap is not None):
                    ppdepthnew += deltap    
                return pt3d_dir*ppdepthnew + tw_l[0]


        # for depth-only minimization
        pt3d_dir = np.concatenate(((x_l[0] - cc)/fc, np.ones((1,))))
        pt3d_dir /= np.linalg.norm(pt3d_dir)
        pt3d_dir = np.dot(R_l[0].T, pt3d_dir).transpose()

        pt3d = func_create_point(pt3d)

        P_l = []
        for i in xrange(len(x_l)):
            P_l.append(func_get_P_from_KRt(fc, cc, R_l[i],tw_l[i]))    
        pt2d_obs = np.vstack(x_l)



        lamb_damp = lamb_damp_init
        residual = func_eval_res(func_create_point(pt3d))
        residual_oldsq = np.mean(residual**2)

        for iter in xrange(noiter):
            if (residual_oldsq > minres):
                if (mswitch==0):
                    Jac = np.zeros((2*len(x_l),3));
                else: 
                    Jac = np.zeros((2*len(x_l),1)); # depth only
                for i in xrange(len(x_l)):
                    if (mswitch==0):
                        # Full Reprojection jacobian at pt
                        denom = np.dot(P_l[i][2,:], np.concatenate((pt3d,np.array([1.0])), axis=0))
                        denom = denom ** 2

                        c0n0 =                       P_l[i][0,1]*pt3d[1] + P_l[i][0,2]*pt3d[2] + P_l[i][0,3]
                        c1n0 = P_l[i][0,0]*pt3d[0] +                       P_l[i][0,2]*pt3d[2] + P_l[i][0,3]
                        c2n0 = P_l[i][0,0]*pt3d[0] + P_l[i][0,1]*pt3d[1]                       + P_l[i][0,3]

                        c0n1 =                       P_l[i][1,1]*pt3d[1] + P_l[i][1,2]*pt3d[2] + P_l[i][1,3]
                        c1n1 = P_l[i][1,0]*pt3d[0] +                       P_l[i][1,2]*pt3d[2] + P_l[i][1,3]
                        c2n1 = P_l[i][1,0]*pt3d[0] + P_l[i][1,1]*pt3d[1]                       + P_l[i][1,3]

                        c0n2 =                       P_l[i][2,1]*pt3d[1] + P_l[i][2,2]*pt3d[2] + P_l[i][2,3]
                        c1n2 = P_l[i][2,0]*pt3d[0] +                       P_l[i][2,2]*pt3d[2] + P_l[i][2,3]
                        c2n2 = P_l[i][2,0]*pt3d[0] + P_l[i][2,1]*pt3d[1]                       + P_l[i][2,3]


                        jach = np.array([[(P_l[i][0,0]*c0n2 - P_l[i][2,0]*c0n0 ) / denom, 
                                          (P_l[i][0,1]*c1n2 - P_l[i][2,1]*c1n0 ) / denom, 
                                          (P_l[i][0,2]*c2n2 - P_l[i][2,2]*c2n0 ) / denom],
                                         [(P_l[i][1,0]*c0n2 - P_l[i][2,0]*c0n1 ) / denom, 
                                          (P_l[i][1,1]*c1n2 - P_l[i][2,1]*c1n1 ) / denom, 
                                          (P_l[i][1,2]*c2n2 - P_l[i][2,2]*c2n1 ) / denom]])

                    else:
                        ptd = np.linalg.norm(pt3d-tw_l[0])
                        dd = np.dot(P_l[i][2,:], np.concatenate((tw_l[0],np.array([1.0])), axis=0))
                        ee = np.dot(P_l[i][2,0:3], pt3d_dir)
                        denom = ee * ptd + dd
                        denom = denom ** 2

                        aa0 = np.dot(P_l[i][0,0:3], pt3d_dir)
                        aa1 = np.dot(P_l[i][1,0:3], pt3d_dir)

                        bb0 = np.dot(P_l[i][0,:], np.concatenate((tw_l[0],np.array([1.0])), axis=0))
                        bb1 = np.dot(P_l[i][1,:], np.concatenate((tw_l[0],np.array([1.0])), axis=0))

                        jach = np.array([[(aa0 * dd - bb0 * ee) / denom],
                                         [(aa1 * dd - bb1 * ee) / denom]])

                    Jac[(2*i):(2*i+2),:] = jach

                JacTJac = np.dot(Jac.T, Jac)
                JacTJac_diag = np.diag(np.diag(JacTJac))



                # Minim. step
                def func_compute_deltap(lamb_damp):
                    #print JacTJac
                    #print lamb_damp * JacTJac_diag
                    #print lamb_damp
                    A = JacTJac + lamb_damp * JacTJac_diag
                    Ainv = np.linalg.inv(A)
                    if (A.shape[0]==3):
                        Ainv = func_inv_sym3x3(A)
                    else:
                        Ainv = 1/A
                    delta_p = np.dot(Ainv, np.dot(Jac.T, np.ndarray.flatten(residual)))
                    return delta_p

                delta_p     = func_compute_deltap(lamb_damp)
                residual   = func_eval_res(func_create_point(pt3d,delta_p))
                residual_sq   = np.mean(residual**2)


                if (residual_sq < (residual_oldsq-minres)): 
                    lamb_damp /= lamp_damp_fact
                else:  # if residual has not decreased significantly, increase regularization
                    lamb_damp *= lamp_damp_fact

                    delta_p     = func_compute_deltap(lamb_damp)

                residual   = func_eval_res(func_create_point(pt3d,delta_p))
                residual_sq   = np.mean(residual**2)

                pt3d = func_create_point(pt3d, delta_p)

                residual_oldsq = np.mean(residual**2)

                if (verbose>0):
                    print ("New residual: " + str(residual_sq) + " " + " lamb: " + str(lamb_damp))# + 
                                            #" GT_Error: " + str(np.linalg.norm(pt-pt3d)))
            if (verbose>1):
                print func_eval_res(func_create_point(pt3d))

        return pt3d
    else: 
        
        # use C interface
        lib = '/home/kroegert/local/Code/AccGPSFuse/libtriang.so'
        dll = ctypes.cdll.LoadLibrary(lib)

        nopts = len(x_l)
        pt2d = np.stack(x_l, axis=1).astype(np.float32).copy()
        
        P_l = [func_get_P_from_KRt(fc, cc, R_l[x], tw_l[x]) for x in xrange(len(R_l))]
        Plin = [P_l[x].reshape(-1)  for x in xrange(len(R_l))]
        Plin = np.stack(Plin).astype(np.float32).transpose().copy() 
        pt3din = pt3dinit.astype(np.float32).copy();
        pt3d_cov_out = np.zeros((3,3)).astype(np.float32)
        depth_cov_out = np.zeros((1,)).astype(np.float32)

        minresidual=ctypes.c_float(minres)
        damp_init=ctypes.c_float(lamb_damp_init)
        damp_fct=ctypes.c_float(lamp_damp_fact)
        maxdamp = ctypes.c_float(1e10)


        if (mswitch==0): # full depth
            print pt3din
            dll.triangulate_full3D_LM.argtypes=[ctypes.POINTER(ctypes.c_float), 
                                 ctypes.POINTER(ctypes.c_float), 
                                 ctypes.POINTER(ctypes.c_float), 
                                 ctypes.POINTER(ctypes.c_float),
                                 ctypes.c_longlong, ctypes.c_longlong, 
                                 ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float]
            #maxdamp =  1e10
            #damp_init = 2.0
            #damp_fct = 10.0
            dll.triangulate_full3D_LM(pt3din.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                       pt3d_cov_out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                       pt2d.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                       Plin.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                       nopts, noiter, damp_init, damp_fct, minresidual, maxdamp)        
            print pt3din
            return pt3din
        else: # depth only
            dll.triangulate_depthonly.argtypes=[ctypes.POINTER(ctypes.c_float), 
                                 ctypes.POINTER(ctypes.c_float), 
                                 ctypes.POINTER(ctypes.c_float), 
                                 ctypes.POINTER(ctypes.c_float),
                                 ctypes.POINTER(ctypes.c_float),
                                 ctypes.POINTER(ctypes.c_float),
                                 ctypes.c_longlong, ctypes.c_longlong, ctypes.c_float]

            # compute viewing direction / vector from camera to 2d point
            pt3d_dir = np.concatenate(((x_l[0] - cc)/fc, np.ones((1,))))
            pt3d_dir /= np.linalg.norm(pt3d_dir)
            pt3d_dir = (np.dot(R_l[0].T, pt3d_dir).astype(np.float32)).transpose().copy()    

            dll.triangulate_depthonly(pt3din.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                       depth_cov_out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                       (tw_l[0]).ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                       pt3d_dir.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                       pt2d.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),                    
                       Plin.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                       nopts, noiter, minresidual)
            return pt3din

