#from ctypes import *
import ctypes
import numpy as np
from func_util_geom import *


R_l = []
tw_l = []


R_l.append(np.array([[-0.339816788236272,-0.264255419756290,0.902603802098247],
[0.689228545528361,-0.722964090039945,0.047821925322753],
[0.639912933500835,0.638350998778996,0.427807713694147]]).astype(np.float32))
 
R_l.append(np.array([[-0.868327303059270,0.368689658667113,0.331776476492446],
[-0.309406694521653,-0.925453868621726,0.218638135827824],
[0.387653443357594,0.087195599917712,0.917671910438579]]).astype(np.float32))
 
R_l.append(np.array([[-0.014529722569767,0.552071591719939,-0.833670105484093],
[-0.959699399200942,0.226321385728413,0.166600400765790],
[0.280652721945462,0.802493356968009,0.526534387943294]]).astype(np.float32))

            
tw_l.append(np.array([2.547559654713416, 2.177420241434722, 5.881176644306613]).astype(np.float32))
tw_l.append(np.array([7.899525346172922, 9.879207105887073,4.264116138330023]).astype(np.float32))
tw_l.append(np.array([7.808409612734156, 1.561875007909654, 5.595205606117695]).astype(np.float32))
            
ptgt = np.array([10.957771555771460, 10.567104034216499,11.503743929189053]).astype(np.float32)

fc = np.array([2000,2000]).astype(np.float32)
cc = np.array([0,0]).astype(np.float32)

P_l = [func_get_P_from_KRt(fc, cc, R_l[x], tw_l[x]) for x in xrange(len(R_l))]
Plin = [P_l[x].reshape(-1)  for x in xrange(len(R_l))]
Plin = np.stack(Plin).astype(np.float32).transpose().copy() 
pt2d = np.zeros((2,3)).astype(np.float32)
x_l = [pt2d[:,x] for x in xrange(pt2d.shape[1])]

pt3d = (ptgt.copy() + 0.1).astype(np.float32)

nopts = len(R_l)

# order contiguous by observation, not elements of P or pt2d
for i in xrange(3):
#    print func_reproject(pt3d[:,None].T, R_l[i], tw_l[i].T, fc, cc, camcenter=0)[0,:]
    ptreproj = np.dot(P_l[i][0:3,0:3], pt3d[:,None]).T + P_l[i][:,3]
    ptreproj = ptreproj[0,:] / ptreproj[0,2]
    #print ptreproj[0:2]
    
#print np.linalg.det(R_l[0])
#print func_get_P_from_KRt(fc, cc, R_l[0], tw_l[0])
#print pt2d

noiter=15
minresidual=ctypes.c_float(1e-5)
damp_init=ctypes.c_float(1e0)
damp_fct=ctypes.c_float(1.5)
maxdamp =  ctypes.c_float(1e10)

lib = '/home/kroegert/local/Code/AccGPSFuse/libtriang.so'
dll = ctypes.cdll.LoadLibrary(lib)



print "DLT TRIANGULATION"
#dll.triangulate_DLT.argtypes=[ctypes.POINTER(ctypes.c_float), 
#                     ctypes.POINTER(ctypes.c_float), 
#                     ctypes.POINTER(ctypes.c_float), 
#                     ctypes.POINTER(ctypes.c_float),
#                     ctypes.c_longlong]
#
#pt3d_out = np.zeros_like(pt3d).astype(np.float32);
#pt3d_cov_out = np.zeros((3,3)).astype(np.float32);
#dll.triangulate_DLT(pt3d_out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
#           pt3d_cov_out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
#           pt2d.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
#           Plin.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
#           nopts)
#print pt3d_out
print (func_pt_triangulate_from_P_linear_sq(fc, cc, R_l, tw_l, x_l, use_c_interf=False)[0] -
       func_pt_triangulate_from_P_linear_sq(fc, cc, R_l, tw_l, x_l, use_c_interf=True)[0])
print (func_pt_triangulate_from_P_linear_sq(fc, cc, R_l, tw_l, x_l, use_c_interf=False)[1] -
       func_pt_triangulate_from_P_linear_sq(fc, cc, R_l, tw_l, x_l, use_c_interf=True)[1])


print "FULL TRIANGULATION - GaussNewton"
dll.triangulate_full3D.argtypes=[ctypes.POINTER(ctypes.c_float), 
                     ctypes.POINTER(ctypes.c_float), 
                     ctypes.POINTER(ctypes.c_float), 
                     ctypes.POINTER(ctypes.c_float),
                     ctypes.c_longlong, ctypes.c_longlong, ctypes.c_float]

pt3din = pt3d.copy();
pt3d_cov_out = np.zeros((3,3)).astype(np.float32)
dll.triangulate_full3D(pt3din.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
           pt3d_cov_out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
           pt2d.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
           Plin.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
           nopts, noiter, minresidual)
#print pt3d_cov_out




print "FULL TRIANGULATION - LM"

#dll.triangulate_full3D_LM.argtypes=[ctypes.POINTER(ctypes.c_float), 
#                     ctypes.POINTER(ctypes.c_float), 
#                     ctypes.POINTER(ctypes.c_float), 
#                     ctypes.POINTER(ctypes.c_float),
#                     ctypes.c_longlong, ctypes.c_longlong, 
#                     ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float]
#pt3din = pt3d.copy();
#pt3d_cov_out = np.zeros((3,3)).astype(np.float32)
#dll.triangulate_full3D_LM(pt3din.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
#           pt3d_cov_out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
#           pt2d.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
#           Plin.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
#           nopts, noiter, damp_init, damp_fct, minresidual, maxdamp)
#print "LM cov: ", pt3d_cov_out

minresidual=1e-5
damp_init=2.0
damp_fct=10.0
maxdamp = 1e10


print func_pt_triangulate_from_P_nonlin_LM(pt3d.copy(), fc, cc, R_l, tw_l, x_l, 8, 0, verbose=1, lamb_damp_init = damp_init, lamp_damp_fact = damp_fct, use_c_interf=False) - func_pt_triangulate_from_P_nonlin_LM(pt3d.copy(), fc, cc, R_l, tw_l, x_l, 8, 0, verbose=1, lamb_damp_init = damp_init, lamp_damp_fact = damp_fct, use_c_interf=True)




print "DEPTH-ONLY TRIANGULATION - GaussNewton"
#dll.triangulate_depthonly.argtypes=[ctypes.POINTER(ctypes.c_float), 
#                     ctypes.POINTER(ctypes.c_float), 
#                     ctypes.POINTER(ctypes.c_float), 
#                     ctypes.POINTER(ctypes.c_float),
#                     ctypes.POINTER(ctypes.c_float),
#                     ctypes.POINTER(ctypes.c_float),
#                     ctypes.c_longlong, ctypes.c_longlong, ctypes.c_float]
#
# # compute viewing direction / vector from camera to 2d point
#pt3d_dir = np.concatenate(((x_l[0] - cc)/fc, np.ones((1,))))
#pt3d_dir /= np.linalg.norm(pt3d_dir)
#pt3d_dir = (np.dot(R_l[0].T, pt3d_dir).astype(np.float32)).transpose().copy()    
#
#depth_cov_out = np.zeros((1,)).astype(np.float32)
#pt3din = pt3d.copy();
#dll.triangulate_depthonly(pt3din.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
#           depth_cov_out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
#           (tw_l[0]).ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
#           pt3d_dir.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
#           pt2d.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),                    
#           Plin.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
#           nopts, noiter, minresidual)
#print depth_cov_out

print func_pt_triangulate_from_P_nonlin_LM(pt3d.copy(), fc, cc, R_l, tw_l, x_l, 8, 1, verbose=1, lamb_damp_init = damp_init, lamp_damp_fact = damp_fct, use_c_interf=False) - func_pt_triangulate_from_P_nonlin_LM(pt3d.copy(), fc, cc, R_l, tw_l, x_l, 8, 1, verbose=1, lamb_damp_init = damp_init, lamp_damp_fact = damp_fct, use_c_interf=True)






#gcc -fpic -c triang.c -o libtriang.o && gcc -shared -o libtriang.so libtriang.o && python run_ctypestest.py
