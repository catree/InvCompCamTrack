
import numpy as np

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
  

class oftrack:
  def __init__(self, bsize, imwidth, imheight, 
               th_flowvalid_ratio = .2, th_flowvalid_abs = 1):
    self.bsize = bsize
    self.imwidth = imwidth
    self.imheight = imheight
    self.th_flowvalid_ratio = th_flowvalid_ratio # max ratio: error vecor magnitude / displacement vector magnitude
    self.th_flowvalid_abs = th_flowvalid_abs # max absolute error in pixels

    self.tracks = []
    self.tracks_valid = []
    self.tracks_absmovement = []
    self.frcounter = 0
    
  def addframe(self, of_forw, of_back, corners=None):

    self.frcounter +=1

    # start new tracks
    if (corners is not None):
      xy_t = np.NaN*np.ones((corners.shape[0],2,self.bsize), dtype=np.float32)
      xy_t[: , : ,0] = corners    
      self.tracks.append(xy_t)
      self.tracks_valid.append(np.ones((corners.shape[0],), dtype=bool))
      self.tracks_absmovement.append(np.zeros((corners.shape[0],), dtype=float))
    else:
      self.tracks.append(None)
      self.tracks_valid.append(None)
      self.tracks_absmovement.append(None)


    # update tracks
    for fri in xrange(np.maximum(self.frcounter-self.bsize+1, 0), self.frcounter):
      
      if (self.tracks[fri] is not None):
  
        idxwasvalid = np.where(self.tracks_valid[fri])[0]
        
        frb = self.frcounter-fri - 1
        #print fri, frb
        
        xy_l = self.tracks[fri][idxwasvalid, : , frb]
        
        xy_lf = func_get_transf_position(xy_l, of_forw[:,:,0], of_forw[:,:,1])

        # forward-backward check
        xy_l_verif_f = func_get_transf_position(xy_lf, of_back[:,:,0], of_back[:,:,1])

        idxvalid = ((np.linalg.norm(xy_l -xy_l_verif_f , axis=1)/np.linalg.norm(xy_l -xy_lf, axis=1) < self.th_flowvalid_ratio) & 
                    (np.linalg.norm(xy_l -xy_l_verif_f , axis=1) < self.th_flowvalid_abs))
        idxinvalid  = np.where(~idxvalid)[0]
        #idxvalid = np.where(idxvalid)[0]

        xy_lf[idxinvalid,:] = np.NaN
        self.tracks[fri][idxwasvalid , : ,frb+1] = xy_lf
        self.tracks_absmovement[fri][idxwasvalid] = np.linalg.norm(self.tracks[fri][idxwasvalid , : ,0] - xy_lf, axis=1)
        self.tracks_valid[fri][idxwasvalid] = np.any((~np.isnan(xy_lf)), axis=1)

    # free memory tracks
    for fri in xrange(0, self.frcounter-self.bsize+1):
      idxwasvalid = np.where(self.tracks_valid[fri])[0]
      self.tracks[fri] = self.tracks[fri][idxwasvalid, : , :]
      self.tracks_valid[fri] = np.ones((self.tracks[fri].shape[0],), dtype=bool)
      self.tracks_absmovement[fri] = self.tracks_absmovement[fri][idxwasvalid]
    
  
  def getpttransfer(self,fr = None, th_min_movement = -1.0): #get points transfered from fr-1 to fr
    if (fr is None):
      fr = self.frcounter
    # th_min_movement : min movement of particle before disregarded


    pttranfs = []
    
    for fri in xrange(np.maximum(fr-self.bsize+1, 0), fr):
      
      if (self.tracks[fri] is not None):
        #print fri
        
        #print self.tracks_valid[fri].shape
        idxwasvalid = np.where(self.tracks_valid[fri] & (self.tracks_absmovement[fri] > th_min_movement))[0]
        #print idxwasvalid
        
        frb = fr - fri - 1
        
        pttranfs.append(self.tracks[fri][idxwasvalid, : , (frb):(frb+2)])
        
        #print fri, frb
        
      #print pttranfs[-1].shape

    pttranfs = np.vstack(pttranfs)
    
    return pttranfs
        
        
  def savetofile(self, filename):
    np.savez_compressed(filename, x=self.tracks)


      
    
      
      
      
      
    
    
    
    
    
