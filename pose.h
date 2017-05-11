#ifndef POSE_HEADER
#define POSE_HEADER

#include <xmmintrin.h>
#include <immintrin.h>

#include "utilities.h"
#include "camera.h"

namespace CTR
{

  // holds, converts and normalizes poses, reprojects
  
class PoseClass
{
  
public:
  PoseClass(const CamClass * camobj_in, const optparam* op_in);
  
  ~PoseClass();
  
  void setpose_se3(const double *p_in, const Eigen::Vector3d meanshift_in, const double varval_in); // sets the pose, normalized if necessary (if requested)
  
  void addpose_se3(const float *p_in);
  void subpose_se3(const float *p_in);

  // TODO: return pose, normalized if necessary
  
  //inline const float * getPose_SE3() const {return cpos_G;};
  //inline const float * getPose_se3() const {return cpos_p;};
  
  void getPose_se3(double * p_out) const;
  
  void project_pt(const float *pt3d, float *pt2d, int nopoints, int sc) const; // project points, save only projected 2D points, // runtime ~ 0.095  musec for 100 points (machine kilroy, options -O3 -msse4  -mavx )
  void project_pt_save_rotated(const float *pt3d, float *pt3d_rot, float *pt2d, int nopoints, int sc) const; // same as project_pt(), but save rotated 3D points as well (separated for speed reasons),  // runtime ~ 0.12  musec for 100 points (machine kilroy, options -O3 -msse4  -mavx )

  //inline float getsw(int sc) const {return camobj->getsw[sc]; };

  const CamClass * camobj;

private:

//     const float sigthresh_f = 1e-4;
//     const float epsilon_f = 1e-10;
//     const double sigthresh_d = 1e-4;
//     const double epsilon_d = 1e-10;

//     void setpose_SE3(const float *G_in); // TODO: do I need this ?
      
    Eigen::Vector3d meanshift;
    double varval;    
      
    float cpos_G[3*4]; //current 6-DOF camera position, row major storage, SE3 group element
    float cpos_p[6];   //current 6-DOF camera position, SE3 group coefficients
    
//     void SE3_coeff_to_group();  // se3 -> SE3 , closed form without matrix exponential, // runtime ~ .05 musec  (machine kilroy, options -O3 -msse4  -mavx )
//     void SE3_group_to_coeff();  // SE3 -> se3 , closed form without matrix logarithm
    
    const optparam* op;
};
















}


#endif
