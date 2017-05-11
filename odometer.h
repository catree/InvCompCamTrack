#ifndef ODOMETER_HEADER
#define ODOMETER_HEADER

#include <xmmintrin.h>
#include <immintrin.h>

#include "utilities.h"
#include "pose.h"
#include "camera.h"

namespace CTR
{

// TODO: accept all inout as double, convert to float    
// TODO: Zero-norm and unit variance for 3d points, and undo at the end
// TODO: select max. feature, evenly spread over image, select from top (expect ordered list)
  
class OdometerClass
{
  
public:
  OdometerClass(PoseClass * pose_in, const optparam* op_in);
  ~OdometerClass();

  void Set3Dpoints(double *pt_in, const int nopoints_in);
  void SetPose(const double * p_in, const float **img_ref_in, const float **img_ref_dx_in, const float **img_ref_dy_in, const float **img_new_in);
  void TrackPose(double * p_out);
  

  inline const float* Get2DPoints() const {return pt2d[op->lv_l];} 

//  void Reproject? < Write function which returns reprojected 2D points at current state (only valid after reprojection)
//  Split TrackPose in initialization (pass 3d points, normalization) and tracking (pose tracking), allowing for 2D reprojection in between
  
  // give points as float
  
  // set images
  
  // give pose as lie algebra
  
  // return pose as lie algebra
  
  // TODO: make pose datatype to double

private:

    void ResetOdometer();
    void ComputeHessian();
    void SolveLinSystem();
    
    // normalization parameters
    Eigen::Vector3d meanshift;
    double varval;
    
    const float **img_ref;
    const float **img_ref_dx;
    const float **img_ref_dy;
    const float **img_new;
    
    int nopoints;
    
    Eigen::Matrix<float, 6, 6> Hes; // Hessian for optimization
    Eigen::Matrix<float, 6, 1> sumsd, delta_p;
    
    Eigen::Matrix<bool, Eigen::Dynamic, 1> ind_ref, ind_new;
    
    
    float *pt3d, *pt3d_ref, **pt2d, *pt2d_new;
  
    float *pat_ref_all, *pat_ref_dx_all, *pat_ref_dy_all, *pat_new_all; // all aligned, continuous memory for patch data
    float *sd1_all, *sd2_all, *sd3_all, *sd4_all, *sd5_all, *sd6_all; // all aligned, continuous memory for steepest descent images
    float *sd1_proj_all, *sd2_proj_all, *sd3_proj_all, *sd4_proj_all, *sd5_proj_all, *sd6_proj_all; // all aligned, continuous memory for patch error imagesprojected on steepest descent images
    
    ArrayXfTr pdiff;
    
    std::vector<Eigen::Map<MatrixXfTr, Eigen::Aligned>> pat_ref_i;
    std::vector<Eigen::Map<MatrixXfTr, Eigen::Aligned>> pat_ref_dx_i;
    std::vector<Eigen::Map<MatrixXfTr, Eigen::Aligned>> pat_ref_dy_i;
    std::vector<Eigen::Map<MatrixXfTr, Eigen::Aligned>> pat_new_i;
    std::vector<Eigen::Map<ArrayXfTr, Eigen::Aligned>> sd1_i;
    std::vector<Eigen::Map<ArrayXfTr, Eigen::Aligned>> sd2_i;
    std::vector<Eigen::Map<ArrayXfTr, Eigen::Aligned>> sd3_i;
    std::vector<Eigen::Map<ArrayXfTr, Eigen::Aligned>> sd4_i;
    std::vector<Eigen::Map<ArrayXfTr, Eigen::Aligned>> sd5_i;
    std::vector<Eigen::Map<ArrayXfTr, Eigen::Aligned>> sd6_i;

    Eigen::Map<Eigen::MatrixXf, Eigen::Aligned> * sd1_a;
    Eigen::Map<Eigen::MatrixXf, Eigen::Aligned> * sd2_a;
    Eigen::Map<Eigen::MatrixXf, Eigen::Aligned> * sd3_a;
    Eigen::Map<Eigen::MatrixXf, Eigen::Aligned> * sd4_a;
    Eigen::Map<Eigen::MatrixXf, Eigen::Aligned> * sd5_a;
    Eigen::Map<Eigen::MatrixXf, Eigen::Aligned> * sd6_a;

    std::vector<Eigen::Map<ArrayXfTr, Eigen::Aligned>> sd1_proj_i;
    std::vector<Eigen::Map<ArrayXfTr, Eigen::Aligned>> sd2_proj_i;
    std::vector<Eigen::Map<ArrayXfTr, Eigen::Aligned>> sd3_proj_i;
    std::vector<Eigen::Map<ArrayXfTr, Eigen::Aligned>> sd4_proj_i;
    std::vector<Eigen::Map<ArrayXfTr, Eigen::Aligned>> sd5_proj_i;
    std::vector<Eigen::Map<ArrayXfTr, Eigen::Aligned>> sd6_proj_i;

    Eigen::Map<Eigen::MatrixXf, Eigen::Aligned> * sd1_proj_a;
    Eigen::Map<Eigen::MatrixXf, Eigen::Aligned> * sd2_proj_a;
    Eigen::Map<Eigen::MatrixXf, Eigen::Aligned> * sd3_proj_a;
    Eigen::Map<Eigen::MatrixXf, Eigen::Aligned> * sd4_proj_a;
    Eigen::Map<Eigen::MatrixXf, Eigen::Aligned> * sd5_proj_a;
    Eigen::Map<Eigen::MatrixXf, Eigen::Aligned> * sd6_proj_a;
    
    
//     Eigen::Map<Eigen::MatrixXf, Eigen::Aligned> * pat_ref_a;
//     Eigen::Map<Eigen::MatrixXf, Eigen::Aligned> * pat_ref_dx_a;
//     Eigen::Map<Eigen::MatrixXf, Eigen::Aligned> * pat_ref_dy_a;
//     Eigen::Map<Eigen::MatrixXf, Eigen::Aligned> * pat_new_a;

    PoseClass * pose;
    const optparam* op;
};
















}


#endif
