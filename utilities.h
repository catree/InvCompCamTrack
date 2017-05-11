#ifndef UTIL_HEADER
#define UTIL_HEADER

#include <xmmintrin.h>
#include <immintrin.h>

#include <Eigen/Core>
#include <Eigen/LU>
#include <Eigen/Dense>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/imgproc/imgproc.hpp"

//#define SSEMULTIPL 1
#define SSEMULTIPL 4
//#define SSEMULTIPL 8

#define ALIGNMENTBYTE 32


#define LIEALG_SIGTHRESH 1e-4
#define LIEALG_EPSILON 1e-10

    
    
namespace CTR
{

#if (SSEMULTIPL==4)
  typedef __v4sf vxsf;
#elif (SSEMULTIPL==8)
  typedef float __m256 __attribute__ ((__vector_size__ (32)));
  typedef __m256 vxsf;  
#else
  typedef float vxsf;
#endif
  

//typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor, Eigen::Aligned> MatrixXfTr;
//typedef Eigen::Array<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor, Eigen::Aligned> ArrayXfTr;
typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> MatrixXfTr;
typedef Eigen::Array<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> ArrayXfTr;


typedef struct 
{
  int maxpttrack;
  int psz; // patch size
  int pszd2; // half patch size
  int pszd2m3; // 1.5 patch size - 1
  int novals;
  int lv_f;
  int lv_l;
  bool donorm;  // point cloud and pose normalization
  bool dopatchnorm ;  // patch mean normalization
  int maxiter;
  float normdp_ratio;
  int verbosity;
  
} optparam ;

void util_constructpyramide(const cv::Mat & img_ao_fmat, cv::Mat * img_ao_fmat_pyr, cv::Mat * img_ao_dx_fmat_pyr, cv::Mat * img_ao_dy_fmat_pyr, 
                            const float ** img_ao_pyr, const float ** img_ao_dx_pyr, const float ** img_ao_dy_pyr, const int lv_f, const bool getgrad, const int imgpadding);

// void util_getPatch(const float* img, const float *mid_in,  Eigen::Matrix<float, Eigen::Dynamic, 1>* tmp_in_e, const optparam* op, const int width);
// // runtime ~ 0.08 musec for one 8x8 patch,  ~ 0.04 music for 4x4 patch
// 
// 
// void util_getPatch_grad(const float* img, const float* img_dx, const float* img_dy, const float *mid_in,  Eigen::Matrix<float, Eigen::Dynamic, 1>* tmp_in_e, Eigen::Matrix<float, Eigen::Dynamic, 1>*  tmp_dx_in_e, Eigen::Matrix<float, Eigen::Dynamic, 1>* tmp_dy_in_e, const optparam* op, const int width);
// // runtime ~ 0.2 musec for one 8x8 patch, 0.11 musec for 4x4


void util_getPatch(const float* img, const float *mid_in,  Eigen::Map<MatrixXfTr, Eigen::Aligned> * tmp_in_e, const optparam* op, const int width);
// runtime ~ 0.08 musec for one 8x8 patch,  ~ 0.04 music for 4x4 patch


void util_getPatch_grad(const float* img, const float* img_dx, const float* img_dy, const float *mid_in,  Eigen::Map<MatrixXfTr, Eigen::Aligned> * tmp_in_e, Eigen::Map<MatrixXfTr, Eigen::Aligned> *  tmp_dx_in_e, Eigen::Map<MatrixXfTr, Eigen::Aligned> * tmp_dy_in_e, const optparam* op, const int width);
// runtime ~ 0.2 musec for one 8x8 patch, 0.11 musec for 4x4

//void util_getPatch_grad_only_I(const float* img, const float *mid_in,  Eigen::Map<MatrixXfTr, Eigen::Aligned> * tmp_in_e, const optparam* op, const int width);


template<typename T>
void util_SE3_coeff_to_group(T * cpos_G, const T *cpos_p) // runtime ~ .05 musec  (machine kilroy, options -O3 -msse4  -mavx )
{
  //following "Lie Groups for Computer Vision", Ethan Eade
  
  // Rotation block
  T ra1 = cpos_p[3]*cpos_p[3]; 
  T ra2 = cpos_p[4]*cpos_p[4];
  T ra3 = cpos_p[5]*cpos_p[5];
  T sig = sqrt(ra1 + ra2 + ra3);
  T sa ; // sin_sig_div_sig;
  T sb; //1mcos_div_sigsq;
  T sc; //sigm_sin_sig_div_sigsq3;
  T sigsq2 = (sig*sig);
  T sigsq3 = (sig*sig*sig);
  if (sig > LIEALG_SIGTHRESH) 
  {
    sa = sin(sig) / sig;
    sb = (1 - cos(sig)) / sigsq2;
    sc = (sig - sin(sig)) / sigsq3;
  }
  else    // sig small, approximate with taylor expansion
  {
    sa =  1 - sigsq2/6 *(1 - sigsq2/20 * (1 - sigsq2/42));
    sb = .5 * (1 - sigsq2/12 * (1 - sigsq2/30 * (1 - sigsq2/56)));
    sc =      (1 - sigsq2/20 * (1 - sigsq2/42 * (1 - sigsq2/72))) / 6;
  }
  
  //Compute I + sa * w_cross + sb * w_cross^2
  T tmp1 =   ra2 * sb;
  T tmp2 =   ra3 * sb;
  T tmp3 =   ra1 * sb;
  T tmp4 =   cpos_p[3] *  cpos_p[4] * sb;
  T tmp5 =   cpos_p[5] * sa;
  T tmp6 =   cpos_p[3] * cpos_p[5] * sb;
  T tmp7 =   cpos_p[4] * sa;
  T tmp8 =   cpos_p[3] * sa;
  T tmp9 =   cpos_p[4] * cpos_p[5] * sb;
  
  cpos_G[0] = 1-tmp1-tmp2;
  cpos_G[1] =  tmp4 - tmp5;
  cpos_G[2] =  tmp7 + tmp6;
  cpos_G[4] =  tmp5 + tmp4;
  cpos_G[5] = 1-tmp3-tmp2;
  cpos_G[6] =  tmp9 -tmp8;
  cpos_G[8] =  tmp6 -tmp7;
  cpos_G[9] =  tmp8 + tmp9;
  cpos_G[10]= 1-tmp3-tmp1;
  

  // Translation block
  tmp1 = cpos_p[5]*sb;
  tmp2 = cpos_p[3]*cpos_p[4]*sc;
  tmp3 = cpos_p[4]*sb;
  tmp4 = cpos_p[3]*cpos_p[5]*sc;
  tmp5 = cpos_p[3]*sb;
  tmp6 = cpos_p[4]*cpos_p[5]*sc;
  
  cpos_G[3] = (1-(ra2+ra3)*sc) * cpos_p[0] + (tmp2-tmp1) * cpos_p[1] + (tmp3+tmp4)* cpos_p[2];
  cpos_G[7] = (tmp1+tmp2) * cpos_p[0]       + (1-(ra1+ra3)*sc)* cpos_p[1] + (tmp6-tmp5)* cpos_p[2];
  cpos_G[11]= (tmp4-tmp3) * cpos_p[0]       + (tmp5+tmp6)* cpos_p[1] + (1-(ra1+ra2)*sc)* cpos_p[2];  
}



template<typename T>
void util_SE3_group_to_coeff(T * cpos_p, const T *cpos_G)
{
//   CORRESPONDING MATLAB CODE
// R = G(1:3,1:3);
// t = G(1:3,4);
// 
// % get omega, theta
// theta = acos(0.5 * (trace(R) - 1));
// p_out = zeros(1,6); 
// if (theta < eps)
//     omega_hat = zeros(3,3);    
//     p_out(4:6) = zeros(1,3);
// else
//     omega_hat = (theta / (2 * sin(theta)))* (R - R');    
//     p_out(4:6) = [-omega_hat(2,3) omega_hat(1,3) -omega_hat(1,2)];
// end
// 
// if (theta < 1e-5)
//   V_inv =  eye(3) - 0.5 * omega_hat + (1./12.)*(omega_hat^2);
// else
//   V_inv = (eye(3) - 0.5 * omega_hat + ( 1 - theta/(2* tan(theta/2))) / (theta^2)*(omega_hat^2) );
// end
// 
// p_out(1:3) = (V_inv * t)';


  T trace = cpos_G[0] + cpos_G[5] + cpos_G[10];
  T theta = acos(0.5f * (trace - 1));

  T omega_hat[9], omega_hat_sq[9], V_inv[9];
  memset(omega_hat   , 0   , sizeof(T) * 9);
  
  if (theta < LIEALG_EPSILON)
  {
    cpos_p[3] = 0.0f;
    cpos_p[4] = 0.0f;
    cpos_p[5] = 0.0f;
    memset(omega_hat_sq, 0   , sizeof(T) * 9);
  }
  else
  {
    T coef = theta / (2.0f * sin(theta));
    
    omega_hat[1] = coef * (cpos_G[1] - cpos_G[4]);
    omega_hat[3] = -omega_hat[1];
    omega_hat[2] = coef * (cpos_G[2] - cpos_G[8]);
    omega_hat[6] = -omega_hat[2];
    omega_hat[5] = coef * (cpos_G[6] - cpos_G[9]);
    omega_hat[7] = -omega_hat[5];

    cpos_p[3] = -omega_hat[5];
    cpos_p[4] =  omega_hat[2];
    cpos_p[5] = -omega_hat[1];

    T omsq1 = omega_hat[1]*omega_hat[1];
    T omsq2 = omega_hat[2]*omega_hat[2];
    T omsq3 = omega_hat[5]*omega_hat[5];
    
    omega_hat_sq[0] = - omsq1 - omsq2;
    omega_hat_sq[1] = -omega_hat[2]*omega_hat[5];
    omega_hat_sq[3] = omega_hat_sq[1];
    omega_hat_sq[2] = omega_hat[1]*omega_hat[5];
    omega_hat_sq[6] = omega_hat_sq[2];
    omega_hat_sq[4] = - omsq1 - omsq3;
    omega_hat_sq[5] = -omega_hat[1]*omega_hat[2];
    omega_hat_sq[7] = omega_hat_sq[5];
    omega_hat_sq[8] = - omsq2 - omsq3;
  }
  

  T theta_help;
  if (theta < LIEALG_SIGTHRESH)
    theta_help = 1.0f/12.0f;
  else
    theta_help = (1.0f - theta / (2.0f * tan(theta/2.0f))) / (theta*theta);
      
    
  V_inv[0] = 1.0f                   + theta_help * omega_hat_sq[0];
  V_inv[1] = - 0.5f * omega_hat[1]  + theta_help * omega_hat_sq[1];
  V_inv[2] = - 0.5f * omega_hat[2]  + theta_help * omega_hat_sq[2]; 
  V_inv[3] = - 0.5f * omega_hat[3]  + theta_help * omega_hat_sq[3];
  V_inv[4] = 1.0f                   + theta_help * omega_hat_sq[4];
  V_inv[5] = - 0.5f * omega_hat[5]  + theta_help * omega_hat_sq[5];
  V_inv[6] = - 0.5f * omega_hat[6]  + theta_help * omega_hat_sq[6];
  V_inv[7] = - 0.5f * omega_hat[7]  + theta_help * omega_hat_sq[7];
  V_inv[8] = 1.0f                   + theta_help * omega_hat_sq[8];


  cpos_p[0] = V_inv[0] * cpos_G[3] + V_inv[1] * cpos_G[7] + V_inv[2] * cpos_G[11];
  cpos_p[1] = V_inv[3] * cpos_G[3] + V_inv[4] * cpos_G[7] + V_inv[5] * cpos_G[11];
  cpos_p[2] = V_inv[6] * cpos_G[3] + V_inv[7] * cpos_G[7] + V_inv[8] * cpos_G[11];
}

}

#endif
