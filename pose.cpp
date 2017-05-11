#include "pose.h"

#include <iostream>
#include <cstring>
#include <math.h>

using std::cout;
using std::endl;


namespace CTR
{
  
PoseClass::PoseClass(const CamClass * camobj_in,  const optparam* op_in)    : 
  camobj(camobj_in),
  op(op_in)
{
}

PoseClass::~PoseClass()
{
}


void PoseClass::setpose_se3(const double *p_in, const Eigen::Vector3d meanshift_in, const double varval_in)
{
  double p_in_norm[6];
  memcpy(p_in_norm, p_in, sizeof(double) * 6);
  
  // Normalize pose: subtract shift vector from translation, divide by variance
  if (op->donorm)
  {
    varval = varval_in;
    meanshift = meanshift_in;
    
    double G_in_norm[3*4];
    util_SE3_coeff_to_group(G_in_norm, p_in_norm);
          
    // Matlab code
    //t = -R_in'*G_init(1:3,4);
    //t = (t - meanshift)/varval;
    //G_init(1:3,4) = -R_in*t;
    
    Eigen::Vector3d ttmp;
    ttmp[0] = -G_in_norm[0]*G_in_norm[3] -G_in_norm[4]*G_in_norm[7] -G_in_norm[8] *G_in_norm[11];
    ttmp[1] = -G_in_norm[1]*G_in_norm[3] -G_in_norm[5]*G_in_norm[7] -G_in_norm[9] *G_in_norm[11];
    ttmp[2] = -G_in_norm[2]*G_in_norm[3] -G_in_norm[6]*G_in_norm[7] -G_in_norm[10]*G_in_norm[11];
    
    
    ttmp[0] = (ttmp[0]-meanshift[0])/varval;
    ttmp[1] = (ttmp[1]-meanshift[1])/varval;
    ttmp[2] = (ttmp[2]-meanshift[2])/varval;

    G_in_norm[3] = -G_in_norm[0]*ttmp[0] -G_in_norm[1]*ttmp[1] -G_in_norm[2] *ttmp[2];
    G_in_norm[7] = -G_in_norm[4]*ttmp[0] -G_in_norm[5]*ttmp[1] -G_in_norm[6] *ttmp[2];
    G_in_norm[11]= -G_in_norm[8]*ttmp[0] -G_in_norm[9]*ttmp[1] -G_in_norm[10]*ttmp[2];

    // convert back
    util_SE3_group_to_coeff(p_in_norm, G_in_norm);
    
  }
  
    
  //memset(cpos_p, 0   , sizeof(float) * 6);
  //memset(cpos_G, 0   , sizeof(float) * 3*4);
    
  cpos_p[0] = static_cast<float> (p_in_norm[0]);
  cpos_p[1] = static_cast<float> (p_in_norm[1]);
  cpos_p[2] = static_cast<float> (p_in_norm[2]);
  cpos_p[3] = static_cast<float> (p_in_norm[3]);
  cpos_p[4] = static_cast<float> (p_in_norm[4]);
  cpos_p[5] = static_cast<float> (p_in_norm[5]);
  
  //SE3_coeff_to_group();
  util_SE3_coeff_to_group(cpos_G, cpos_p);
}


void PoseClass::getPose_se3(double * p_out) const
{
  float p_out_unnorm[6];
  memcpy(p_out_unnorm, cpos_p, sizeof(float) * 6);

  // Unnormalize pose: multiply by variance, add shift vector from translation
  if (op->donorm)
  {
    float G_out_unnorm[3*4];
    memcpy(G_out_unnorm, cpos_G, sizeof(double) * 6);
    
    Eigen::Vector3d ttmp;
    ttmp[0] = -G_out_unnorm[0]*G_out_unnorm[3] -G_out_unnorm[4]*G_out_unnorm[7] -G_out_unnorm[8] *G_out_unnorm[11];
    ttmp[1] = -G_out_unnorm[1]*G_out_unnorm[3] -G_out_unnorm[5]*G_out_unnorm[7] -G_out_unnorm[9] *G_out_unnorm[11];
    ttmp[2] = -G_out_unnorm[2]*G_out_unnorm[3] -G_out_unnorm[6]*G_out_unnorm[7] -G_out_unnorm[10]*G_out_unnorm[11];
    
    ttmp[0] = ttmp[0]*varval+meanshift[0];
    ttmp[1] = ttmp[1]*varval+meanshift[1];
    ttmp[2] = ttmp[2]*varval+meanshift[2];

    G_out_unnorm[3] = -G_out_unnorm[0]*ttmp[0] -G_out_unnorm[1]*ttmp[1] -G_out_unnorm[2] *ttmp[2];
    G_out_unnorm[7] = -G_out_unnorm[4]*ttmp[0] -G_out_unnorm[5]*ttmp[1] -G_out_unnorm[6] *ttmp[2];
    G_out_unnorm[11]= -G_out_unnorm[8]*ttmp[0] -G_out_unnorm[9]*ttmp[1] -G_out_unnorm[10]*ttmp[2];

    // convert back
    util_SE3_group_to_coeff(p_out_unnorm, G_out_unnorm);
  }
  
  p_out[0] = static_cast<double> (p_out_unnorm[0]);
  p_out[1] = static_cast<double> (p_out_unnorm[1]);
  p_out[2] = static_cast<double> (p_out_unnorm[2]);
  p_out[3] = static_cast<double> (p_out_unnorm[3]);
  p_out[4] = static_cast<double> (p_out_unnorm[4]);
  p_out[5] = static_cast<double> (p_out_unnorm[5]);  
}


void PoseClass::addpose_se3(const float *p_in)
{
  cpos_p[0] += p_in[0];
  cpos_p[1] += p_in[1];
  cpos_p[2] += p_in[2];
  cpos_p[3] += p_in[3];
  cpos_p[4] += p_in[4];
  cpos_p[5] += p_in[5];
  
  util_SE3_coeff_to_group(cpos_G, cpos_p);
  
  //cout << cpos_p[0] << ", " << cpos_p[1] << ", " << cpos_p[2] << ", " << cpos_p[3] << ", " << cpos_p[4] << ", " << cpos_p[5] << ", " << cpos_p[6] << endl;

}

void PoseClass::subpose_se3(const float *p_in)
{
  cpos_p[0] -= p_in[0];
  cpos_p[1] -= p_in[1];
  cpos_p[2] -= p_in[2];
  cpos_p[3] -= p_in[3];
  cpos_p[4] -= p_in[4];
  cpos_p[5] -= p_in[5];
  
  util_SE3_coeff_to_group(cpos_G, cpos_p);
  //util_SE3_group_to_coeff(cpos_p, cpos_G)
}

// void PoseClass::SE3_group_to_coeff()
// {
//   
// /*  CORRESPONDING MATLAB CODE
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
// p_out(1:3) = (V_inv * t)';*/
// 
// 
// 
// 
//   float trace = cpos_G[0] + cpos_G[5] + cpos_G[10];
//   float theta = acos(0.5f * (trace - 1));
// 
//   float omega_hat[9], omega_hat_sq[9], V_inv[9];
//   memset(omega_hat   , 0   , sizeof(float) * 9);
//   
//   if (theta < epsilon_f)
//   {
//     cpos_p[3] = 0.0f;
//     cpos_p[4] = 0.0f;
//     cpos_p[5] = 0.0f;
//     memset(omega_hat_sq, 0   , sizeof(float) * 9);
//   }
//   else
//   {
//     float coef = theta / (2.0f * sin(theta));
//     
//     omega_hat[1] = coef * (cpos_G[1] - cpos_G[4]);
//     omega_hat[3] = -omega_hat[1];
//     omega_hat[2] = coef * (cpos_G[2] - cpos_G[8]);
//     omega_hat[6] = -omega_hat[2];
//     omega_hat[5] = coef * (cpos_G[6] - cpos_G[9]);
//     omega_hat[7] = -omega_hat[5];
// 
//     cpos_p[3] = -omega_hat[5];
//     cpos_p[4] =  omega_hat[2];
//     cpos_p[5] = -omega_hat[1];
// 
//     float omsq1 = omega_hat[1]*omega_hat[1];
//     float omsq2 = omega_hat[2]*omega_hat[2];
//     float omsq3 = omega_hat[5]*omega_hat[5];
//     
//     omega_hat_sq[0] = - omsq1 - omsq2;
//     omega_hat_sq[1] = -omega_hat[2]*omega_hat[5];
//     omega_hat_sq[3] = omega_hat_sq[1];
//     omega_hat_sq[2] = omega_hat[1]*omega_hat[5];
//     omega_hat_sq[6] = omega_hat_sq[2];
//     omega_hat_sq[4] = - omsq1 - omsq3;
//     omega_hat_sq[5] = -omega_hat[1]*omega_hat[2];
//     omega_hat_sq[7] = omega_hat_sq[5];
//     omega_hat_sq[8] = - omsq2 - omsq3;
//   }
//   
// 
//   float theta_help;
//   if (theta < sigthresh_f)
//     theta_help = 1.0f/12.0f;
//   else
//     theta_help = (1.0f - theta / (2.0f * tan(theta/2.0f))) / (theta*theta);
//       
//     
//   V_inv[0] = 1.0f                   + theta_help * omega_hat_sq[0];
//   V_inv[1] = - 0.5f * omega_hat[1]  + theta_help * omega_hat_sq[1];
//   V_inv[2] = - 0.5f * omega_hat[2]  + theta_help * omega_hat_sq[2]; 
//   V_inv[3] = - 0.5f * omega_hat[3]  + theta_help * omega_hat_sq[3];
//   V_inv[4] = 1.0f                   + theta_help * omega_hat_sq[4];
//   V_inv[5] = - 0.5f * omega_hat[5]  + theta_help * omega_hat_sq[5];
//   V_inv[6] = - 0.5f * omega_hat[6]  + theta_help * omega_hat_sq[6];
//   V_inv[7] = - 0.5f * omega_hat[7]  + theta_help * omega_hat_sq[7];
//   V_inv[8] = 1.0f                   + theta_help * omega_hat_sq[8];
// 
// 
//   cpos_p[0] = V_inv[0] * cpos_G[3] + V_inv[1] * cpos_G[7] + V_inv[2] * cpos_G[11];
//   cpos_p[1] = V_inv[3] * cpos_G[3] + V_inv[4] * cpos_G[7] + V_inv[5] * cpos_G[11];
//   cpos_p[2] = V_inv[6] * cpos_G[3] + V_inv[7] * cpos_G[7] + V_inv[8] * cpos_G[11];
//   
// }

// void PoseClass::SE3_coeff_to_group()
// {
//   //following "Lie Groups for Computer Vision", Ethan Eade
// 
//   //std::memset(cpos_G, 12, sizeof(float));
//   
//   // Rotation block
//   float ra1 = cpos_p[3]*cpos_p[3]; 
//   float ra2 = cpos_p[4]*cpos_p[4];
//   float ra3 = cpos_p[5]*cpos_p[5];
//   float sig = sqrt(ra1 + ra2 + ra3);
//   float sa ; // sin_sig_div_sig;
//   float sb; //1mcos_div_sigsq;
//   float sc; //sigm_sin_sig_div_sigsq3;
//   float sigsq2 = (sig*sig);
//   float sigsq3 = (sig*sig*sig);
//   if (sig > sigthresh_f) 
//   {
//     sa = sin(sig) / sig;
//     sb = (1 - cos(sig)) / sigsq2;
//     sc = (sig - sin(sig)) / sigsq3;
//   }
//   else    // sig small, approximate with taylor expansion
//   {
//     sa =  1 - sigsq2/6 *(1 - sigsq2/20 * (1 - sigsq2/42));
//     sb = .5 * (1 - sigsq2/12 * (1 - sigsq2/30 * (1 - sigsq2/56)));
//     sc =      (1 - sigsq2/20 * (1 - sigsq2/42 * (1 - sigsq2/72))) / 6;
//   }
//   
//   //Compute I + sa * w_cross + sb * w_cross^2
//   float tmp1 =   ra2 * sb;
//   float tmp2 =   ra3 * sb;
//   float tmp3 =   ra1 * sb;
//   float tmp4 =   cpos_p[3] *  cpos_p[4] * sb;
//   float tmp5 =   cpos_p[5] * sa;
//   float tmp6 =   cpos_p[3] * cpos_p[5] * sb;
//   float tmp7 =   cpos_p[4] * sa;
//   float tmp8 =   cpos_p[3] * sa;
//   float tmp9 =   cpos_p[4] * cpos_p[5] * sb;
//   
//   cpos_G[0] = 1-tmp1-tmp2;
//   cpos_G[1] =  tmp4 - tmp5;
//   cpos_G[2] =  tmp7 + tmp6;
//   cpos_G[4] =  tmp5 + tmp4;
//   cpos_G[5] = 1-tmp3-tmp2;
//   cpos_G[6] =  tmp9 -tmp8;
//   cpos_G[8] =  tmp6 -tmp7;
//   cpos_G[9] =  tmp8 + tmp9;
//   cpos_G[10]= 1-tmp3-tmp1;
//   
// 
//   // Translation block
//   tmp1 = cpos_p[5]*sb;
//   tmp2 = cpos_p[3]*cpos_p[4]*sc;
//   tmp3 = cpos_p[4]*sb;
//   tmp4 = cpos_p[3]*cpos_p[5]*sc;
//   tmp5 = cpos_p[3]*sb;
//   tmp6 = cpos_p[4]*cpos_p[5]*sc;
//   
//   cpos_G[3] = (1-(ra2+ra3)*sc) * cpos_p[0] + (tmp2-tmp1) * cpos_p[1] + (tmp3+tmp4)* cpos_p[2];
//   cpos_G[7] = (tmp1+tmp2) * cpos_p[0]       + (1-(ra1+ra3)*sc)* cpos_p[1] + (tmp6-tmp5)* cpos_p[2];
//   cpos_G[11]= (tmp4-tmp3) * cpos_p[0]       + (tmp5+tmp6)* cpos_p[1] + (1-(ra1+ra2)*sc)* cpos_p[2];
// }



void PoseClass::project_pt(const float *pt3d, float *pt2d, int nopoints, int sc) const
{
  float fx_s = camobj->getfx(sc);
  float fy_s = camobj->getfy(sc);
  float cx_s = camobj->getcx(sc);
  float cy_s = camobj->getcy(sc);
  
  vxsf * X = (vxsf*) (pt3d);
  vxsf * Y = (vxsf*) (pt3d + op->maxpttrack  );
  vxsf * Z = (vxsf*) (pt3d + op->maxpttrack*2);

  vxsf * x = (vxsf*) (pt2d);
  vxsf * y = (vxsf*) (pt2d + op->maxpttrack  );
  
  vxsf tmp3d_X;
  vxsf tmp3d_Y;
  vxsf tmp3d_Z;
  
  
#if (SSEMULTIPL>1)
  int div = nopoints % SSEMULTIPL;
  if (div>0) 
    nopoints = nopoints + (SSEMULTIPL - div);
  
  #if (SSEMULTIPL == 4)
  vxsf fx = (vxsf) {fx_s, fx_s, fx_s, fx_s};
  vxsf fy = (vxsf) {fy_s, fy_s, fy_s, fy_s};
  vxsf cx = (vxsf) {cx_s, cx_s, cx_s, cx_s};
  vxsf cy = (vxsf) {cy_s, cy_s, cy_s, cy_s};
  
  vxsf cpos0 = (vxsf) {cpos_G[0], cpos_G[0], cpos_G[0], cpos_G[0]};
  vxsf cpos1 = (vxsf) {cpos_G[1], cpos_G[1], cpos_G[1], cpos_G[1]};
  vxsf cpos2 = (vxsf) {cpos_G[2], cpos_G[2], cpos_G[2], cpos_G[2]};
  vxsf cpos3 = (vxsf) {cpos_G[3], cpos_G[3], cpos_G[3], cpos_G[3]};
  vxsf cpos4 = (vxsf) {cpos_G[4], cpos_G[4], cpos_G[4], cpos_G[4]};
  vxsf cpos5 = (vxsf) {cpos_G[5], cpos_G[5], cpos_G[5], cpos_G[5]};
  vxsf cpos6 = (vxsf) {cpos_G[6], cpos_G[6], cpos_G[6], cpos_G[6]};
  vxsf cpos7 = (vxsf) {cpos_G[7], cpos_G[7], cpos_G[7], cpos_G[7]};
  vxsf cpos8 = (vxsf) {cpos_G[8], cpos_G[8], cpos_G[8], cpos_G[8]};
  vxsf cpos9 = (vxsf) {cpos_G[9], cpos_G[9], cpos_G[9], cpos_G[9]};
  vxsf cpos10= (vxsf) {cpos_G[10],cpos_G[10],cpos_G[10],cpos_G[10]};
  vxsf cpos11= (vxsf) {cpos_G[11],cpos_G[11],cpos_G[11],cpos_G[11]};
  
  #else
  vxsf fx = (vxsf) {fx_s, fx_s, fx_s, fx_s,fx_s, fx_s, fx_s, fx_s};
  vxsf fy = (vxsf) {fy_s, fy_s, fy_s, fy_s,fy_s, fy_s, fy_s, fy_s};
  vxsf cx = (vxsf) {cx_s, cx_s, cx_s, cx_s,cx_s, cx_s, cx_s, cx_s};
  vxsf cy = (vxsf) {cy_s, cy_s, cy_s, cy_s,cy_s, cy_s, cy_s, cy_s};
  
  vxsf cpos0 = (vxsf) {cpos_G[0], cpos_G[0], cpos_G[0], cpos_G[0],cpos_G[0], cpos_G[0], cpos_G[0], cpos_G[0]};
  vxsf cpos1 = (vxsf) {cpos_G[1], cpos_G[1], cpos_G[1], cpos_G[1],cpos_G[1], cpos_G[1], cpos_G[1], cpos_G[1]};
  vxsf cpos2 = (vxsf) {cpos_G[2], cpos_G[2], cpos_G[2], cpos_G[2],cpos_G[2], cpos_G[2], cpos_G[2], cpos_G[2]};
  vxsf cpos3 = (vxsf) {cpos_G[3], cpos_G[3], cpos_G[3], cpos_G[3],cpos_G[3], cpos_G[3], cpos_G[3], cpos_G[3]};
  vxsf cpos4 = (vxsf) {cpos_G[4], cpos_G[4], cpos_G[4], cpos_G[4],cpos_G[4], cpos_G[4], cpos_G[4], cpos_G[4]};
  vxsf cpos5 = (vxsf) {cpos_G[5], cpos_G[5], cpos_G[5], cpos_G[5],cpos_G[5], cpos_G[5], cpos_G[5], cpos_G[5]};
  vxsf cpos6 = (vxsf) {cpos_G[6], cpos_G[6], cpos_G[6], cpos_G[6],cpos_G[6], cpos_G[6], cpos_G[6], cpos_G[6]};
  vxsf cpos7 = (vxsf) {cpos_G[7], cpos_G[7], cpos_G[7], cpos_G[7],cpos_G[7], cpos_G[7], cpos_G[7], cpos_G[7]};
  vxsf cpos8 = (vxsf) {cpos_G[8], cpos_G[8], cpos_G[8], cpos_G[8],cpos_G[8], cpos_G[8], cpos_G[8], cpos_G[8]};
  vxsf cpos9 = (vxsf) {cpos_G[9], cpos_G[9], cpos_G[9], cpos_G[9],cpos_G[9], cpos_G[9], cpos_G[9], cpos_G[9]};
  vxsf cpos10= (vxsf) {cpos_G[10],cpos_G[10],cpos_G[10],cpos_G[10],cpos_G[10],cpos_G[10],cpos_G[10],cpos_G[10]};
  vxsf cpos11= (vxsf) {cpos_G[11],cpos_G[11],cpos_G[11],cpos_G[11],cpos_G[11],cpos_G[11],cpos_G[11],cpos_G[11]};
  #endif
  
  //vxsf *X = (vxsf *)__builtin_assume_aligned ((vxsf*) (pt3d)                    , ALIGNMENTBYTE);
  
  for (int i=nopoints/SSEMULTIPL; i--  ; ++X, ++Y, ++Z, ++x, ++y)
  {
    // Apply camera position, move to plane at Z=1
    tmp3d_X =  cpos0 * (*X) + cpos1 * (*Y) + cpos2  * (*Z) + cpos3;
    tmp3d_Y =  cpos4 * (*X) + cpos5 * (*Y) + cpos6  * (*Z) + cpos7;
    tmp3d_Z =  cpos8 * (*X) + cpos9 * (*Y) + cpos10 * (*Z) + cpos11;
    
    // Project into 2D
    (*x) = (tmp3d_X/tmp3d_Z) * fx + cx;
    (*y) = (tmp3d_Y/tmp3d_Z) * fy + cy;
  }
  
#else   // NO SSE INSTRUCTIONS
  for (int i=nopoints; i--  ; ++X, ++Y, ++Z, ++x, ++y)
  {
    // Apply camera position, move to plane at Z=1
    tmp3d_X =  cpos_G[0] * (*X) + cpos_G[1] * (*Y) + cpos_G[2]  * (*Z) + cpos_G[3];
    tmp3d_Y =  cpos_G[4] * (*X) + cpos_G[5] * (*Y) + cpos_G[6]  * (*Z) + cpos_G[7];
    tmp3d_Z =  cpos_G[8] * (*X) + cpos_G[9] * (*Y) + cpos_G[10] * (*Z) + cpos_G[11];
    
    // Project into 2D
    (*x) = (tmp3d_X/tmp3d_Z) * fx_s + cx_s;
    (*y) = (tmp3d_Y/tmp3d_Z) * fy_s + cy_s;
  }
#endif
}


void PoseClass::project_pt_save_rotated(const float *pt3d, float *pt3d_rot, float *pt2d, int nopoints, int sc) const
{
  float fx_s = camobj->getfx(sc);
  float fy_s = camobj->getfy(sc);
  float cx_s = camobj->getcx(sc);
  float cy_s = camobj->getcy(sc);
  
  vxsf * X = (vxsf*) (pt3d);
  vxsf * Y = (vxsf*) (pt3d + op->maxpttrack  );
  vxsf * Z = (vxsf*) (pt3d + op->maxpttrack*2);
  
  vxsf * X_rot = (vxsf*) (pt3d_rot);
  vxsf * Y_rot = (vxsf*) (pt3d_rot + op->maxpttrack  );
  vxsf * Z_rot = (vxsf*) (pt3d_rot + op->maxpttrack*2);
  
  vxsf * x = (vxsf*) (pt2d);
  vxsf * y = (vxsf*) (pt2d + op->maxpttrack  );  
  

#if (SSEMULTIPL>1)
  int div = nopoints % SSEMULTIPL;
  if (div>0) 
    nopoints = nopoints + (SSEMULTIPL - div);
  
  #if (SSEMULTIPL == 4)
  vxsf fx = (vxsf) {fx_s, fx_s, fx_s, fx_s};
  vxsf fy = (vxsf) {fy_s, fy_s, fy_s, fy_s};
  vxsf cx = (vxsf) {cx_s, cx_s, cx_s, cx_s};
  vxsf cy = (vxsf) {cy_s, cy_s, cy_s, cy_s};
  
  vxsf cpos0 = (vxsf) {cpos_G[0], cpos_G[0], cpos_G[0], cpos_G[0]};
  vxsf cpos1 = (vxsf) {cpos_G[1], cpos_G[1], cpos_G[1], cpos_G[1]};
  vxsf cpos2 = (vxsf) {cpos_G[2], cpos_G[2], cpos_G[2], cpos_G[2]};
  vxsf cpos3 = (vxsf) {cpos_G[3], cpos_G[3], cpos_G[3], cpos_G[3]};
  vxsf cpos4 = (vxsf) {cpos_G[4], cpos_G[4], cpos_G[4], cpos_G[4]};
  vxsf cpos5 = (vxsf) {cpos_G[5], cpos_G[5], cpos_G[5], cpos_G[5]};
  vxsf cpos6 = (vxsf) {cpos_G[6], cpos_G[6], cpos_G[6], cpos_G[6]};
  vxsf cpos7 = (vxsf) {cpos_G[7], cpos_G[7], cpos_G[7], cpos_G[7]};
  vxsf cpos8 = (vxsf) {cpos_G[8], cpos_G[8], cpos_G[8], cpos_G[8]};
  vxsf cpos9 = (vxsf) {cpos_G[9], cpos_G[9], cpos_G[9], cpos_G[9]};
  vxsf cpos10= (vxsf) {cpos_G[10],cpos_G[10],cpos_G[10],cpos_G[10]};
  vxsf cpos11= (vxsf) {cpos_G[11],cpos_G[11],cpos_G[11],cpos_G[11]};
  
  #else
  vxsf fx = (vxsf) {fx_s, fx_s, fx_s, fx_s,fx_s, fx_s, fx_s, fx_s};
  vxsf fy = (vxsf) {fy_s, fy_s, fy_s, fy_s,fy_s, fy_s, fy_s, fy_s};
  vxsf cx = (vxsf) {cx_s, cx_s, cx_s, cx_s,cx_s, cx_s, cx_s, cx_s};
  vxsf cy = (vxsf) {cy_s, cy_s, cy_s, cy_s,cy_s, cy_s, cy_s, cy_s};
  
  vxsf cpos0 = (vxsf) {cpos_G[0], cpos_G[0], cpos_G[0], cpos_G[0],cpos_G[0], cpos_G[0], cpos_G[0], cpos_G[0]};
  vxsf cpos1 = (vxsf) {cpos_G[1], cpos_G[1], cpos_G[1], cpos_G[1],cpos_G[1], cpos_G[1], cpos_G[1], cpos_G[1]};
  vxsf cpos2 = (vxsf) {cpos_G[2], cpos_G[2], cpos_G[2], cpos_G[2],cpos_G[2], cpos_G[2], cpos_G[2], cpos_G[2]};
  vxsf cpos3 = (vxsf) {cpos_G[3], cpos_G[3], cpos_G[3], cpos_G[3],cpos_G[3], cpos_G[3], cpos_G[3], cpos_G[3]};
  vxsf cpos4 = (vxsf) {cpos_G[4], cpos_G[4], cpos_G[4], cpos_G[4],cpos_G[4], cpos_G[4], cpos_G[4], cpos_G[4]};
  vxsf cpos5 = (vxsf) {cpos_G[5], cpos_G[5], cpos_G[5], cpos_G[5],cpos_G[5], cpos_G[5], cpos_G[5], cpos_G[5]};
  vxsf cpos6 = (vxsf) {cpos_G[6], cpos_G[6], cpos_G[6], cpos_G[6],cpos_G[6], cpos_G[6], cpos_G[6], cpos_G[6]};
  vxsf cpos7 = (vxsf) {cpos_G[7], cpos_G[7], cpos_G[7], cpos_G[7],cpos_G[7], cpos_G[7], cpos_G[7], cpos_G[7]};
  vxsf cpos8 = (vxsf) {cpos_G[8], cpos_G[8], cpos_G[8], cpos_G[8],cpos_G[8], cpos_G[8], cpos_G[8], cpos_G[8]};
  vxsf cpos9 = (vxsf) {cpos_G[9], cpos_G[9], cpos_G[9], cpos_G[9],cpos_G[9], cpos_G[9], cpos_G[9], cpos_G[9]};
  vxsf cpos10= (vxsf) {cpos_G[10],cpos_G[10],cpos_G[10],cpos_G[10],cpos_G[10],cpos_G[10],cpos_G[10],cpos_G[10]};
  vxsf cpos11= (vxsf) {cpos_G[11],cpos_G[11],cpos_G[11],cpos_G[11],cpos_G[11],cpos_G[11],cpos_G[11],cpos_G[11]};
  #endif
  
  for (int i=nopoints/SSEMULTIPL; i--  ; ++X, ++Y, ++Z, ++X_rot, ++Y_rot, ++Z_rot, ++x, ++y)
  {
    // Apply camera position, move to plane at Z=1
    (*X_rot) =  cpos0 * (*X) + cpos1 * (*Y) + cpos2  * (*Z) + cpos3;
    (*Y_rot) =  cpos4 * (*X) + cpos5 * (*Y) + cpos6  * (*Z) + cpos7;
    (*Z_rot) =  cpos8 * (*X) + cpos9 * (*Y) + cpos10 * (*Z) + cpos11;
    
    // Project into 2D
    (*x) = ((*X_rot)/(*Z_rot)) * fx + cx;
    (*y) = ((*Y_rot)/(*Z_rot)) * fy + cy;    
  }
  
#else   // NO SSE INSTRUCTIONS
  for (int i=nopoints; i--  ; ++X, ++Y, ++Z, ++X_rot, ++Y_rot, ++Z_rot, ++x, ++y)
  {
    // Apply camera position, move to plane at Z=1
    (*X_rot) =  cpos_G[0] * (*X) + cpos_G[1] * (*Y) + cpos_G[2]  * (*Z) + cpos_G[3];
    (*Y_rot) =  cpos_G[4] * (*X) + cpos_G[5] * (*Y) + cpos_G[6]  * (*Z) + cpos_G[7];
    (*Z_rot) =  cpos_G[8] * (*X) + cpos_G[9] * (*Y) + cpos_G[10] * (*Z) + cpos_G[11];
    
    // Project into 2D
    (*x) = ((*X_rot)/(*Z_rot)) * fx_s + cx_s;
    (*y) = ((*Y_rot)/(*Z_rot)) * fy_s + cy_s;    
  }
#endif
}






}















/*



{
  float fx_s = camobj->getfx(sc);
  float fy_s = camobj->getfy(sc);
  float cx_s = camobj->getcx(sc);
  float cy_s = camobj->getcy(sc);
  
#if (SSEMULTIPL>1)
  int div = nopoints % SSEMULTIPL;
  if (div>0) 
    nopoints = nopoints + (SSEMULTIPL - div);
  
  #if (SSEMULTIPL == 4)
  vxsf fx = (vxsf) {fx_s, fx_s, fx_s, fx_s};
  vxsf fy = (vxsf) {fy_s, fy_s, fy_s, fy_s};
  vxsf cx = (vxsf) {cx_s, cx_s, cx_s, cx_s};
  vxsf cy = (vxsf) {cy_s, cy_s, cy_s, cy_s};
  
  vxsf cpos0 = (vxsf) {cpos_G[0], cpos_G[0], cpos_G[0], cpos_G[0]};
  vxsf cpos1 = (vxsf) {cpos_G[1], cpos_G[1], cpos_G[1], cpos_G[1]};
  vxsf cpos2 = (vxsf) {cpos_G[2], cpos_G[2], cpos_G[2], cpos_G[2]};
  vxsf cpos3 = (vxsf) {cpos_G[3], cpos_G[3], cpos_G[3], cpos_G[3]};
  vxsf cpos4 = (vxsf) {cpos_G[4], cpos_G[4], cpos_G[4], cpos_G[4]};
  vxsf cpos5 = (vxsf) {cpos_G[5], cpos_G[5], cpos_G[5], cpos_G[5]};
  vxsf cpos6 = (vxsf) {cpos_G[6], cpos_G[6], cpos_G[6], cpos_G[6]};
  vxsf cpos7 = (vxsf) {cpos_G[7], cpos_G[7], cpos_G[7], cpos_G[7]};
  vxsf cpos8 = (vxsf) {cpos_G[8], cpos_G[8], cpos_G[8], cpos_G[8]};
  vxsf cpos9 = (vxsf) {cpos_G[9], cpos_G[9], cpos_G[9], cpos_G[9]};
  vxsf cpos10= (vxsf) {cpos_G[10],cpos_G[10],cpos_G[10],cpos_G[10]};
  vxsf cpos11= (vxsf) {cpos_G[11],cpos_G[11],cpos_G[11],cpos_G[11]};
  
  #else
  vxsf fx = (vxsf) {fx_s, fx_s, fx_s, fx_s,fx_s, fx_s, fx_s, fx_s};
  vxsf fy = (vxsf) {fy_s, fy_s, fy_s, fy_s,fy_s, fy_s, fy_s, fy_s};
  vxsf cx = (vxsf) {cx_s, cx_s, cx_s, cx_s,cx_s, cx_s, cx_s, cx_s};
  vxsf cy = (vxsf) {cy_s, cy_s, cy_s, cy_s,cy_s, cy_s, cy_s, cy_s};
  
  vxsf cpos0 = (vxsf) {cpos_G[0], cpos_G[0], cpos_G[0], cpos_G[0],cpos_G[0], cpos_G[0], cpos_G[0], cpos_G[0]};
  vxsf cpos1 = (vxsf) {cpos_G[1], cpos_G[1], cpos_G[1], cpos_G[1],cpos_G[1], cpos_G[1], cpos_G[1], cpos_G[1]};
  vxsf cpos2 = (vxsf) {cpos_G[2], cpos_G[2], cpos_G[2], cpos_G[2],cpos_G[2], cpos_G[2], cpos_G[2], cpos_G[2]};
  vxsf cpos3 = (vxsf) {cpos_G[3], cpos_G[3], cpos_G[3], cpos_G[3],cpos_G[3], cpos_G[3], cpos_G[3], cpos_G[3]};
  vxsf cpos4 = (vxsf) {cpos_G[4], cpos_G[4], cpos_G[4], cpos_G[4],cpos_G[4], cpos_G[4], cpos_G[4], cpos_G[4]};
  vxsf cpos5 = (vxsf) {cpos_G[5], cpos_G[5], cpos_G[5], cpos_G[5],cpos_G[5], cpos_G[5], cpos_G[5], cpos_G[5]};
  vxsf cpos6 = (vxsf) {cpos_G[6], cpos_G[6], cpos_G[6], cpos_G[6],cpos_G[6], cpos_G[6], cpos_G[6], cpos_G[6]};
  vxsf cpos7 = (vxsf) {cpos_G[7], cpos_G[7], cpos_G[7], cpos_G[7],cpos_G[7], cpos_G[7], cpos_G[7], cpos_G[7]};
  vxsf cpos8 = (vxsf) {cpos_G[8], cpos_G[8], cpos_G[8], cpos_G[8],cpos_G[8], cpos_G[8], cpos_G[8], cpos_G[8]};
  vxsf cpos9 = (vxsf) {cpos_G[9], cpos_G[9], cpos_G[9], cpos_G[9],cpos_G[9], cpos_G[9], cpos_G[9], cpos_G[9]};
  vxsf cpos10= (vxsf) {cpos_G[10],cpos_G[10],cpos_G[10],cpos_G[10],cpos_G[10],cpos_G[10],cpos_G[10],cpos_G[10]};
  vxsf cpos11= (vxsf) {cpos_G[11],cpos_G[11],cpos_G[11],cpos_G[11],cpos_G[11],cpos_G[11],cpos_G[11],cpos_G[11]};
  #endif
  
  vxsf * X = (vxsf*) (pt3d);
  vxsf * Y = (vxsf*) (pt3d + op->maxpttrack  );
  vxsf * Z = (vxsf*) (pt3d + op->maxpttrack*2);

  vxsf * x = (vxsf*) (pt2d);
  vxsf * y = (vxsf*) (pt2d + op->maxpttrack  );

  //vxsf *X = (vxsf *)__builtin_assume_aligned ((vxsf*) (pt3d)                    , ALIGNMENTBYTE);
  
//   vxsf tmp3d_X;
//   vxsf tmp3d_Y;
//   vxsf tmp3d_Z;
  vxsf *tmp3d_X = (vxsf*) (pt3d);
  vxsf *tmp3d_Y = (vxsf*) (pt3d + op->maxpttrack  );
  vxsf *tmp3d_Z = (vxsf*) (pt3d + op->maxpttrack*2);
  
  for (int i=nopoints/SSEMULTIPL; i--  ; ++X, ++Y, ++Z, ++x, ++y)
  {
    // Apply camera position, move to plane at Z=1
    (*tmp3d_Z) =  cpos8 * (*X) + cpos9 * (*Y) + cpos10 * (*Z) + cpos11;
    (*tmp3d_X) =  cpos0 * (*X) + cpos1 * (*Y) + cpos2 * (*Z)  + cpos3;
    (*tmp3d_Y) =  cpos4 * (*X) + cpos5 * (*Y) + cpos6 * (*Z)  + cpos7;

    // Project into 2D
    (*x) = ((*tmp3d_X)/(*tmp3d_Z)) * fx + cx;
    (*y) = ((*tmp3d_Y)/(*tmp3d_Z)) * fy + cy;
  }
  
#else   // NO SSE INSTRUCTIONS
  float * X = (float*) (pt3d);
  float * Y = (float*) (pt3d + op->maxpttrack  );
  float * Z = (float*) (pt3d + op->maxpttrack*2);
  float * x = (float*) (pt2d);
  float * y = (float*) (pt2d + op->maxpttrack  );

  float tmp3d_X;
  float tmp3d_Y;
  float tmp3d_Z;
  
  for (int i=nopoints; i--  ; ++X, ++Y, ++Z, ++x, ++y)
  {
    // Apply camera position, move to plane at Z=1
    tmp3d_Z =  cpos_G[8] * (*X) + cpos_G[9] * (*Y) + cpos_G[10] * (*Z) + cpos_G[11];
    tmp3d_X =  cpos_G[0] * (*X) + cpos_G[1] * (*Y) + cpos_G[2] * (*Z)  + cpos_G[3];
    tmp3d_Y =  cpos_G[4] * (*X) + cpos_G[5] * (*Y) + cpos_G[6] * (*Z)  + cpos_G[7];

    // Project into 2D
    (*x) = (tmp3d_X/tmp3d_Z) * fx_s + cx_s;
    (*y) = (tmp3d_Y/tmp3d_Z) * fy_s + cy_s;
  }
#endif
}





{ 
  
  vxsf* fx = (vxsf*) camobj->getfx_s(sc);
  vxsf* fy = (vxsf*) camobj->getfy_s(sc);
  vxsf* cx = (vxsf*) camobj->getcx_s(sc);
  vxsf* cy = (vxsf*) camobj->getcy_s(sc);
  
// #if (SSEMULTIPL==4 || SSEMULTIPL==8)
  int div = nopoints % SSEMULTIPL;
  if (div>0) 
    nopoints = nopoints + (SSEMULTIPL - div);
  
//   #if (SSEMULTIPL==4)  
//     vxsf cpos0 = (vxsf) {cpos_G[0], cpos_G[0], cpos_G[0], cpos_G[0]};
//     vxsf cpos1 = (vxsf) {cpos_G[1], cpos_G[1], cpos_G[1], cpos_G[1]};
//     vxsf cpos2 = (vxsf) {cpos_G[2], cpos_G[2], cpos_G[2], cpos_G[2]};
//     vxsf cpos3 = (vxsf) {cpos_G[3], cpos_G[3], cpos_G[3], cpos_G[3]};
//     vxsf cpos4 = (vxsf) {cpos_G[4], cpos_G[4], cpos_G[4], cpos_G[4]};
//     vxsf cpos5 = (vxsf) {cpos_G[5], cpos_G[5], cpos_G[5], cpos_G[5]};
//     vxsf cpos6 = (vxsf) {cpos_G[6], cpos_G[6], cpos_G[6], cpos_G[6]};
//     vxsf cpos7 = (vxsf) {cpos_G[7], cpos_G[7], cpos_G[7], cpos_G[7]};
//     vxsf cpos8 = (vxsf) {cpos_G[8], cpos_G[8], cpos_G[8], cpos_G[8]};
//     vxsf cpos9 = (vxsf) {cpos_G[9], cpos_G[9], cpos_G[9], cpos_G[9]};
//     vxsf cpos10= (vxsf) {cpos_G[10],cpos_G[10],cpos_G[10],cpos_G[10]};
//     vxsf cpos11= (vxsf) {cpos_G[11],cpos_G[11],cpos_G[11],cpos_G[11]};
//   #elif (SSEMULTIPL==8)
//     vxsf cpos0 = (vxsf) {cpos_G[0], cpos_G[0], cpos_G[0], cpos_G[0],cpos_G[0], cpos_G[0], cpos_G[0], cpos_G[0]};
//     vxsf cpos1 = (vxsf) {cpos_G[1], cpos_G[1], cpos_G[1], cpos_G[1],cpos_G[1], cpos_G[1], cpos_G[1], cpos_G[1]};
//     vxsf cpos2 = (vxsf) {cpos_G[2], cpos_G[2], cpos_G[2], cpos_G[2],cpos_G[2], cpos_G[2], cpos_G[2], cpos_G[2]};
//     vxsf cpos3 = (vxsf) {cpos_G[3], cpos_G[3], cpos_G[3], cpos_G[3],cpos_G[3], cpos_G[3], cpos_G[3], cpos_G[3]};
//     vxsf cpos4 = (vxsf) {cpos_G[4], cpos_G[4], cpos_G[4], cpos_G[4],cpos_G[4], cpos_G[4], cpos_G[4], cpos_G[4]};
//     vxsf cpos5 = (vxsf) {cpos_G[5], cpos_G[5], cpos_G[5], cpos_G[5],cpos_G[5], cpos_G[5], cpos_G[5], cpos_G[5]};
//     vxsf cpos6 = (vxsf) {cpos_G[6], cpos_G[6], cpos_G[6], cpos_G[6],cpos_G[6], cpos_G[6], cpos_G[6], cpos_G[6]};
//     vxsf cpos7 = (vxsf) {cpos_G[7], cpos_G[7], cpos_G[7], cpos_G[7],cpos_G[7], cpos_G[7], cpos_G[7], cpos_G[7]};
//     vxsf cpos8 = (vxsf) {cpos_G[8], cpos_G[8], cpos_G[8], cpos_G[8],cpos_G[8], cpos_G[8], cpos_G[8], cpos_G[8]};
//     vxsf cpos9 = (vxsf) {cpos_G[9], cpos_G[9], cpos_G[9], cpos_G[9],cpos_G[9], cpos_G[9], cpos_G[9], cpos_G[9]};
//     vxsf cpos10= (vxsf) {cpos_G[10],cpos_G[10],cpos_G[10],cpos_G[10],cpos_G[10],cpos_G[10],cpos_G[10],cpos_G[10]};
//     vxsf cpos11= (vxsf) {cpos_G[11],cpos_G[11],cpos_G[11],cpos_G[11],cpos_G[11],cpos_G[11],cpos_G[11],cpos_G[11]};
//   #endif
    
    //void *x = __builtin_assume_aligned (arg, 16);
    
    vxsf * X = (vxsf*) (pt3d);
    vxsf * Y = (vxsf*) (pt3d + op->maxpttrack  );
    vxsf * Z = (vxsf*) (pt3d + op->maxpttrack*2);

    vxsf * x = (vxsf*) (pt2d);
    vxsf * y = (vxsf*) (pt2d + op->maxpttrack  );

    //vxsf *X = (vxsf *)__builtin_assume_aligned ((vxsf*) (pt3d)                    , ALIGNMENTBYTE);
    
    vxsf tmp3d_X;
    vxsf tmp3d_Y;
    vxsf tmp3d_Z;
    
    for (int i=nopoints/SSEMULTIPL; i--  ; ++X, ++Y, ++Z, ++x, ++y)
    {
      // Apply camera position, move to plane at Z=1
      tmp3d_Z =  (*cpos8) * (*X) + (*cpos9) * (*Y) + (*cpos10) * (*Z) + (*cpos11);
      tmp3d_X = ((*cpos0) * (*X) + (*cpos1) * (*Y) + (*cpos2) * (*Z)  + (*cpos3)) / tmp3d_Z;
      tmp3d_Y = ((*cpos4) * (*X) + (*cpos5) * (*Y) + (*cpos6) * (*Z)  + (*cpos7)) / tmp3d_Z;

      // Project into 2D
      (*x) = tmp3d_X * (*fx) + (*cx);
      (*y) = tmp3d_Y * (*fy) + (*cy);
    }
// #else   // NO SSE INSTRUCTIONS
//   float * X = (float*) (pt3d);
//   float * Y = (float*) (pt3d + op->maxpttrack  );
//   float * Z = (float*) (pt3d + op->maxpttrack*2);
//   float * x = (float*) (pt2d);
//   float * y = (float*) (pt2d + op->maxpttrack  );
// 
//   float tmp3d[3];
//   
//   for (int i=nopoints; i--  ; ++X, ++Y, ++Z, ++x, ++y)
//   {
//     // Apply camera position, move to plane at Z=1
//     tmp3d[2] =  cpos_G[8] * (*X) + cpos_G[9] * (*Y) + cpos_G[10] * (*Z) + cpos_G[11];
//     tmp3d[0] = (cpos_G[0] * (*X) + cpos_G[1] * (*Y) + cpos_G[2] * (*Z)  + cpos_G[3]) / tmp3d[2];
//     tmp3d[1] = (cpos_G[4] * (*X) + cpos_G[5] * (*Y) + cpos_G[6] * (*Z)  + cpos_G[7]) / tmp3d[2];
// 
//     // Project into 2D
//     (*x) = tmp3d[0] * (*fx) + (*cx);
//     (*y) = tmp3d[1] * (*fy) + (*cy);
//   }
// #endif
}
  
  
  */