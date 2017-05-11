
#include "utilities.h"

#include <iostream>
#include <cstring>
#include <math.h>

using std::cout;
using std::endl;

namespace CTR
{
  
void util_constructpyramide(const cv::Mat & img_ao_fmat, cv::Mat * img_ao_fmat_pyr, cv::Mat * img_ao_dx_fmat_pyr, cv::Mat * img_ao_dy_fmat_pyr, 
                            const float ** img_ao_pyr, const float ** img_ao_dx_pyr, const float ** img_ao_dy_pyr, const int lv_f, const bool getgrad, const int imgpadding)
{
    for (int i=0; i<=lv_f; ++i)  // Construct image and gradient pyramides
    {
      if (i==0) // at finest scale: copy directly, for all other: downscale previous scale by .5
      {
        img_ao_fmat_pyr[i] = img_ao_fmat.clone();
      }
      else
        cv::resize(img_ao_fmat_pyr[i-1], img_ao_fmat_pyr[i], cv::Size(), .5, .5, cv::INTER_LINEAR);

      img_ao_fmat_pyr[i].convertTo(img_ao_fmat_pyr[i], CV_32FC1);
        
      if ( getgrad ) 
      {
        cv::Sobel( img_ao_fmat_pyr[i], img_ao_dx_fmat_pyr[i], CV_32F, 1, 0, 1, 1, 0, cv::BORDER_DEFAULT );
        cv::Sobel( img_ao_fmat_pyr[i], img_ao_dy_fmat_pyr[i], CV_32F, 0, 1, 1, 1, 0, cv::BORDER_DEFAULT );
        img_ao_dx_fmat_pyr[i].convertTo(img_ao_dx_fmat_pyr[i], CV_32F);
        img_ao_dy_fmat_pyr[i].convertTo(img_ao_dy_fmat_pyr[i], CV_32F);
      }
    }
    
    // pad images
    for (int i=0; i<=lv_f; ++i)  // Construct image and gradient pyramides
    {
      copyMakeBorder(img_ao_fmat_pyr[i],img_ao_fmat_pyr[i],imgpadding,imgpadding,imgpadding,imgpadding,cv::BORDER_REPLICATE);
      img_ao_pyr[i] = (float*)img_ao_fmat_pyr[i].data;

      if ( getgrad ) 
      {
        copyMakeBorder(img_ao_dx_fmat_pyr[i],img_ao_dx_fmat_pyr[i],imgpadding,imgpadding,imgpadding,imgpadding,cv::BORDER_CONSTANT , 0); // BORDER_CONSTANT 
        copyMakeBorder(img_ao_dy_fmat_pyr[i],img_ao_dy_fmat_pyr[i],imgpadding,imgpadding,imgpadding,imgpadding,cv::BORDER_CONSTANT , 0);

        img_ao_dx_pyr[i] = (float*)img_ao_dx_fmat_pyr[i].data;
        img_ao_dy_pyr[i] = (float*)img_ao_dy_fmat_pyr[i].data;      
      }
    }
}


void util_getPatch(const float* img, const float *mid_in,  Eigen::Map<MatrixXfTr, Eigen::Aligned> * tmp_in_e, const optparam* op, const int width) // pass argument width with padding !
{
  float *tmp_in    = tmp_in_e->data();

  
  float resid[2];
  float we[4]; // bilinear weight vector
  int pos[4];
  int pos_it[2];
  
  // Compute the bilinear weight vector, for patch without orientation/scale change -> weight vector is constant
  pos[0] = ceil(mid_in[0]+.00001f); // make sure they are rounded up for full natural numbers
  pos[1] = ceil(mid_in[1]+.00001f);
  pos[2] = floor(mid_in[0]);
  pos[3] = floor(mid_in[1]);  
  
  resid[0] = mid_in[0] - (float)pos[2];
  resid[1] = mid_in[1] - (float)pos[3];
  we[0] = resid[0]*resid[1];
  we[1] = (1-resid[0])*resid[1];
  we[2] = resid[0]*(1-resid[1]);
  we[3] = (1-resid[0])*(1-resid[1]);

  //pos[0] += op->psz; // add padding
  //pos[1] += op->psz;
  
  float * tmp_it = tmp_in;
  const float * img_a, * img_b, * img_c, * img_d, *img_e; 
   
  img_e = img    + pos[0]+op->pszd2;
 
//   int lb = -op->pszd2;
//   int ub = op->pszd2-1;     

//   int paddtt = op->psz+op->pszd2-1;
  int lbo = pos[1]+op->pszd2;
  int ubo = pos[1]+op->pszd2m3;
  
  int lbi = pos[0]+op->pszd2;
  int ubi = pos[0]+op->pszd2m3;

  
  for (pos_it[1]= lbo; pos_it[1] <= ubo; ++pos_it[1])    
  {
      img_a = img_e +  pos_it[1]    * width;
      img_c = img_e + (pos_it[1]-1) * width;
      img_b = img_a-1;
      img_d = img_c-1;
  
    for (pos_it[0]= lbi ; pos_it[0] <= ubi; ++pos_it[0], 
            ++tmp_it,++img_a,++img_b,++img_c,++img_d)    
    {
      (*tmp_it)     = we[0] * (*img_a) + we[1] * (*img_b) + we[2] * (*img_c) + we[3] * (*img_d); 
    }
  }
  // PATCH NORMALIZATION
  if (op->dopatchnorm) // Subtract Mean
    tmp_in_e->array() -= (tmp_in_e->sum() / op->novals );
}  

void util_getPatch_grad(const float* img, const float* img_dx, const float* img_dy, const float *mid_in,  Eigen::Map<MatrixXfTr, Eigen::Aligned>* tmp_in_e, Eigen::Map<MatrixXfTr, Eigen::Aligned> *  tmp_dx_in_e, Eigen::Map<MatrixXfTr, Eigen::Aligned> * tmp_dy_in_e, const optparam* op, const int width) // pass width with padding !
{
  float *tmp_it    = tmp_in_e->data();
  float *tmp_it_dx = tmp_dx_in_e->data();
  float *tmp_it_dy = tmp_dy_in_e->data();

  
  float resid[2];
  float we[4]; // bilinear weight vector
  int pos[4];
  int pos_it[2];

  // Compute the bilinear weight vector, for patch without orientation/scale change -> weight vector is constant
  pos[0] = ceil(mid_in[0]+.00001f); // make sure they are rounded up for full natural numbers
  pos[1] = ceil(mid_in[1]+.00001f);
  pos[2] = floor(mid_in[0]);
  pos[3] = floor(mid_in[1]);  
  
  resid[0] = mid_in[0] - (float)pos[2];
  resid[1] = mid_in[1] - (float)pos[3];
  we[0] = resid[0]*resid[1];
  we[1] = (1-resid[0])*resid[1];
  we[2] = resid[0]*(1-resid[1]);
  we[3] = (1-resid[0])*(1-resid[1]);
  
  const float * img_a, * img_b, * img_c, * img_d, *img_e;
  const float * img_a_dx, * img_b_dx, * img_c_dx, * img_d_dx, *img_e_dx; 
  const float * img_a_dy, * img_b_dy, * img_c_dy, * img_d_dy, *img_e_dy; 
   
  int pos0tt = pos[0]+op->pszd2;
  img_e    = img    + pos0tt;
  img_e_dx = img_dx + pos0tt;
  img_e_dy = img_dy + pos0tt;
 
//   int lb = -op->pszd2;
//   int ub = op->pszd2-1;     

//   int paddtt = op->psz+op->pszd2-1;
  int lbo = pos[1]+op->pszd2;
  int ubo = pos[1]+op->pszd2m3;
  
  int lbi = pos0tt;
  int ubi = pos[0]+op->pszd2m3;

  
  for (pos_it[1]= lbo; pos_it[1] <= ubo; ++pos_it[1])    
  {
      int posit1tt1 = pos_it[1]    * width;
      int posit1tt2 = (pos_it[1]-1) * width;
      img_a    = img_e    + posit1tt1;
      img_c    = img_e    + posit1tt2;
      img_b    = img_a    -1;
      img_d    = img_c    -1;
      img_a_dx = img_e_dx + posit1tt1;
      img_c_dx = img_e_dx + posit1tt2;
      img_b_dx = img_a_dx -1;
      img_d_dx = img_c_dx -1;
      img_a_dy = img_e_dy + posit1tt1;
      img_c_dy = img_e_dy + posit1tt2;
      img_b_dy = img_a_dy -1;
      img_d_dy = img_c_dy -1;  
      for (pos_it[0]= lbi ; pos_it[0] <= ubi; ++pos_it[0], 
              ++tmp_it   ,++img_a   ,++img_b   ,++img_c   ,++img_d   ,
              ++tmp_it_dx,++img_a_dx,++img_b_dx,++img_c_dx,++img_d_dx,
              ++tmp_it_dy,++img_a_dy,++img_b_dy,++img_c_dy,++img_d_dy)    
      {
        (*tmp_it)     = we[0] * (*img_a   ) + we[1] * (*img_b   ) + we[2] * (*img_c   ) + we[3] * (*img_d   );
        (*tmp_it_dx)  = we[0] * (*img_a_dx) + we[1] * (*img_b_dx) + we[2] * (*img_c_dx) + we[3] * (*img_d_dx); 
        (*tmp_it_dy)  = we[0] * (*img_a_dy) + we[1] * (*img_b_dy) + we[2] * (*img_c_dy) + we[3] * (*img_d_dy); 
      }
  }
  // PATCH NORMALIZATION
  if (op->dopatchnorm) // Subtract Mean
    tmp_in_e->array() -= (tmp_in_e->sum() / op->novals );
}  
 
 
/*void util_getPatch_grad_only_I(const float* img, const float *mid_in,  Eigen::Map<MatrixXfTr, Eigen::Aligned>* tmp_in_e, const optparam* op, const int width) // pass width with padding !
{
  float *tmp_it    = tmp_in_e->data();
  
  float resid[2];
  float we[4]; // bilinear weight vector
  int pos[4];
  int pos_it[2];

  // Compute the bilinear weight vector, for patch without orientation/scale change -> weight vector is constant
  pos[0] = ceil(mid_in[0]+.00001f); // make sure they are rounded up for full natural numbers
  pos[1] = ceil(mid_in[1]+.00001f);
  pos[2] = floor(mid_in[0]);
  pos[3] = floor(mid_in[1]);  
  
  resid[0] = mid_in[0] - (float)pos[2];
  resid[1] = mid_in[1] - (float)pos[3];
  we[0] = resid[0]*resid[1];
  we[1] = (1-resid[0])*resid[1];
  we[2] = resid[0]*(1-resid[1]);
  we[3] = (1-resid[0])*(1-resid[1]);
  
  const float * img_a, * img_b, * img_c, * img_d, *img_e;
   
  int pos0tt = pos[0]+op->pszd2;
  img_e    = img    + pos0tt;
 
//   int lb = -op->pszd2;
//   int ub = op->pszd2-1;     

//   int paddtt = op->psz+op->pszd2-1;
  int lbo = pos[1]+op->pszd2;
  int ubo = pos[1]+op->pszd2m3;
  
  int lbi = pos0tt;
  int ubi = pos[0]+op->pszd2m3;

  
  for (pos_it[1]= lbo; pos_it[1] <= ubo; ++pos_it[1])    
  {
      int posit1tt1 = pos_it[1]    * width;
      int posit1tt2 = (pos_it[1]-1) * width;
      img_a    = img_e    + posit1tt1;
      img_c    = img_e    + posit1tt2;
      img_b    = img_a    -1;
      img_d    = img_c    -1;
      for (pos_it[0]= lbi ; pos_it[0] <= ubi; ++pos_it[0], 
              ++tmp_it   ,++img_a   ,++img_b   ,++img_c   ,++img_d)    
      {
        (*tmp_it)     = we[0] * (*img_a   ) + we[1] * (*img_b   ) + we[2] * (*img_c   ) + we[3] * (*img_d   );
      }
  }
  // PATCH NORMALIZATION
  if (op->dopatchnorm) // Subtract Mean
    tmp_in_e->array() -= (tmp_in_e->sum() / op->novals );
}  */



// void util_project_pt(const float *pt3d, float *pt2d, const float* cpos, int nopoints, const camparam* cam, const optparam* op)
// {  
// #if (SSEMULTIPL==4 || SSEMULTIPL==8)
//   int div = nopoints % SSEMULTIPL;
//   if (div>0) 
//     nopoints = nopoints + (SSEMULTIPL - div);
//   
//   #if (SSEMULTIPL==4)  
//     vxsf fx = (vxsf) {cam->fc[0], cam->fc[0], cam->fc[0], cam->fc[0]};
//     vxsf fy = (vxsf) {cam->fc[1], cam->fc[1], cam->fc[1], cam->fc[1]};
//     vxsf cx = (vxsf) {cam->cc[0], cam->cc[0], cam->cc[0], cam->cc[0]};
//     vxsf cy = (vxsf) {cam->cc[1], cam->cc[1], cam->cc[1], cam->cc[1]};
//     
//     vxsf cpos0 = (vxsf) {cpos[0], cpos[0], cpos[0], cpos[0]};
//     vxsf cpos1 = (vxsf) {cpos[1], cpos[1], cpos[1], cpos[1]};
//     vxsf cpos2 = (vxsf) {cpos[2], cpos[2], cpos[2], cpos[2]};
//     vxsf cpos3 = (vxsf) {cpos[3], cpos[3], cpos[3], cpos[3]};
//     vxsf cpos4 = (vxsf) {cpos[4], cpos[4], cpos[4], cpos[4]};
//     vxsf cpos5 = (vxsf) {cpos[5], cpos[5], cpos[5], cpos[5]};
//     vxsf cpos6 = (vxsf) {cpos[6], cpos[6], cpos[6], cpos[6]};
//     vxsf cpos7 = (vxsf) {cpos[7], cpos[7], cpos[7], cpos[7]};
//     vxsf cpos8 = (vxsf) {cpos[8], cpos[8], cpos[8], cpos[8]};
//     vxsf cpos9 = (vxsf) {cpos[9], cpos[9], cpos[9], cpos[9]};
//     vxsf cpos10= (vxsf) {cpos[10],cpos[10],cpos[10],cpos[10]};
//     vxsf cpos11= (vxsf) {cpos[11],cpos[11],cpos[11],cpos[11]};
//     
//     //void *fx_t = __builtin_assume_aligned ((float*)  tmp3d_arr              , ALIGNMENTBYTE);
//     //void *fy_t = __builtin_assume_aligned ((float*)  tmp3d_arr+  SSEMULTIPL , ALIGNMENTBYTE);
//     
//   #elif (SSEMULTIPL==8)
//     vxsf fx = (vxsf) {cam->fc[0], cam->fc[0], cam->fc[0], cam->fc[0],cam->fc[0], cam->fc[0], cam->fc[0], cam->fc[0]};
//     vxsf fy = (vxsf) {cam->fc[1], cam->fc[1], cam->fc[1], cam->fc[1],cam->fc[1], cam->fc[1], cam->fc[1], cam->fc[1]};
//     vxsf cx = (vxsf) {cam->cc[0], cam->cc[0], cam->cc[0], cam->cc[0],cam->cc[0], cam->cc[0], cam->cc[0], cam->cc[0]};
//     vxsf cy = (vxsf) {cam->cc[1], cam->cc[1], cam->cc[1], cam->cc[1],cam->cc[1], cam->cc[1], cam->cc[1], cam->cc[1]};
//     
//     vxsf cpos0 = (vxsf) {cpos[0], cpos[0], cpos[0], cpos[0],cpos[0], cpos[0], cpos[0], cpos[0]};
//     vxsf cpos1 = (vxsf) {cpos[1], cpos[1], cpos[1], cpos[1],cpos[1], cpos[1], cpos[1], cpos[1]};
//     vxsf cpos2 = (vxsf) {cpos[2], cpos[2], cpos[2], cpos[2],cpos[2], cpos[2], cpos[2], cpos[2]};
//     vxsf cpos3 = (vxsf) {cpos[3], cpos[3], cpos[3], cpos[3],cpos[3], cpos[3], cpos[3], cpos[3]};
//     vxsf cpos4 = (vxsf) {cpos[4], cpos[4], cpos[4], cpos[4],cpos[4], cpos[4], cpos[4], cpos[4]};
//     vxsf cpos5 = (vxsf) {cpos[5], cpos[5], cpos[5], cpos[5],cpos[5], cpos[5], cpos[5], cpos[5]};
//     vxsf cpos6 = (vxsf) {cpos[6], cpos[6], cpos[6], cpos[6],cpos[6], cpos[6], cpos[6], cpos[6]};
//     vxsf cpos7 = (vxsf) {cpos[7], cpos[7], cpos[7], cpos[7],cpos[7], cpos[7], cpos[7], cpos[7]};
//     vxsf cpos8 = (vxsf) {cpos[8], cpos[8], cpos[8], cpos[8],cpos[8], cpos[8], cpos[8], cpos[8]};
//     vxsf cpos9 = (vxsf) {cpos[9], cpos[9], cpos[9], cpos[9],cpos[9], cpos[9], cpos[9], cpos[9]};
//     vxsf cpos10= (vxsf) {cpos[10],cpos[10],cpos[10],cpos[10],cpos[10],cpos[10],cpos[10],cpos[10]};
//     vxsf cpos11= (vxsf) {cpos[11],cpos[11],cpos[11],cpos[11],cpos[11],cpos[11],cpos[11],cpos[11]};
//   #endif
//     
//     //void *x = __builtin_assume_aligned (arg, 16);
//     
//     vxsf * X = (vxsf*) (pt3d);
//     vxsf * Y = (vxsf*) (pt3d + op->maxpttrack  );
//     vxsf * Z = (vxsf*) (pt3d + op->maxpttrack*2);
// 
//     vxsf * x = (vxsf*) (pt2d);
//     vxsf * y = (vxsf*) (pt2d + op->maxpttrack  );
// 
//     //vxsf *X = (vxsf *)__builtin_assume_aligned ((vxsf*) (pt3d)                    , ALIGNMENTBYTE);
//     
//     vxsf tmp3d_X;
//     vxsf tmp3d_Y;
//     vxsf tmp3d_Z;
//     
//     for (int i=nopoints/SSEMULTIPL; i--  ; ++X, ++Y, ++Z, ++x, ++y)
//     {
//       // Apply camera position, move to plane at Z=1
//       tmp3d_Z =  cpos8 * (*X) + cpos9 * (*Y) + cpos10 * (*Z) + cpos11;
//       tmp3d_X = (cpos0 * (*X) + cpos1 * (*Y) + cpos2 * (*Z)  + cpos3) / tmp3d_Z;
//       tmp3d_Y = (cpos4 * (*X) + cpos5 * (*Y) + cpos6 * (*Z)  + cpos7) / tmp3d_Z;
// 
//       // Project into 2D
//       (*x) = tmp3d_X * fx + cx;
//       (*y) = tmp3d_Y * fy + cy;
//     }
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
//     tmp3d[2] =  cpos[8] * (*X) + cpos[9] * (*Y) + cpos[10] * (*Z) + cpos[11];
//     tmp3d[0] = (cpos[0] * (*X) + cpos[1] * (*Y) + cpos[2] * (*Z)  + cpos[3]) / tmp3d[2];
//     tmp3d[1] = (cpos[4] * (*X) + cpos[5] * (*Y) + cpos[6] * (*Z)  + cpos[7]) / tmp3d[2];
// 
//     // Project into 2D
//     (*x) = tmp3d[0] * cam->fc[0] + cam->cc[0];
//     (*y) = tmp3d[1] * cam->fc[1] + cam->cc[1];
//   }
// #endif
// }
// 
// 
// 
// 
// void util_SE3_coeff_to_Group(float * cpos_G, const float *cpos_p)
// {
//   
//   float sigthresh = 1e-4;
//   std::memset(cpos_G, 12, sizeof(float));
//   //following "Lie Groups for Computer Vision", Ethan Eade
//   
//   // Rotation block
//   float ra1 = cpos_p[3]*cpos_p[3]; // TODO: switch IDs one <-
//   float ra2 = cpos_p[4]*cpos_p[4];
//   float ra3 = cpos_p[5]*cpos_p[5];
//   float sig = sqrt(ra1 + ra2 + ra3);
//   float sa ; // sin_sig_div_sig;
//   float sb; //1mcos_div_sigsq;
//   float sc; //sigm_sin_sig_div_sigsq3;
//   float sigsq2 = (sig*sig);
//   float sigsq3 = (sig*sig*sig);
//   if (sig > sigthresh) 
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
// 
// //   int n = 0 ;
// //   for (int i = 0 ; i< 3; i++)
// //   {
// //     for (int j = 0 ; j< 4; j++, n++)
// //     {
// //       cout << cpos_G[n] << ", ";
// //     }
// //     cout << endl;
// //   }
//   
// }
// 






}























/*

{
  
#if (SSEMULTIPL==4 || SSEMULTIPL==8)
  int div = nopoints % SSEMULTIPL;
  if (div>0) 
    nopoints = nopoints + (SSEMULTIPL - div);
  
  #if (SSEMULTIPL==4)  
    vxsf fx = (vxsf) {cam->fc[0], cam->fc[0], cam->fc[0], cam->fc[0]};
    vxsf fy = (vxsf) {cam->fc[1], cam->fc[1], cam->fc[1], cam->fc[1]};
    vxsf cx = (vxsf) {cam->cc[0], cam->cc[0], cam->cc[0], cam->cc[0]};
    vxsf cy = (vxsf) {cam->cc[1], cam->cc[1], cam->cc[1], cam->cc[1]};
    
    vxsf cpos0 = (vxsf) {cpos[0], cpos[0], cpos[0], cpos[0]};
    vxsf cpos1 = (vxsf) {cpos[1], cpos[1], cpos[1], cpos[1]};
    vxsf cpos2 = (vxsf) {cpos[2], cpos[2], cpos[2], cpos[2]};
    vxsf cpos3 = (vxsf) {cpos[3], cpos[3], cpos[3], cpos[3]};
    vxsf cpos4 = (vxsf) {cpos[4], cpos[4], cpos[4], cpos[4]};
    vxsf cpos5 = (vxsf) {cpos[5], cpos[5], cpos[5], cpos[5]};
    vxsf cpos6 = (vxsf) {cpos[6], cpos[6], cpos[6], cpos[6]};
    vxsf cpos7 = (vxsf) {cpos[7], cpos[7], cpos[7], cpos[7]};
    vxsf cpos8 = (vxsf) {cpos[8], cpos[8], cpos[8], cpos[8]};
    vxsf cpos9 = (vxsf) {cpos[9], cpos[9], cpos[9], cpos[9]};
    vxsf cpos10= (vxsf) {cpos[10],cpos[10],cpos[10],cpos[10]};
    vxsf cpos11= (vxsf) {cpos[11],cpos[11],cpos[11],cpos[11]};
    
    void *cpos0_t = __builtin_assume_aligned ((float*)  cpos_arr              , ALIGNMENTBYTE);
    void *cpos1_t = __builtin_assume_aligned ((float*) (cpos_arr+1*SSEMULTIPL), ALIGNMENTBYTE);
    void *cpos2_t = __builtin_assume_aligned ((float*) (cpos_arr+2*SSEMULTIPL), ALIGNMENTBYTE);
    void *cpos3_t = __builtin_assume_aligned ((float*) (cpos_arr+3*SSEMULTIPL), ALIGNMENTBYTE);
    void *cpos4_t = __builtin_assume_aligned ((float*) (cpos_arr+4*SSEMULTIPL), ALIGNMENTBYTE);
    void *cpos5_t = __builtin_assume_aligned ((float*) (cpos_arr+5*SSEMULTIPL), ALIGNMENTBYTE);
    void *cpos6_t = __builtin_assume_aligned ((float*) (cpos_arr+6*SSEMULTIPL), ALIGNMENTBYTE);
    void *cpos7_t = __builtin_assume_aligned ((float*) (cpos_arr+7*SSEMULTIPL), ALIGNMENTBYTE);
    void *cpos8_t = __builtin_assume_aligned ((float*) (cpos_arr+8*SSEMULTIPL), ALIGNMENTBYTE);
    void *cpos9_t = __builtin_assume_aligned ((float*) (cpos_arr+9*SSEMULTIPL), ALIGNMENTBYTE);
    void *cpos10_t = __builtin_assume_aligned((float*) (cpos_arr+10*SSEMULTIPL),ALIGNMENTBYTE);
    void *cpos11_t = __builtin_assume_aligned((float*) (cpos_arr+11*SSEMULTIPL),ALIGNMENTBYTE);
    void *fx_t    = __builtin_assume_aligned ((float*) (cpos_arr+12*SSEMULTIPL), ALIGNMENTBYTE);
    void *fy_t    = __builtin_assume_aligned ((float*) (cpos_arr+13*SSEMULTIPL), ALIGNMENTBYTE);
    void *cx_t    = __builtin_assume_aligned ((float*) (cpos_arr+14*SSEMULTIPL), ALIGNMENTBYTE);
    void *cy_t    = __builtin_assume_aligned ((float*) (cpos_arr+15*SSEMULTIPL), ALIGNMENTBYTE);
    
  #elif (SSEMULTIPL==8)
    vxsf fx = (vxsf) {cam->fc[0], cam->fc[0], cam->fc[0], cam->fc[0],cam->fc[0], cam->fc[0], cam->fc[0], cam->fc[0]};
    vxsf fy = (vxsf) {cam->fc[1], cam->fc[1], cam->fc[1], cam->fc[1],cam->fc[1], cam->fc[1], cam->fc[1], cam->fc[1]};
    vxsf cx = (vxsf) {cam->cc[0], cam->cc[0], cam->cc[0], cam->cc[0],cam->cc[0], cam->cc[0], cam->cc[0], cam->cc[0]};
    vxsf cy = (vxsf) {cam->cc[1], cam->cc[1], cam->cc[1], cam->cc[1],cam->cc[1], cam->cc[1], cam->cc[1], cam->cc[1]};
    
    vxsf cpos0 = (vxsf) {cpos[0], cpos[0], cpos[0], cpos[0],cpos[0], cpos[0], cpos[0], cpos[0]};
    vxsf cpos1 = (vxsf) {cpos[1], cpos[1], cpos[1], cpos[1],cpos[1], cpos[1], cpos[1], cpos[1]};
    vxsf cpos2 = (vxsf) {cpos[2], cpos[2], cpos[2], cpos[2],cpos[2], cpos[2], cpos[2], cpos[2]};
    vxsf cpos3 = (vxsf) {cpos[3], cpos[3], cpos[3], cpos[3],cpos[3], cpos[3], cpos[3], cpos[3]};
    vxsf cpos4 = (vxsf) {cpos[4], cpos[4], cpos[4], cpos[4],cpos[4], cpos[4], cpos[4], cpos[4]};
    vxsf cpos5 = (vxsf) {cpos[5], cpos[5], cpos[5], cpos[5],cpos[5], cpos[5], cpos[5], cpos[5]};
    vxsf cpos6 = (vxsf) {cpos[6], cpos[6], cpos[6], cpos[6],cpos[6], cpos[6], cpos[6], cpos[6]};
    vxsf cpos7 = (vxsf) {cpos[7], cpos[7], cpos[7], cpos[7],cpos[7], cpos[7], cpos[7], cpos[7]};
    vxsf cpos8 = (vxsf) {cpos[8], cpos[8], cpos[8], cpos[8],cpos[8], cpos[8], cpos[8], cpos[8]};
    vxsf cpos9 = (vxsf) {cpos[9], cpos[9], cpos[9], cpos[9],cpos[9], cpos[9], cpos[9], cpos[9]};
    vxsf cpos10= (vxsf) {cpos[10],cpos[10],cpos[10],cpos[10],cpos[10],cpos[10],cpos[10],cpos[10]};
    vxsf cpos11= (vxsf) {cpos[11],cpos[11],cpos[11],cpos[11],cpos[11],cpos[11],cpos[11],cpos[11]};
  #endif
    
    //void *x = __builtin_assume_aligned (arg, 16);
    
    vxsf * X = (vxsf*) (pt3d);
    vxsf * Y = (vxsf*) (pt3d + op->maxpttrack  );
    vxsf * Z = (vxsf*) (pt3d + op->maxpttrack*2);

    vxsf * x = (vxsf*) (pt2d);
    vxsf * y = (vxsf*) (pt2d + op->maxpttrack  );

    vxsf tmp3d_X;
    vxsf tmp3d_Y;
    vxsf tmp3d_Z;
    
    for (int i=nopoints/SSEMULTIPL; i--  ; ++X, ++Y, ++Z, ++x, ++y)
    {
      // Apply camera position, move to plane at Z=1
      tmp3d_Z =  cpos8 * (*X) + cpos9 * (*Y) + cpos10 * (*Z) + cpos11;
      tmp3d_X = (cpos0 * (*X) + cpos1 * (*Y) + cpos2 * (*Z)  + cpos3) / tmp3d_Z;
      tmp3d_Y = (cpos4 * (*X) + cpos5 * (*Y) + cpos6 * (*Z)  + cpos7) / tmp3d_Z;

      // Project into 2D
      (*x) = tmp3d_X * fx + cx;
      (*y) = tmp3d_Y * fy + cy;
    }
#else   // NO SSE INSTRUCTIONS
  float * X = (float*) (pt3d);
  float * Y = (float*) (pt3d + op->maxpttrack  );
  float * Z = (float*) (pt3d + op->maxpttrack*2);
  float * x = (float*) (pt2d);
  float * y = (float*) (pt2d + op->maxpttrack  );

  float tmp3d[3];
  
  for (int i=nopoints; i--  ; ++X, ++Y, ++Z, ++x, ++y)
  {
    // Apply camera position, move to plane at Z=1
    tmp3d[2] =  cpos[8] * (*X) + cpos[9] * (*Y) + cpos[10] * (*Z) + cpos[11];
    tmp3d[0] = (cpos[0] * (*X) + cpos[1] * (*Y) + cpos[2] * (*Z)  + cpos[3]) / tmp3d[2];
    tmp3d[1] = (cpos[4] * (*X) + cpos[5] * (*Y) + cpos[6] * (*Z)  + cpos[7]) / tmp3d[2];

    // Project into 2D
    (*x) = tmp3d[0] * cam->fc[0] + cam->cc[0];
    (*y) = tmp3d[1] * cam->fc[1] + cam->cc[1];
  }
#endif
}*/
