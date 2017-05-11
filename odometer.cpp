#include "utilities.h"
#include "pose.h"
#include "camera.h"
#include "odometer.h"

#include <sys/time.h>

#include <iostream>
#include <cstring>
#include <math.h>

using std::cout;
using std::endl;


namespace CTR
{
  
OdometerClass::OdometerClass(PoseClass * pose_in, const optparam* op_in)   
    : pose(pose_in),
      op(op_in)
{
  // *** ALLOCATE: memory for 3D, 2D points
  pt2d = new float*[op->lv_f+1];
  
  // reserve guaranteed aligned memory, requires POSIX compliant OS
    posix_memalign((void**)(&pt3d)    , ALIGNMENTBYTE, sizeof(float)*op->maxpttrack*3); // note: check return value!
    posix_memalign((void**)(&pt3d_ref), ALIGNMENTBYTE, sizeof(float)*op->maxpttrack*3);
    posix_memalign((void**)(&pt2d_new), ALIGNMENTBYTE, sizeof(float)*op->maxpttrack*2);
        
    for (int i = 0; i < (op->lv_f+1); ++i)   // memory for all reprojected points on all scales
      posix_memalign((void**)(&pt2d[i])    , ALIGNMENTBYTE, sizeof(float)*op->maxpttrack*2);
          
        
        
        
        
//     pt3d    = new float[op.maxpttrack*3]; // storage order XXXXX,YYYYY,ZZZZ
//     pt3d_ref= new float[op.maxpttrack*3]; // storage order XXXXX,YYYYY,ZZZZ
//     pt2d    = new float[op.maxpttrack*2]; // storage order XXXXX,YYYYY
//     pt2d_GT = new float[op.maxpttrack*2]; // storage order XXXXX,YYYYY  

  // *** ALLOCATE: memory reference and query patches, and steepest descent images
  posix_memalign((void**)(&pat_ref_all   ), ALIGNMENTBYTE, sizeof(float)*op->novals*op->maxpttrack);
  posix_memalign((void**)(&pat_ref_dx_all), ALIGNMENTBYTE, sizeof(float)*op->novals*op->maxpttrack);
  posix_memalign((void**)(&pat_ref_dy_all), ALIGNMENTBYTE, sizeof(float)*op->novals*op->maxpttrack);
  posix_memalign((void**)(&pat_new_all)   , ALIGNMENTBYTE, sizeof(float)*op->novals*op->maxpttrack);
  posix_memalign((void**)(&sd1_all)       , ALIGNMENTBYTE, sizeof(float)*op->novals*op->maxpttrack);
  posix_memalign((void**)(&sd2_all)       , ALIGNMENTBYTE, sizeof(float)*op->novals*op->maxpttrack);
  posix_memalign((void**)(&sd3_all)       , ALIGNMENTBYTE, sizeof(float)*op->novals*op->maxpttrack);
  posix_memalign((void**)(&sd4_all)       , ALIGNMENTBYTE, sizeof(float)*op->novals*op->maxpttrack);
  posix_memalign((void**)(&sd5_all)       , ALIGNMENTBYTE, sizeof(float)*op->novals*op->maxpttrack);
  posix_memalign((void**)(&sd6_all)       , ALIGNMENTBYTE, sizeof(float)*op->novals*op->maxpttrack);
  posix_memalign((void**)(&sd1_proj_all)  , ALIGNMENTBYTE, sizeof(float)*op->novals*op->maxpttrack);
  posix_memalign((void**)(&sd2_proj_all)  , ALIGNMENTBYTE, sizeof(float)*op->novals*op->maxpttrack);
  posix_memalign((void**)(&sd3_proj_all)  , ALIGNMENTBYTE, sizeof(float)*op->novals*op->maxpttrack);
  posix_memalign((void**)(&sd4_proj_all)  , ALIGNMENTBYTE, sizeof(float)*op->novals*op->maxpttrack);
  posix_memalign((void**)(&sd5_proj_all)  , ALIGNMENTBYTE, sizeof(float)*op->novals*op->maxpttrack);
  posix_memalign((void**)(&sd6_proj_all)  , ALIGNMENTBYTE, sizeof(float)*op->novals*op->maxpttrack);
  
  // maps to continuous 1xN memory
//   pat_ref_a    = new Eigen::Map<Eigen::MatrixXf, Eigen::Aligned>(pat_ref_all   , op->novals*op->maxpttrack, 1);
//   pat_ref_dx_a = new Eigen::Map<Eigen::MatrixXf, Eigen::Aligned>(pat_ref_dx_all, op->novals*op->maxpttrack, 1);
//   pat_ref_dy_a = new Eigen::Map<Eigen::MatrixXf, Eigen::Aligned>(pat_ref_dy_all, op->novals*op->maxpttrack, 1);
//   pat_new_a    = new Eigen::Map<Eigen::MatrixXf, Eigen::Aligned>(pat_new_all   , op->novals*op->maxpttrack, 1);
  sd1_a        = new Eigen::Map<Eigen::MatrixXf, Eigen::Aligned>(sd1_all       , op->novals*op->maxpttrack, 1);
  sd2_a        = new Eigen::Map<Eigen::MatrixXf, Eigen::Aligned>(sd2_all       , op->novals*op->maxpttrack, 1);
  sd3_a        = new Eigen::Map<Eigen::MatrixXf, Eigen::Aligned>(sd3_all       , op->novals*op->maxpttrack, 1);
  sd4_a        = new Eigen::Map<Eigen::MatrixXf, Eigen::Aligned>(sd4_all       , op->novals*op->maxpttrack, 1);
  sd5_a        = new Eigen::Map<Eigen::MatrixXf, Eigen::Aligned>(sd5_all       , op->novals*op->maxpttrack, 1);
  sd6_a        = new Eigen::Map<Eigen::MatrixXf, Eigen::Aligned>(sd6_all       , op->novals*op->maxpttrack, 1);

  sd1_proj_a        = new Eigen::Map<Eigen::MatrixXf, Eigen::Aligned>(sd1_proj_all       , op->novals*op->maxpttrack, 1);
  sd2_proj_a        = new Eigen::Map<Eigen::MatrixXf, Eigen::Aligned>(sd2_proj_all       , op->novals*op->maxpttrack, 1);
  sd3_proj_a        = new Eigen::Map<Eigen::MatrixXf, Eigen::Aligned>(sd3_proj_all       , op->novals*op->maxpttrack, 1);
  sd4_proj_a        = new Eigen::Map<Eigen::MatrixXf, Eigen::Aligned>(sd4_proj_all       , op->novals*op->maxpttrack, 1);
  sd5_proj_a        = new Eigen::Map<Eigen::MatrixXf, Eigen::Aligned>(sd5_proj_all       , op->novals*op->maxpttrack, 1);
  sd6_proj_a        = new Eigen::Map<Eigen::MatrixXf, Eigen::Aligned>(sd6_proj_all       , op->novals*op->maxpttrack, 1);

  
  // vectors of maps for each patch
  pat_ref_i   .reserve(op->maxpttrack);
  pat_ref_dx_i.reserve(op->maxpttrack);
  pat_ref_dy_i.reserve(op->maxpttrack);
  pat_new_i   .reserve(op->maxpttrack);
  sd1_i       .reserve(op->maxpttrack);
  sd2_i       .reserve(op->maxpttrack);
  sd3_i       .reserve(op->maxpttrack);
  sd4_i       .reserve(op->maxpttrack);
  sd5_i       .reserve(op->maxpttrack);
  sd6_i       .reserve(op->maxpttrack);
  sd1_proj_i       .reserve(op->maxpttrack);
  sd2_proj_i       .reserve(op->maxpttrack);
  sd3_proj_i       .reserve(op->maxpttrack);
  sd4_proj_i       .reserve(op->maxpttrack);
  sd5_proj_i       .reserve(op->maxpttrack);
  sd6_proj_i       .reserve(op->maxpttrack);
  
/*//   std::vector<Eigen::Map<MatrixXfTr, Eigen::Aligned>*> test_i;
//   test_i.resize(op->maxpttrack);
//   
//   for (int i=0; i <  op->maxpttrack; ++i)
//   {
//     
//     Eigen::Map<MatrixXfTr, Eigen::Aligned> testptr(NULL);
//     
//     new (&testptr) Eigen::Map<MatrixXfTr, Eigen::Aligned>(pat_ref_all, op->psz, op->psz);
    
//, Eigen::RowMajor

    Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor, Eigen::Aligned>, Eigen::Aligned> testptr(pat_ref_all, op->psz, op->psz);
    cout << testptr << endl;
    
    //test_i[i] = testptr;
//  }
          */          
                    
//   for (int i=0; i <  op->maxpttrack; ++i)
//   {
//     
//     
//     new (&(test_i[i])) Eigen::Map<MatrixXfTr, Eigen::Aligned>(pat_ref_all, op->psz, op->psz);
//     
//     cout << i << ", " << test_i[i] << endl;
//   }
//   cout << "FINISHED" << endl;
// abort();  

  for (int i=0; i <  op->maxpttrack; ++i)
  {
    pat_ref_i   .push_back(Eigen::Map<MatrixXfTr, Eigen::Aligned>(pat_ref_all   + (i*op->novals), op->psz, op->psz));
    pat_ref_dx_i.push_back(Eigen::Map<MatrixXfTr, Eigen::Aligned>(pat_ref_dx_all+ (i*op->novals), op->psz, op->psz));
    pat_ref_dy_i.push_back(Eigen::Map<MatrixXfTr, Eigen::Aligned>(pat_ref_dy_all+ (i*op->novals), op->psz, op->psz));
    pat_new_i   .push_back(Eigen::Map<MatrixXfTr, Eigen::Aligned>(pat_new_all   + (i*op->novals), op->psz, op->psz));
    sd1_i       .push_back(Eigen::Map<ArrayXfTr, Eigen::Aligned>(sd1_all        + (i*op->novals), op->psz, op->psz));
    sd2_i       .push_back(Eigen::Map<ArrayXfTr, Eigen::Aligned>(sd2_all        + (i*op->novals), op->psz, op->psz));
    sd3_i       .push_back(Eigen::Map<ArrayXfTr, Eigen::Aligned>(sd3_all        + (i*op->novals), op->psz, op->psz));
    sd4_i       .push_back(Eigen::Map<ArrayXfTr, Eigen::Aligned>(sd4_all        + (i*op->novals), op->psz, op->psz));
    sd5_i       .push_back(Eigen::Map<ArrayXfTr, Eigen::Aligned>(sd5_all        + (i*op->novals), op->psz, op->psz));
    sd6_i       .push_back(Eigen::Map<ArrayXfTr, Eigen::Aligned>(sd6_all        + (i*op->novals), op->psz, op->psz));
    
    sd1_proj_i       .push_back(Eigen::Map<ArrayXfTr, Eigen::Aligned>(sd1_proj_all        + (i*op->novals), op->psz, op->psz));
    sd2_proj_i       .push_back(Eigen::Map<ArrayXfTr, Eigen::Aligned>(sd2_proj_all        + (i*op->novals), op->psz, op->psz));
    sd3_proj_i       .push_back(Eigen::Map<ArrayXfTr, Eigen::Aligned>(sd3_proj_all        + (i*op->novals), op->psz, op->psz));
    sd4_proj_i       .push_back(Eigen::Map<ArrayXfTr, Eigen::Aligned>(sd4_proj_all        + (i*op->novals), op->psz, op->psz));
    sd5_proj_i       .push_back(Eigen::Map<ArrayXfTr, Eigen::Aligned>(sd5_proj_all        + (i*op->novals), op->psz, op->psz));
    sd6_proj_i       .push_back(Eigen::Map<ArrayXfTr, Eigen::Aligned>(sd6_proj_all        + (i*op->novals), op->psz, op->psz));    
  }

  ind_ref.resize(op->maxpttrack,1);
  ind_new.resize(op->maxpttrack,1);

  ResetOdometer();
}

  /* Timing (musec) for 100 patches
   * 1. Point loading, conversion:                     0.067
   * 2/3. Pose setting, point reference reprojection:  0.482
   * 4. Extract ref. patches and gradients:           26.141 (per scale)
   * 5. Steepest descent images                        8.562 (per scale)
   * 6. Compute Hessian                               16.097 (per scale)
   * 7. Projection of query points                     0.237 (per scale and iteration)
   * 8. Query patches, error image, sd projection     19.9   (per scale and iteration)
   * 9a. Sum for Lin. system                           4.00  (per scale and iteration)
   * 9b. Solve lin. system                             0.323 (per scale and iteration)
   * 10. Update Pose                                   0.064 (per scale and iteration)
   **/ 


  
void OdometerClass::Set3Dpoints(double *pt_in, const int nopoints_in)
{
  ResetOdometer();
  
  memset(meanshift.data()    , 0   , sizeof(double)*3);
  varval=0;
  float *pt3d_ptr1 = pt3d;
  float *pt3d_ptr2 = pt3d +   op->maxpttrack;
  float *pt3d_ptr3 = pt3d + 2*op->maxpttrack;
  double *ptr1, *ptr2, *ptr3;
  
  nopoints = std::min(nopoints_in, op->maxpttrack);
  
  // *** 1. Compute shift vector and variance for zero centering and isotropic variance, normalize 3d points, 
  // save as float (after normalization)
  if (op->donorm)
  {
    double nopointsdbl = static_cast<double> (nopoints);
    ptr1 = pt_in,
    ptr2 = pt_in +   nopoints_in,
    ptr3 = pt_in + 2*nopoints_in;
    for   (int i=nopoints; i--; ++ptr1)
      meanshift[0] += (*ptr1);
    for   (int i=nopoints; i--; ++ptr2)
      meanshift[1] += (*ptr2);    
    for   (int i=nopoints; i--; ++ptr3)
      meanshift[2] += (*ptr3);    
    
    meanshift[0] /= nopointsdbl;
    meanshift[1] /= nopointsdbl;
    meanshift[2] /= nopointsdbl;
    
    ptr1 = pt_in;
    ptr2 = pt_in +   nopoints_in;
    ptr3 = pt_in + 2*nopoints_in;
    
    for   (int i=nopoints; i--; ++ptr1, ++ptr2, ++ptr3)
    {
      (*ptr1) -= meanshift[0];
      (*ptr2) -= meanshift[1];
      (*ptr3) -= meanshift[2];
      
      varval += (*ptr1)*(*ptr1) + (*ptr2)*(*ptr2) + (*ptr3)*(*ptr3);
    }
    varval /= nopointsdbl;

    ptr1 = pt_in;
    ptr2 = pt_in +   nopoints_in;
    ptr3 = pt_in + 2*nopoints_in;
    for   (int i=nopoints; i--; ++ptr1, ++ptr2, ++ptr3, ++pt3d_ptr1, ++pt3d_ptr2, ++pt3d_ptr3)
    {
      (*pt3d_ptr1) = static_cast<float>((*ptr1) / varval);
      (*pt3d_ptr2) = static_cast<float>((*ptr2) / varval);
      (*pt3d_ptr3) = static_cast<float>((*ptr3) / varval);
    }
  }
  else
  { // only convert to float
    ptr1 = pt_in;
    ptr2 = pt_in +   nopoints_in;
    ptr3 = pt_in + 2*nopoints_in;
    for   (int i=nopoints; i--; ++ptr1, ++ptr2, ++ptr3, ++pt3d_ptr1, ++pt3d_ptr2, ++pt3d_ptr3)
    {
      (*pt3d_ptr1) = static_cast<float>(*ptr1);
      (*pt3d_ptr2) = static_cast<float>(*ptr2);
      (*pt3d_ptr3) = static_cast<float>(*ptr3);
    }    
  }  
}
  
void OdometerClass::SetPose(const double * p_in, const float **img_ref_in, const float **img_ref_dx_in, const float **img_ref_dy_in, const float **img_new_in)
{
  img_ref    = img_ref_in;
  img_ref_dx = img_ref_dx_in;
  img_ref_dy = img_ref_dy_in;
  img_new    = img_new_in;
  
  // *** 2. Set pose in pose object, normalize if necessary
  pose->setpose_se3(p_in, meanshift, varval); 
  
  // *** 3. Reproject into reference view to get 2D coordinates for reference patches, save correctly rotated points only at first reprojection (necessary for sd image + hessian).
  pose->project_pt_save_rotated(pt3d, pt3d_ref, pt2d[op->lv_f], nopoints, op->lv_f);
  for (int sl = op->lv_f-1; sl >= op->lv_l; --sl)
    pose->project_pt(pt3d, pt2d[sl], nopoints, sl);  
}

void OdometerClass::TrackPose(double * p_out)
{
  
  // Iterate from coarsest to finest scale
  for (int sl = op->lv_f; sl >= op->lv_l; --sl) 
  {
//     cout << "TEST1" << pat_ref_dx_i.size() << ", " << nopoints << endl;
//     cout << "TEST2" << pat_ref_dx_i[0] << endl;
//     abort();
    
    // *** 4. Extract reference patches + Gradients
    for (int i=0; i < nopoints; ++i)
    {
      float mid[2] = {pt2d[sl][i], pt2d[sl][i+op->maxpttrack]};
      
      //check for bounday violation in reference image
      if ((mid[0] < 0) | (mid[1] < 0) | 
          (mid[0] > pose->camobj->getswo(sl)) |  
          (mid[1] > pose->camobj->getsho(sl)) )
      {   // point is outside image frustum
          ind_ref[i] = false;
//                           cout << mid[0]  << ", " << mid[1] << " OUTSIDE" <<  endl;
      }
      else
      {  // points lies in image
        ind_ref[i] = true;
//         cout << nopoints << ", " << ind_ref.size() << ", " << pat_ref_i.size() << ", " << i << endl;
//         cout << mid[0]  << ", " << mid[1] << ", " << pose->camobj->getswo(sl) << ", " << pose->camobj->getsho(sl) << endl;
        
        util_getPatch_grad(img_ref[sl], img_ref_dx[sl], img_ref_dy[sl], mid,  &(pat_ref_i[i]), &(pat_ref_dx_i[i]), &(pat_ref_dy_i[i]), op, pose->camobj->getsw(sl));  // runtime ~ 0.2 musec for one 8x8 patch, 0.11 musec for 4x4
//cout << op->maxpttrack << " "  <<   op->psz << " "  <<op->pszd2 << " "  <<   op->pszd2m3 << " "  <<   op->novals << " "  << op->lv_f << " "  <<   op->lv_l << " "  << op->donorm << " "  <<   op->dopatchnorm << " "  << op->maxiter << " "  <<   op->normdp_ratio << " "  <<op->verbosity << endl;

          
          
//         float mid[2] = {0.5f, 0.5f};
                           
//         util_getPatch_grad(img_ref[sl], img_ref_dx[sl], img_ref_dy[sl], mid,  &(pat_ref_i[i]), &(pat_ref_dx_i[i]), &(pat_ref_dy_i[i]), op, pose->camobj->getsw(sl));  // runtime ~ 0.2 musec for one 8x8 patch, 0.11 musec for 4x4
//         //util_getPatch(img_ref[sl], mid,  &(pat_ref_i[i]), op, pose->camobj->getsw(sl));  // runtime ~ 0.2 musec for one 8x8 patch, 0.11 musec for 4x4
//                            cout << pat_ref_dx_i[i] << endl;
//                           break;
      }
    }
    
    
    // *** 5. Steepest descent image
    for (int i=0; i < nopoints; ++i)
    {
      if (ind_ref[i]) // patch is visible in reference view
      {
        float pt_x = pt3d_ref[i];
        float pt_y = pt3d_ref[i+  op->maxpttrack];
        float pt_z = pt3d_ref[i+2*op->maxpttrack];
        float pt_zsq = pt_z*pt_z;
        float fx = pose->camobj->getfx(sl);
        float fy = pose->camobj->getfy(sl);
        
        // Jacobian for each point, (without fc, which changes at each scale), USE POINTS ROTATED INTO CAM. REF. FRAME
        //  J = [1/x_in(3)   0            -x_in(1)/(x_in(3).^2)   -x_in(1)*x_in(2)/(x_in(3).^2)      (1.0 + x_in(1).^2 / x_in(3).^2)    -x_in(2)/x_in(3); ... 
        //       0           1/x_in(3)    -x_in(2)/(x_in(3).^2)   -(1.0 + x_in(2).^2 / x_in(3).^2)   x_in(1)*x_in(2)/(x_in(3).^2)       x_in(1)/x_in(3); ]
        
        sd1_i[i] = (pat_ref_dx_i[i] * (fx/pt_z)).array();
        sd2_i[i] = (pat_ref_dy_i[i] * (fy/pt_z)).array();
        sd3_i[i] = (pat_ref_dx_i[i] * (-pt_x/pt_zsq * fx) +
                    pat_ref_dy_i[i] * (-pt_y/pt_zsq * fy)).array();
        sd4_i[i] = (pat_ref_dx_i[i] * ( -       pt_x*pt_y / pt_zsq   * fx) + 
                    pat_ref_dy_i[i] * ((-(1.0 + pt_y*pt_y / pt_zsq)) * fy)).array();
        sd5_i[i] = (pat_ref_dx_i[i] * ((1.0 + pt_x*pt_x / pt_zsq) * fx)   + 
                    pat_ref_dy_i[i] * (       pt_x*pt_y / pt_zsq  * fy)).array(); 
        sd6_i[i] = (pat_ref_dx_i[i] * (-pt_y/pt_z * fx) + 
                    pat_ref_dy_i[i] * ( pt_x/pt_z * fy)).array();
      }
    }

    // *** 6. Get Hessian 
// struct timeval tv_start_all, tv_end_all;
// gettimeofday(&tv_start_all, NULL);
// for (int i = 0; i<1000; i++)    
    ComputeHessian();
// gettimeofday(&tv_end_all, NULL);
// double tt = (tv_end_all.tv_sec-tv_start_all.tv_sec)*1000.0f + (tv_end_all.tv_usec-tv_start_all.tv_usec)/1000.0f; // time in milliseconds for 1000 runs
// printf("TIME (hes) (musec): %3g\n", tt);    
// cout << nopoints << endl;

    // *** Compute camera update on this scale in iterations
    float normdp_init = 1e-10;
    float normdp=normdp_init;
    
    for (int it = 0 ; (it < op->maxiter) &  
                      ((normdp / normdp_init) > op->normdp_ratio); 
                      ++it)
    {
      //cout << it << endl;
      //float absres = 0;
      
      // reset projected steepest descent error images
      memset(sd1_proj_all  , 0   , sizeof(float)*op->novals*op->maxpttrack);
      memset(sd2_proj_all  , 0   , sizeof(float)*op->novals*op->maxpttrack);
      memset(sd3_proj_all  , 0   , sizeof(float)*op->novals*op->maxpttrack);
      memset(sd4_proj_all  , 0   , sizeof(float)*op->novals*op->maxpttrack);
      memset(sd5_proj_all  , 0   , sizeof(float)*op->novals*op->maxpttrack);
      memset(sd6_proj_all  , 0   , sizeof(float)*op->novals*op->maxpttrack);

      // *** 7. Project points with current pose estimate in new image
      pose->project_pt(pt3d, pt2d_new, nopoints, sl);

      // *** 8. Extract each query, error image, projection on steepest descent image
      for (int i=0; i < nopoints; ++i)
      {
        // *** Extract reference patches + Gradients
        float mid[2] = {pt2d_new[i], pt2d_new[i+op->maxpttrack]};
        
        //check for bounday violation in reference image
        if ((mid[0] < 0) | (mid[1] < 0) | 
            (mid[0] > pose->camobj->getswo(sl)) |  
            (mid[1] > pose->camobj->getsho(sl)) )
        {   // point is outside image frustum
            ind_new[i] = false;
        }
        else
        {  // points lies in image
          ind_new[i] = true;
          util_getPatch(img_new[sl], mid,  &(pat_new_i[i]), op, pose->camobj->getsw(sl));  // runtime ~ 0.2 musec for one 8x8 patch, 0.11 musec for 4x4
          
          // *** Compute error
          pdiff = (pat_ref_i[i]-pat_new_i[i]).array();
          
          //absres += (pdiff*pdiff).sum();
          
          // get sd patch arrays for writing the projected error images
          sd1_proj_i[i] = sd1_i[i] * pdiff;
          sd2_proj_i[i] = sd2_i[i] * pdiff;
          sd3_proj_i[i] = sd3_i[i] * pdiff;
          sd4_proj_i[i] = sd4_i[i] * pdiff;
          sd5_proj_i[i] = sd5_i[i] * pdiff;
          sd6_proj_i[i] = sd6_i[i] * pdiff;
        }
        
        
        
      }

      // ***9a.  Sum for Lin. system
      sumsd[0] = sd1_proj_a->sum();
      sumsd[1] = sd2_proj_a->sum();
      sumsd[2] = sd3_proj_a->sum();
      sumsd[3] = sd4_proj_a->sum();
      sumsd[4] = sd5_proj_a->sum();
      sumsd[5] = sd6_proj_a->sum();

      // ***9b.  Sum for Lin. system
      SolveLinSystem();
      
      // *** 10. Update pose
      pose->addpose_se3(delta_p.data());     
      
      normdp = delta_p.lpNorm<1>();
      if (it==0)
        normdp_init = normdp;
      
      if (op->verbosity == 2)
        printf("Sc%02i,It%02i: %g\n", sl,it,normdp);
      
    }
  }
  
  
  // *** Get Pose, write output
  pose->getPose_se3(p_out);
  
}

void OdometerClass::ComputeHessian()
{
    Hes(0,0) = (sd1_a->array() * sd1_a->array()).sum();
    Hes(0,1) = (sd1_a->array() * sd2_a->array()).sum();
    Hes(0,2) = (sd1_a->array() * sd3_a->array()).sum();
    Hes(0,3) = (sd1_a->array() * sd4_a->array()).sum();
    Hes(0,4) = (sd1_a->array() * sd5_a->array()).sum();
    Hes(0,5) = (sd1_a->array() * sd6_a->array()).sum();
    
    Hes(1,1) = (sd2_a->array() * sd2_a->array()).sum();
    Hes(1,2) = (sd2_a->array() * sd3_a->array()).sum();
    Hes(1,3) = (sd2_a->array() * sd4_a->array()).sum();
    Hes(1,4) = (sd2_a->array() * sd5_a->array()).sum();
    Hes(1,5) = (sd2_a->array() * sd6_a->array()).sum();

    Hes(2,2) = (sd3_a->array() * sd3_a->array()).sum();
    Hes(2,3) = (sd3_a->array() * sd4_a->array()).sum();
    Hes(2,4) = (sd3_a->array() * sd5_a->array()).sum();
    Hes(2,5) = (sd3_a->array() * sd6_a->array()).sum();

    Hes(3,3) = (sd4_a->array() * sd4_a->array()).sum();
    Hes(3,4) = (sd4_a->array() * sd5_a->array()).sum();
    Hes(3,5) = (sd4_a->array() * sd6_a->array()).sum();

    Hes(4,4) = (sd5_a->array() * sd5_a->array()).sum();
    Hes(4,5) = (sd5_a->array() * sd6_a->array()).sum();

    Hes(5,5) = (sd6_a->array() * sd6_a->array()).sum();
    
    
    Hes(1,0) = Hes(0,1);
    Hes(2,0) = Hes(0,2);
    Hes(3,0) = Hes(0,3);
    Hes(4,0) = Hes(0,4);
    Hes(5,0) = Hes(0,5);
    Hes(2,1) = Hes(1,2);
    Hes(3,1) = Hes(1,3);
    Hes(4,1) = Hes(1,4);
    Hes(5,1) = Hes(1,5);
    Hes(3,2) = Hes(2,3);
    Hes(4,2) = Hes(2,4);
    Hes(5,2) = Hes(2,5);
    Hes(4,3) = Hes(3,4);
    Hes(5,3) = Hes(3,5);
    Hes(5,4) = Hes(4,5);

//     Hes <<    9,   100,    78,    81  ,  14 ,   63,
//     23,     8,    82,    44 ,   87 ,   36,
//     92,   45,    87 ,   92 ,   58  ,  52,
//     16,    11,     9 ,   19  ,  55 ,   41,
//     83,    97,    40 ,   27 ,   15 ,    8,
//     54,    1,    26  ,  15 ,   86  ,  24;
//     
//     cout << Hes << endl;
//     
// //     if (pc->Hes.determinant()==0)
// //       for (int i = 0; i<6; ++i)
// //         pc->Hes(i,i)+=1e-10;
//     
//   
//     sumsd <<   12.1500,
//    11.1200,
//    14.1300,
//     6.6200,
//     6.2800,
//     7.6800;
//     
//             struct timeval tv_start_all, tv_end_all;
//             gettimeofday(&tv_start_all, NULL);    
// 
//     for (int i = 0; i<1000; i++)
//       SolveLinSystem();
//     
//             gettimeofday(&tv_end_all, NULL);
//             double tt = (tv_end_all.tv_sec-tv_start_all.tv_sec)*1000.0f + (tv_end_all.tv_usec-tv_start_all.tv_usec)/1000.0f;
//             printf("TIME (lin. system) (ms): %3g\n", tt);
//     
//     
     //cout << Hes << endl;
}

void OdometerClass::SolveLinSystem()
{
  //delta_p = Hes.llt().solve(sumsd);    // 0.06 musec
  //delta_p = Hes.ldlt().solve(sumsd);    // 0.15 musec
  //delta_p = Hes.colPivHouseholderQr().solve(sumsd);    // 0.5 musec
  delta_p = Hes.fullPivLu().solve(sumsd);  // 0.31 musec
}


OdometerClass::~OdometerClass()
{
  // *** FREE: memory for 3D, 2D points
  free(pt3d);
  free(pt3d_ref);
  free(pt2d_new);
  
  
  for (int i = 0; i < (op->lv_f+1); ++i)
    free(pt2d[i]);
  delete[] pt2d;
    
  // *** FREE: memory reference and query patches
  free(pat_ref_all);
  free(pat_ref_dx_all);
  free(pat_ref_dy_all);
  free(pat_new_all);
  free(sd1_all);
  free(sd2_all);
  free(sd3_all);
  free(sd4_all);
  free(sd5_all);
  free(sd6_all);
  free(sd1_proj_all);
  free(sd2_proj_all);
  free(sd3_proj_all);
  free(sd4_proj_all);
  free(sd5_proj_all);
  free(sd6_proj_all);
  
  
//   delete(pat_ref_a);
//   delete(pat_ref_dx_a);
//   delete(pat_ref_dy_a);
//   delete(pat_new_a);
  delete(sd1_a);
  delete(sd2_a);
  delete(sd3_a);
  delete(sd4_a);
  delete(sd5_a);
  delete(sd6_a);
  delete(sd1_proj_a);
  delete(sd2_proj_a);
  delete(sd3_proj_a);
  delete(sd4_proj_a);
  delete(sd5_proj_a);
  delete(sd6_proj_a);
      
    
/*    
  for (int i=0; i <  op->maxpttrack; ++i)
  {
    free(pat_ref[i]   .data());
    free(pat_ref_dx[i].data());
    free(pat_ref_dy[i].data());
    free(pat_new[i]   .data());
  }*/

    
}


void OdometerClass::ResetOdometer()
{
  memset(Hes.data()    , 0   , sizeof(float)*6*6);
  
  memset(ind_ref.data(), 1   , sizeof(bool)*op->maxpttrack);
  memset(ind_new.data(), 1   , sizeof(bool)*op->maxpttrack);
  
  
  
  
  //Set all patch memory to zero
  memset(pat_ref_all   , 0   , sizeof(float)*op->novals*op->maxpttrack);
  memset(pat_ref_dx_all, 0   , sizeof(float)*op->novals*op->maxpttrack);
  memset(pat_ref_dy_all, 0   , sizeof(float)*op->novals*op->maxpttrack);
  memset(pat_new_all   , 0   , sizeof(float)*op->novals*op->maxpttrack);
  
  memset(sd1_all       , 0   , sizeof(float)*op->novals*op->maxpttrack);
  memset(sd2_all       , 0   , sizeof(float)*op->novals*op->maxpttrack);
  memset(sd3_all       , 0   , sizeof(float)*op->novals*op->maxpttrack);
  memset(sd4_all       , 0   , sizeof(float)*op->novals*op->maxpttrack);
  memset(sd5_all       , 0   , sizeof(float)*op->novals*op->maxpttrack);
  memset(sd6_all       , 0   , sizeof(float)*op->novals*op->maxpttrack);
  
  memset(sd1_proj_all  , 0   , sizeof(float)*op->novals*op->maxpttrack);
  memset(sd2_proj_all  , 0   , sizeof(float)*op->novals*op->maxpttrack);
  memset(sd3_proj_all  , 0   , sizeof(float)*op->novals*op->maxpttrack);
  memset(sd4_proj_all  , 0   , sizeof(float)*op->novals*op->maxpttrack);
  memset(sd5_proj_all  , 0   , sizeof(float)*op->novals*op->maxpttrack);
  memset(sd6_proj_all  , 0   , sizeof(float)*op->novals*op->maxpttrack);
}
  
  


}
  