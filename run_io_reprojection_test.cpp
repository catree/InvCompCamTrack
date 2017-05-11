
// g++ `pkg-config --cflags opencv eigen3` -O3   -g  -msse4  -mavx -Wall  -std=c++11 -o run_io_reprojection_test ../run_io_reprojection_test.cpp ../utilities.cpp ../camera.cpp ../pose.cpp ../odometer.cpp `pkg-config --libs opencv eigen3`
// g++ `pkg-config --cflags opencv` -I/home/kroegert/local/usrlocal/include/eigen3 -O0  -g -msse4  -Wall -std=c++11 -o run_io_reprojection_test ../run_io_reprojection_test.cpp ../utilities.cpp ../camera.cpp ../pose.cpp ../odometer.cpp `pkg-config --libs opencv`
// g++ `pkg-config --cflags opencv` -I/home/kroegert/local/usrlocal/include/eigen3 -O3 -msse4  -mavx -Wall -std=c++11 -o run_io_reprojection_test ../run_io_reprojection_test.cpp ../utilities.cpp ../camera.cpp ../pose.cpp ../odometer.cpp `pkg-config --libs opencv`

//-march=core-avx-i -mtune=core-avx-i 

//-ftree-vectorize
//-march=corei7-avx -mtune=corei7-avx
//-march=core-avx-i -mtune=core-avx-i
//-march=corei7 -mtune=corei7

//./run_io_reprojection_test /home/till/zinc/local/Code/VidReg_CodePackage/ToyDataset_LionFlorence/VidFrames_Seq/frame-00002.jpg /home/till/zinc/local/Code/VidReg_CodePackage/ToyDataset_LionFlorence/VidFrames_Seq/frame-00027.jpg  ~/zinc/local/Results/CameraTrack/myFile.txt /tmp/outfile.txt 3 0 8 5 0.1 0 0 100 0
//./run_io_reprojection_test /home/kroegert/local/Code/VidReg_CodePackage/ToyDataset_LionFlorence/VidFrames_Seq/frame-00002.jpg /home/kroegert/local/Code/VidReg_CodePackage/ToyDataset_LionFlorence/VidFrames_Seq/frame-00027.jpg  ~/local/Results/CameraTrack/myFile.txt /tmp/outfile.txt 3 0 8 5 0.1 0 0 100 0
//./run_io_reprojection_test /home/kroegert/local/Results/CameraTrack/imgtest.png /home/kroegert/local/Results/CameraTrack/imgtest.png ~/local/Results/CameraTrack/myFile.txt /tmp/outfile.txt 1 0 8 5 0.1 0 0 100 0

// 

// valgrind --tool=callgrin d./(Your binary)
// Visualize with kcachegrind

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>

#include <Eigen/Core>
#include <Eigen/LU>
#include <Eigen/Dense>

#include <sys/time.h>    // timeof day

#include <fstream>

#include "utilities.h"
#include "camera.h"
#include "pose.h"
#include "odometer.h"

#define MAXPTREAD 10000

using std::cout;
using std::endl;


    // TODO, Normalization test: Display Hessian for normalized, and non-normalized points, 
    // TODO, Normalization test: Add large value to points, see if pose accuracy drops
    // TODO, Patch Normalization test: Try patch mean normalization, and without normalization
  // TODO: Select points homogeneously in cell block


using namespace CTR;

void ReadPointCamFile(double *pt3d, float *pt2d_GT, int* nopoints_in, float *fc, float* cc, int *wh, optparam *op, double *cpos_p, const char* filename)
{
    FILE *stream = fopen(filename, "rb");
    if (stream == 0)
        cout << "ReadFile: could not open %s" << endl;
    
    uint32_t wh_tmp[2];
    fread(cpos_p,    sizeof(double), 6, stream); // note: bad style, check return value!
    fread(fc,   sizeof(float), 2, stream);
    fread(cc,   sizeof(float), 2, stream);
    fread(wh_tmp,   sizeof(uint32_t), 2, stream);
    wh[0] = (int) wh_tmp[0];
    wh[1] = (int) wh_tmp[1];
    
    uint64_t nopoints;    
    fread(&nopoints,   sizeof(uint64_t), 1, stream);
    (*nopoints_in) = (int) nopoints;
    
    fread((pt3d)                  ,   sizeof(double), nopoints, stream);
    fread((pt3d+MAXPTREAD  ) ,   sizeof(double), nopoints, stream);
    fread((pt3d+MAXPTREAD*2) ,   sizeof(double), nopoints, stream);
    fread((pt2d_GT)               ,   sizeof(float), nopoints, stream);
    fread((pt2d_GT+MAXPTREAD),   sizeof(float), nopoints, stream);

    fclose(stream);
}


// save pose
void WritePoseResult(const double* pose, const char* filename)
{
  
  FILE *stream = fopen(filename, "wb");
  if (stream == 0)
      cout << "WriteFile: could not open file" << endl;

  if ((int)fwrite(pose, sizeof(double), 6, stream) != 6)
      cout << "WriteFile: problem writing data" << endl;         

                    
  // close file
  fclose(stream);
    
}
 
int main( int argc, char** argv )
{
    optparam op;  
    
    // Parse parameter
    int acnt = 1; // start with first argument;
    char *imgfile_ao, *imgfile_bo;
    imgfile_ao = argv[acnt++];
    imgfile_bo = argv[acnt++];
    
    char * infile = argv[acnt++];
    char *outfile = argv[acnt++];
    //char * outfile = argv[acnt++];
    op.lv_f = atoi(argv[acnt++]);
    op.lv_l = atoi(argv[acnt++]); 
    op.psz = atoi(argv[acnt++]);  
    op.pszd2 = op.psz/2;
    op.pszd2m3 = op.psz + op.pszd2 - 1;
    op.novals = op.psz*op.psz;
    op.maxiter = atoi(argv[acnt++]);
    op.normdp_ratio = atof(argv[acnt++]);    
    op.donorm = atoi(argv[acnt++]);
    op.dopatchnorm = atoi(argv[acnt++]);
    op.maxpttrack = atoi(argv[acnt++]);
    #if (SSEMULTIPL>1)  // pad max. pt array, ensure divisibility by SSEMULTIPL
    int maxptpadd = op.maxpttrack % SSEMULTIPL;
    if (maxptpadd>0) op.maxpttrack = op.maxpttrack + (SSEMULTIPL - maxptpadd); 
    #endif
    op.verbosity = atoi(argv[acnt++]);
    
    //cout << op.verbosity << endl;

    //float mindprate = atof(argv[acnt++]);

    // load images & construct image pyramides
    cv::Mat img_ao_mat, img_bo_mat, img_ao_fmat, img_bo_fmat;
    img_ao_mat = cv::imread(imgfile_ao, CV_LOAD_IMAGE_GRAYSCALE);   // Read the file
    img_bo_mat = cv::imread(imgfile_bo, CV_LOAD_IMAGE_GRAYSCALE);   // Read the file    
    
    //cout << cv::getBuildInformation() << endl;
    
    img_ao_mat.convertTo(img_ao_fmat, CV_32F); // convert to float
    img_bo_mat.convertTo(img_bo_fmat, CV_32F);
    
    const float* img_ao_pyr[op.lv_f+1];
    const float* img_bo_pyr[op.lv_f+1];
    const float* img_ao_dx_pyr[op.lv_f+1];
    const float* img_ao_dy_pyr[op.lv_f+1];
    const float* img_bo_dx_pyr[op.lv_f+1];
    const float* img_bo_dy_pyr[op.lv_f+1];
    
    cv::Mat img_ao_fmat_pyr[op.lv_f+1];
    cv::Mat img_bo_fmat_pyr[op.lv_f+1];
    cv::Mat img_ao_dx_fmat_pyr[op.lv_f+1];
    cv::Mat img_ao_dy_fmat_pyr[op.lv_f+1];
    cv::Mat img_bo_dx_fmat_pyr[op.lv_f+1];
    cv::Mat img_bo_dy_fmat_pyr[op.lv_f+1];
    
    util_constructpyramide(img_ao_fmat, img_ao_fmat_pyr, img_ao_dx_fmat_pyr, img_ao_dy_fmat_pyr, img_ao_pyr, img_ao_dx_pyr, img_ao_dy_pyr, op.lv_f, 1, op.psz);
    util_constructpyramide(img_bo_fmat, img_bo_fmat_pyr, img_bo_dx_fmat_pyr, img_bo_dy_fmat_pyr, img_bo_pyr, img_bo_dx_pyr, img_bo_dy_pyr, op.lv_f, 1, op.psz);
    
    
    // allocate aligned point reprojection memory
    double *pt3d;
    float *pt2d_GT;
    //, *pt3d_ref, , *pt2d_GT;
    
    // reserve guaranteed aligned memory, requires POSIX compliant OS
//     int ret = posix_memalign((void**)(&pt3d)    , ALIGNMENTBYTE, sizeof(float)*op.maxpttrack*3);
//         ret = posix_memalign((void**)(&pt3d_ref), ALIGNMENTBYTE, sizeof(float)*op.maxpttrack*3);
//         ret = posix_memalign((void**)(&pt2d)    , ALIGNMENTBYTE, sizeof(float)*op.maxpttrack*2);
//         ret = posix_memalign((void**)(&pt2d_GT) , ALIGNMENTBYTE, sizeof(float)*op.maxpttrack*2);
    pt3d    = new double[MAXPTREAD*3]; // storage order XXXXX,YYYYY,ZZZZ
//     pt3d_ref= new float[op.maxpttrack*3]; // storage order XXXXX,YYYYY,ZZZZ
//     pt2d    = new float[op.maxpttrack*2]; // storage order XXXXX,YYYYY
    pt2d_GT = new float[MAXPTREAD*2]; // storage order XXXXX,YYYYY
    memset(pt3d   , 0   , sizeof(double)*MAXPTREAD*3);
    memset(pt2d_GT, 0   , sizeof(float)*MAXPTREAD*2);
    
    
    // read initial camera and 3d points from file
    int nopoints;
    float fc[2];
    float cc[2];
    int wh[2];                //img width/height, including image padding
    double cpos_p[6];   //current 6-DOF camera position, SE3 group coefficients (t, R)
    double cpos_p_out[6];   //result 6-DOF camera position, SE3 group coefficients (t, R)

    ReadPointCamFile(pt3d, pt2d_GT, &nopoints, fc, cc, wh, &op, cpos_p, infile);  
    
    const CamClass camobj = CamClass(op.lv_f+1, fc, cc, wh, op.psz );
    
    PoseClass posobj = PoseClass(&camobj, &op);
    
    OdometerClass odomobj = OdometerClass(&posobj, &op);
        
    double *pt3d_in = new double[nopoints*3]; // storage order XXXXX,YYYYY,ZZZZ
    float *pt2d_in = new float[nopoints*2]; // storage order XXXXX,YYYYY
    for (int i=0; i<nopoints; ++i)
    {
      pt3d_in[i           ] = pt3d[i];
      pt3d_in[i+  nopoints] = pt3d[i+  MAXPTREAD];
      pt3d_in[i+2*nopoints] = pt3d[i+2*MAXPTREAD];
      pt2d_in[i           ]= pt2d_GT[i];
      pt2d_in[i+  nopoints]= pt2d_GT[i+MAXPTREAD];
    }

    struct timeval tv_start_all, tv_end_all;
    gettimeofday(&tv_start_all, NULL);

    if (op.verbosity==1) // if timing is requested, track pose 1000 times to get more reliable measurement
    {
      
      for (int i = 0; i<1000; i++)
      {
        odomobj.Set3Dpoints(pt3d_in, nopoints); 
        odomobj.SetPose(cpos_p, img_ao_pyr, img_ao_dx_pyr, img_ao_dy_pyr, img_bo_pyr);
        odomobj.TrackPose(cpos_p_out);
      }
    }
    else
    {
        odomobj.Set3Dpoints(pt3d_in, nopoints); 
        odomobj.SetPose(cpos_p, img_ao_pyr, img_ao_dx_pyr, img_ao_dy_pyr, img_bo_pyr);
        odomobj.TrackPose(cpos_p_out);
    }

    gettimeofday(&tv_end_all, NULL);
    if (op.verbosity==1)
    {
      double tt = (tv_end_all.tv_sec-tv_start_all.tv_sec)*1000.0f + (tv_end_all.tv_usec-tv_start_all.tv_usec)/1000.0f; // time in milliseconds for 1000 runs
      printf("TIME (pose tracking) (musec): %3g\n", tt);
    }

    //printf("%.10e %.10e %.10e %.10e %.10e %.10e\n", cpos_p_out[0], cpos_p_out[1], cpos_p_out[2], cpos_p_out[3], cpos_p_out[4], cpos_p_out[5]);
    //cout << cpos_p_out[0] << " " << cpos_p_out[1] << " " << cpos_p_out[2] << " " << cpos_p_out[3] << " " << cpos_p_out[4] << " " << cpos_p_out[5] << endl;
    WritePoseResult(cpos_p_out, outfile);

    delete[] pt3d_in;
    delete[] pt2d_in;
    //posobj.setpose_se3(cpos_p);
    //util_SE3_coeff_to_Group(cpos_G, cpos_p); // runtime ~ .05 musec  (machine kilroy, options -O3 -msse4  -mavx )
    
    

/*    Eigen::Matrix<float, Eigen::Dynamic, 1> pat_tmp;    
    Eigen::Matrix<float, Eigen::Dynamic, 1> pat_dx_tmp;  
    Eigen::Matrix<float, Eigen::Dynamic, 1> pat_dy_tmp;  
    pat_tmp.resize(op.novals,1);
    pat_dx_tmp.resize(op.novals,1);
    pat_dy_tmp.resize(op.novals,1);
    
    float mid[2] = {5.0f, 8.0f};
    util_getPatch(img_ao_pyr[0], mid,  &pat_tmp, &op, camobj.getsw(0));


            struct timeval tv_start_all, tv_end_all;
            gettimeofday(&tv_start_all, NULL);

    for (int i = 0; i<1000; i++)
      //util_getPatch(img_ao_pyr[0], mid,  &pat_tmp, &op, camobj.getsw(0));  // runtime ~ 0.08 musec for one 8x8 patch,  ~ 0.04 music for 4x4 patch
      //util_getPatch_grad(img_ao_pyr[0], img_ao_dx_pyr[0], img_ao_dy_pyr[0], mid,  &pat_tmp, &pat_dx_tmp, &pat_dy_tmp, &op, camobj.getsw(0));  // runtime ~ 0.2 musec for one 8x8 patch, 0.11 musec for 4x4
      //posobj.project_pt(pt3d, pt2d, nopoints, 0);                        // runtime ~ 0.095  musec for 100 points (machine kilroy, options -O3 -msse4  -mavx )
      posobj.project_pt_save_rotated(pt3d, pt3d_ref, pt2d, nopoints, 0);                        // runtime ~ 0.12  musec for 100 points (machine kilroy, options -O3 -msse4  -mavx )
    
            gettimeofday(&tv_end_all, NULL);
            double tt = (tv_end_all.tv_sec-tv_start_all.tv_sec)*1000.0f + (tv_end_all.tv_usec-tv_start_all.tv_usec)/1000.0f;
            printf("TIME (pt reprojection) (ms): %3g\n", tt);
*/
      
      
//     //Display reprojection residuals
//     float res = 0;
//     for (int i = 0; i < nopoints; ++i)
//     {
//       cout << pt3d[i] << ", " << pt3d[i+op.maxpttrack] << ", " << pt3d[i+op.maxpttrack*2] << " -> " << pt2d_GT[i]-pt2d[i] << ", " << pt2d_GT[i+op.maxpttrack]-pt2d[i+op.maxpttrack] <<  endl;
//       res += sqrt((pt2d_GT[i]-pt2d[i])*(pt2d_GT[i]-pt2d[i]) + (pt2d_GT[i+op.maxpttrack]-pt2d[i+op.maxpttrack])*(pt2d_GT[i+op.maxpttrack]-pt2d[i+op.maxpttrack]));
//     }
//     cout << "res: " << res<< endl; 

//     int k =0;
//     for (int i = 0; i < op.psz; ++i)
//     {
//       for (int j = 0; j < op.psz; ++j, ++k)
//       {
//         cout << (pat_tmp.data())[k] << ", " ;
//       }
//       cout << endl;
//     }

//     int sc = 0;
//     int k=0;
//     for (int i = 0; i < (int)camobj.getsh(sc); ++i)
//     {
//       for (int j = 0; j < (int)camobj.getsw(sc); ++j, ++k)
//       {
//         cout << img_ao_pyr[sc][k] << ", " ;
//       }
//       cout << endl;
//     }
// cout << (int)camobj.getsh(sc) << endl;
    
      // Jacobian Test
//     {
      // FIXED OPERATIONs:
      // create jacobian for each point (without fc, which changes at each iteration), ROTATE POINTS INTO REF. CAMERA REFERENCE FRAME      
//       func_get_Jw = @(x_in) [1/x_in(3) 0 -x_in(1)/(x_in(3).^2)   -x_in(1)*x_in(2)/(x_in(3).^2)    (1.0 + x_in(1).^2 / x_in(3).^2) -x_in(2)/x_in(3); ... 
//                        0 1/x_in(3) -x_in(2)/(x_in(3).^2)   -(1.0 + x_in(2).^2 / x_in(3).^2) x_in(1)*x_in(2)/(x_in(3).^2)    x_in(1)/x_in(3); ]


      // A: create 6 eigen row with nopoints*op.novals elements, fill with Jacobian
      // B: create 6 eigen row with nopoints*op.novals elements, empty
      // C: create 2 eigen row with nopoints*op.novals elements, image gradients
      
      
      // FOR EACH SCALE
      // set B to zero
      // for each point, get gradient into C
      // multiply A with C into B, multiply focal length
      // get Hessian
      
//     }


//     free(pt3d);
//     free(pt3d_ref);
//     free(pt2d);
//     free(pt2d_GT);
    delete[] pt3d;
    //delete[] pt3d_ref;
    //delete[] pt2d;
    delete[] pt2d_GT;
    

    return 0;
}


