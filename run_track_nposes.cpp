
// g++ `pkg-config --cflags opencv eigen3` -O3   -g  -msse4  -mavx -Wall  -std=c++11 -o run_track_nposes ../run_track_nposes.cpp ../utilities.cpp ../camera.cpp ../pose.cpp ../odometer.cpp `pkg-config --libs opencv eigen3`
// g++ `pkg-config --cflags opencv` -I/home/kroegert/local/usrlocal/include/eigen3 -O3 -msse4  -mavx -Wall -std=c++11 -o run_track_nposes ../run_track_nposes.cpp ../utilities.cpp ../camera.cpp ../pose.cpp ../odometer.cpp `pkg-config --libs opencv`
// ./run_track_nposes  /home/till/zinc/local/Results/CameraTrack/myFileRANSAC.txt /tmp/outfileRANSAC.txt
// ./run_track_nposes  /home/kroegert/local/Results/CameraTrack/myFileRANSAC.txt /tmp/outfileRANSAC.txt


#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <string>
#include <sstream>
#include <iomanip>    
#include <fstream>

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
using std::vector;
using std::string;

using namespace CTR;

void ReadInputFile(float* fc, float* cc, int* wh, int* fbframes, int &nocorresp, int &nosamples, vector<string> &filenames, vector<vector<double>> &pt3d, vector<vector<double>> &pt2d, vector<vector<double>> &poses, vector<vector<int>> &inlids, optparam &op, const char* inputfile) 
{
        std::ifstream infile(inputfile);
        string line;

        getline(infile, line); 
        std::stringstream lineStream(line);
        
        lineStream >> op.lv_f >> op.lv_l >> op.psz >> op.maxiter >> op.normdp_ratio >> op.donorm >> op.dopatchnorm >>  op.maxpttrack  >> op.verbosity;
        #if (SSEMULTIPL>1)  // pad max. pt array, ensure divisibility by SSEMULTIPL
        int maxptpadd = op.maxpttrack % SSEMULTIPL;
        if (maxptpadd>0) op.maxpttrack = op.maxpttrack + (SSEMULTIPL - maxptpadd); 
        #endif        
        op.pszd2 = op.psz/2;
        op.pszd2m3 = op.psz + op.pszd2 - 1;
        op.novals = op.psz*op.psz;

        getline(infile, line); lineStream.str(line);  lineStream.clear();
        lineStream >> fc[0] >> fc[1] >> cc[0] >> cc[1] >> wh[0] >> wh[1];
        
        getline(infile, line); lineStream.str(line);  lineStream.clear();
        lineStream >> fbframes[0] >> fbframes[1];

        
        for (int i=0; i< (fbframes[0]+fbframes[1]+1); ++i)
        {
          getline(infile, line); lineStream.str(line);  lineStream.clear();
          string strtmp;
          lineStream >> strtmp;
          filenames.push_back(strtmp);
        }

        getline(infile, line); lineStream.str(line);  lineStream.clear();
        lineStream >> nocorresp;        
        pt3d.resize(nocorresp);
        pt2d.resize(nocorresp);        
        for (int i=0; i< nocorresp; ++i)
        {
          pt3d[i].resize(3);
          pt2d[i].resize(2);

          getline(infile, line); lineStream.str(line);  lineStream.clear();
          lineStream >> pt2d[i][0] >> pt2d[i][1] >> pt3d[i][0] >> pt3d[i][1] >> pt3d[i][2];        
        }

        getline(infile, line); lineStream.str(line);  lineStream.clear();
        lineStream >> nosamples;        
        poses.resize(nosamples);
        inlids.resize(nosamples);
        for (int i=0; i< nosamples; ++i)
        {
          getline(infile, line); lineStream.str(line);  lineStream.clear();
          poses[i].resize(6);
          for (int j=0; j<6; ++j)
            lineStream >> poses[i][j];
          
          int noids;
          lineStream >> noids;
          inlids[i].resize(noids);
          for (int j=0; j<noids; ++j)
            lineStream >> inlids[i][j];
        }
        
        infile.close();
}

// SaveOutputFile
void WriteResult(const vector<vector<double>> &out_corr, const vector<vector<vector<double>>> &out_pose, const char* outfilename)
{
  std::ofstream outfile(outfilename, std::ofstream::out);
  
  int nosamples = out_corr.size();
  for (int sid=0; sid < nosamples; ++sid) // over all pose samples
  {
    outfile << std::setprecision(8);
    int noimages = out_pose[sid].size();
    for (int j=0; j < noimages; ++j)
    {
      for (int k=0; k < 6; ++k)
        outfile << out_pose[sid][j][k] <<  " ";
      outfile << endl;
    }
    
    outfile << std::setprecision(3);
    int nopoints =  out_corr[sid].size();
    for (int j=0; j < nopoints; ++j)
    {
      outfile << out_corr[sid][j] <<  " ";
    }
    outfile << endl;
  }
  outfile.close(); 
}
 
int main( int argc, char** argv )
{
    optparam op;  

    float fc[2];
    float cc[2];
    int wh[2];  
    int fbframes[2];  
    int nocorresp; // number of 2d-3d correspondences
    int nosamples; // number of pose samples to check
    vector<string> filenames;
    vector<vector<double>> pt3d;
    vector<vector<double>> pt2d;
    vector<vector<double>> poses;
    vector<vector<int>> inlids;

    // Read & prepare input data
    char *inputfile, *outputfile;
    inputfile = argv[1];
    outputfile = argv[2];

    ReadInputFile(fc, cc, wh, fbframes, nocorresp, nosamples, filenames, pt3d, pt2d, poses, inlids, op, inputfile);
    
    int noimages = filenames.size();
    vector<const float**> img_ao_pyr(noimages), img_ao_dx_pyr(noimages), img_ao_dy_pyr(noimages); // are only pointers to img_ao_fmat_pyr
    vector<cv::Mat*>      img_ao_fmat_pyr(noimages), img_ao_dx_fmat_pyr(noimages), img_ao_dy_fmat_pyr(noimages); //actual storage for images
    
    for (int i=0; i<noimages; ++i)
    {
        // load images & construct image pyramides
        cv::Mat img_ao_mat, img_ao_fmat;
        img_ao_mat = cv::imread(filenames[i], CV_LOAD_IMAGE_GRAYSCALE);   // Read the file
                                      
        img_ao_mat.convertTo(img_ao_fmat, CV_32F); // convert to float
        
        img_ao_pyr[i]    = new const float*[op.lv_f+1];
        img_ao_dx_pyr[i] = new const float*[op.lv_f+1];
        img_ao_dy_pyr[i] = new const float*[op.lv_f+1];
        
        img_ao_fmat_pyr[i]    = new cv::Mat[op.lv_f+1];
        img_ao_dx_fmat_pyr[i] = new cv::Mat[op.lv_f+1];
        img_ao_dy_fmat_pyr[i] = new cv::Mat[op.lv_f+1];
        
        //cv::Mat img_ao_fmat_pyr[op.lv_f+1];
        //cv::Mat img_ao_dx_fmat_pyr[op.lv_f+1];
        //cv::Mat img_ao_dy_fmat_pyr[op.lv_f+1];
        
        util_constructpyramide(img_ao_fmat, img_ao_fmat_pyr[i], img_ao_dx_fmat_pyr[i], img_ao_dy_fmat_pyr[i], img_ao_pyr[i], img_ao_dx_pyr[i], img_ao_dy_pyr[i], op.lv_f, 1, op.psz);
    }
    
    
    // Initialize visual odometry system
    const CamClass camobj = CamClass(op.lv_f+1, fc, cc, wh, op.psz );
    PoseClass posobj = PoseClass(&camobj, &op);
    OdometerClass odomobj = OdometerClass(&posobj, &op);
    
    
    // Track for every pose sample
    vector<vector<double>> out_corr(nosamples); // patch correlation reference frame patch to forward/backward frame patch (use min correlation)
    vector<vector<vector<double>>> out_pose(nosamples); // for each pose sample, forward and backward pose tracks
    for (int sid=0; sid < nosamples; ++sid) // over all pose samples
    {
      // resize out_corr and out_pose vectors
      int nopoints = inlids[sid].size();
      out_corr[sid].resize(nopoints);
      out_pose[sid].resize(noimages);
      for (int i=0; i<noimages; ++i) 
        out_pose[sid][i].resize(6);
      
      // copy only points used in for pose sample
      double *pt3d_in = new double[nopoints*3]; // storage order XXXXX,YYYYY,ZZZZ
      double *pt2d_back = new double[3*nopoints*2]; // storage order XXXXX,YYYYY
      double *pt2d_refe = new double[3*nopoints*2]; // storage order XXXXX,YYYYY
      double *pt2d_forw = new double[3*nopoints*2]; // storage order XXXXX,YYYYY
      for (int i=0; i<nopoints; ++i) 
      {
        int ptid = inlids[sid][i]-1;
        pt3d_in[i           ] = pt3d[ptid][0];
        pt3d_in[i+  nopoints] = pt3d[ptid][1];
        pt3d_in[i+2*nopoints] = pt3d[ptid][2];
      }
      odomobj.Set3Dpoints(pt3d_in, nopoints); 
      
      // get reprojected 2D points in reference view
      odomobj.SetPose(&(poses[sid][0]), img_ao_pyr[0], img_ao_dx_pyr[0], img_ao_dy_pyr[0], img_ao_pyr[0]); // set last pose for reference reprojection, images are not needed / dummies
      const float* tt = odomobj.Get2DPoints();
      for (int i=0; i<nopoints; ++i) 
      {
        pt2d_refe[i           ] = tt[i];
        pt2d_refe[i+  nopoints] = tt[i + op.maxpttrack];
        //int ptid = inlids[sid][i]-1;
        //cout << pt2d_refe[i]-pt2d[ptid][0] <<  " " << pt2d_refe[i+nopoints]-pt2d[ptid][1]   << endl; // DO NOT MATCH EACTLY DUE TO FEATURE DETECTOR UNCERTAINTY
      }      
      
      
      // forward track
      double cpos_p[6];
      memcpy(cpos_p, &poses[sid][0], sizeof(double) * 6); // set to reference poses
      memcpy(&out_pose[sid][fbframes[0]][0], cpos_p, sizeof(double) * 6);
      for (int fr = 0; fr < fbframes[1]; ++fr)
      {
        int fr_t = fr + fbframes[0];
        //cout << "forward: " << fr_t << " " << fr_t+1 << endl;        
        odomobj.SetPose(cpos_p, img_ao_pyr[fr_t], img_ao_dx_pyr[fr_t], img_ao_dy_pyr[fr_t], img_ao_pyr[fr_t+1]);
        odomobj.TrackPose(cpos_p); // get new pose
        memcpy(&out_pose[sid][fr_t+1][0], cpos_p, sizeof(double) * 6);
      }
      odomobj.SetPose(cpos_p, img_ao_pyr[0], img_ao_dx_pyr[0], img_ao_dy_pyr[0], img_ao_pyr[0]); // set last pose for last reprojection, images are not needed / dummies
      tt = odomobj.Get2DPoints();
      for (int i=0; i<nopoints; ++i) 
      {
        pt2d_forw[i           ] = tt[i];
        pt2d_forw[i+  nopoints] = tt[i + op.maxpttrack];
      }      

      
      // backward track
      memcpy(cpos_p, &poses[sid][0], sizeof(double) * 6); // set to reference poses
      for (int fr = 0; fr < fbframes[0]; ++fr)
      {
        int fr_t = fbframes[0] - fr;
        //cout << "backward: " << fr_t << " " << fr_t-1 << endl;
        odomobj.SetPose(cpos_p, img_ao_pyr[fr_t], img_ao_dx_pyr[fr_t], img_ao_dy_pyr[fr_t], img_ao_pyr[fr_t-1]);
        odomobj.TrackPose(cpos_p); // get new pose
        memcpy(&out_pose[sid][fr_t-1][0], cpos_p, sizeof(double) * 6);
      }
      odomobj.SetPose(cpos_p, img_ao_pyr[0], img_ao_dx_pyr[0], img_ao_dy_pyr[0], img_ao_pyr[0]); // set last pose for last reprojection, images are not needed / dummies
      tt = odomobj.Get2DPoints();
      for (int i=0; i<nopoints; ++i) 
      {
        pt2d_back[i           ] = tt[i];
        pt2d_back[i+  nopoints] = tt[i + op.maxpttrack];
      }      

      
      //cout << "Finished tracking" << endl;
      
      // compare patches, compute cross correlation
      MatrixXfTr patch_b, patch_r, patch_f;
      patch_b .resize(op.psz, op.psz);
      patch_r .resize(op.psz, op.psz);
      patch_f .resize(op.psz, op.psz);
      
      Eigen::Map<MatrixXfTr, Eigen::Aligned> ref_b(patch_b.data(), op.psz, op.psz);
      Eigen::Map<MatrixXfTr, Eigen::Aligned> ref_r(patch_r.data(), op.psz, op.psz);
      Eigen::Map<MatrixXfTr, Eigen::Aligned> ref_f(patch_f.data(), op.psz, op.psz);
      

      op.dopatchnorm=true; // mean normalized extracted patches for NCC computation.
      
      float mid[2];
      float weight[2];
      bool b_val, r_val, f_val;
      for (int i=0; i<nopoints; ++i)
      {
        b_val=true; r_val=true; f_val=true;
        
        mid[0] = pt2d_back[i           ];
        mid[1] = pt2d_back[i+  nopoints];
        if ((mid[0] > 0) & (mid[1] > 0) & (mid[0] < camobj.getswo(op.lv_l)) &  (mid[1] < camobj.getsho(op.lv_l)) )
          util_getPatch(img_ao_pyr[0]          [op.lv_l], mid,  &ref_b, &op, camobj.getsw(op.lv_l));
        else
          b_val=false;
        
        mid[0] = pt2d_refe[i           ];
        mid[1] = pt2d_refe[i+  nopoints];
        if ((mid[0] > 0) & (mid[1] > 0) & (mid[0] < camobj.getswo(op.lv_l)) &  (mid[1] < camobj.getsho(op.lv_l)) )
          util_getPatch(img_ao_pyr[fbframes[0]][op.lv_l], mid,  &ref_r, &op, camobj.getsw(op.lv_l));
        else
          r_val=false;

        
        mid[0] = pt2d_forw[i           ];
        mid[1] = pt2d_forw[i+  nopoints];
        if ((mid[0] > 0) & (mid[1] > 0) & (mid[0] < camobj.getswo(op.lv_l)) &  (mid[1] < camobj.getsho(op.lv_l)) )
          util_getPatch(img_ao_pyr[noimages-1] [op.lv_l], mid,  &ref_f, &op, camobj.getsw(op.lv_l));
        else
          f_val=false;
        

        float corr=-1;
        if (r_val == true)
        {
          
          patch_b /= patch_b.norm();
          patch_r /= patch_r.norm();
          patch_f /= patch_f.norm();
          
          float corr_br, corr_rf; 
          if (b_val == true)
          {
            corr_br = std::max(0.0f,(patch_b.array() * patch_r.array()).sum());
            weight[0] = fbframes[0]*fbframes[0];
          }
          else
          {
            corr_br = -1;
            weight[0] = 0;
          }

          if (f_val == true)
          {
            corr_rf = std::max(0.0f,(patch_r.array() * patch_f.array()).sum());
            weight[1] = fbframes[1]*fbframes[1];
          }
          else
          {
            corr_rf = -1;
            weight[1] = 0;
          }

          //if (b_val == true) cout << patch_b <<endl<<endl;
          //if (r_val == true) cout << patch_r <<endl<<endl;
          //if (f_val == true) cout << patch_f <<endl<<endl;
          
          corr = std::max(0.0f,(corr_br*weight[0] +  corr_rf*weight[1])   / (weight[0] + weight[1]));
          //cout << corr_br << " " << corr_rf << " " << corr << endl;
          
        }
        out_corr[sid][i] = corr;
        //cout << corr << endl;

      }
      
      delete[] pt3d_in;
      delete[] pt2d_back;
      delete[] pt2d_refe;
      delete[] pt2d_forw;
    }
    
    // Write out correlations and poses
    WriteResult(out_corr, out_pose, outputfile);
    
    
    /*

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
        odomobj.TrackPose(cpos_p, cpos_p_out, pt3d_in, nopoints, img_ao_pyr, img_ao_dx_pyr, img_ao_dy_pyr, img_bo_pyr);
    }
    else
        odomobj.TrackPose(cpos_p, cpos_p_out, pt3d_in, nopoints, img_ao_pyr, img_ao_dx_pyr, img_ao_dy_pyr, img_bo_pyr);      

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
    */
        
    for (int i=0; i<noimages; ++i)
    {
        delete[] img_ao_pyr[i];
        delete[] img_ao_dx_pyr[i];
        delete[] img_ao_dy_pyr[i];
        
        delete[] img_ao_fmat_pyr[i];
        delete[] img_ao_dx_fmat_pyr[i];
        delete[] img_ao_dy_fmat_pyr[i];
    }

    return 0;
}


