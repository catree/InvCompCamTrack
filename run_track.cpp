
// g++ `pkg-config --cflags opencv eigen3` -O0 -g -std=c++11 -msse4 -o run_track ../run_track.cpp ../utilities.cpp `pkg-config --libs opencv eigen3`
// g++ `pkg-config --cflags opencv eigen3` -O3    -std=c++11 -msse4 -o run_track ../run_track.cpp ../utilities.cpp `pkg-config --libs opencv eigen3`


// valgrind --tool=callgrin d./(Your binary)
// Visualize with kcachegrind

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>

#include <sys/time.h>    // timeof day

#include <fstream>

#include "utilities.h"


// #include <string>
// #include <vector>
// #include <iostream>
// #include <algorithm> 
// #include <iterator>
// #include <sstream>
// #include <fstream>
// #include <iomanip>    
// #include <numeric>
// #include <valarray>
// 
// 

// TODO: 6 or 3 DOF
// TODO: test reprojection
// TODO: SE(3) algebras, algebra test


using namespace std;

using namespace CTR;

void PointCamFile(camparam &cam, optparam &op const char* filename)
{
    FILE *stream = fopen(filename, "rb");
    if (stream == 0)
        cout << "ReadFile: could not open %s" << endl;
    
    int width, height;
    float tag;
    int nc = img.channels();
    float tmp[nc];  

  
}



int main( int argc, char** argv )
{
    // Parse parameter
    int acnt = 1; // start with first argument;
    char *imgfile_ao, *imgfile_bo;
    imgfile_ao = argv[acnt++];
    imgfile_bo = argv[acnt++];
    
    char * infile = argv[acnt++];
    //char * outfile = argv[acnt++];
    int lv_f = atoi(argv[acnt++]);  // generates scale octaves in range 0:THIS, with 0: original scale
    int lv_l = atoi(argv[acnt++]);
    //float mindprate = atof(argv[acnt++]);

    // read images
    img_ao_mat = cv::imread(imgfile_ao, CV_LOAD_IMAGE_GRAYSCALE);   // Read the file
    img_bo_mat = cv::imread(imgfile_bo, CV_LOAD_IMAGE_GRAYSCALE);   // Read the file        
    img_ao_mat.convertTo(img_ao_fmat, CV_32F); // convert to float
    img_bo_mat.convertTo(img_bo_fmat, CV_32F);

    
    camparam cam;
    optparam op;  
    
    op.maxpttrack = 100000;
    int maxptpadd = op.maxpttrack % SSEMULTIPL;
    if (maxptpadd>0) op.maxpttrack = op.maxpttrack + (SSEMULTIPL - maxptpadd); // pad max. pt array, ensure divisibility by 4

    float* pt3d, *pt2d, *pt2d_GT;
    pt3d    = new float[op.maxpttrack*3]; // storage order XXXXX,YYYYY,ZZZZ
    pt2d    = new float[op.maxpttrack*2]; // storage order XXXXX,YYYYY
    pt2d_GT = new float[op.maxpttrack*2]; // storage order XXXXX,YYYYY    
    
    float cpos[3*4]; //current 6-DOF camera position, row major storage
    
    
    
    // read initial camera and 3d points from file
    PointCamFile(camparam &cam, optparam &op const char* filename);
    
    infile

    cam.fc[0] = 500;     cam.fc[1] = 500;
    cam.cc[0] = 50;      cam.cc[1] = 50;
    cam.width = 500;     cam.height = 500;
    cam.valid_lb = 5;
    cam.valid_ubw = 5-cam.width;
    cam.valid_ubw = 5-cam.height;
    cam.sc_fct = .5;
    cam.curr_lv = 2;
    
    
    

    

  
  
  
  
  
    
    int nopoints = 100;
    maxptpadd = nopoints % SSEMULTIPL;
    if (maxptpadd>0) 
      maxptpadd = nopoints + (SSEMULTIPL - maxptpadd);
    else 
      maxptpadd = nopoints;

    
            struct timeval tv_start_all, tv_end_all;
            gettimeofday(&tv_start_all, NULL);
    
    for (int i = 0; i<1000; i++)
      CTR::util_project_pt(pt3d, pt2d, cpos, maxptpadd, &cam, &op);
    
            gettimeofday(&tv_end_all, NULL);
            double tt = (tv_end_all.tv_sec-tv_start_all.tv_sec)*1000.0f + (tv_end_all.tv_usec-tv_start_all.tv_usec)/1000.0f;
            printf("TIME (pt reprojection) (ms): %3g\n", tt);
      
      
    delete[] pt3d;
    delete[] pt2d;
    delete[] pt2d_GT;
    
    return 0;
}


    


