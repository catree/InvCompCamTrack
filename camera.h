#ifndef CAM_HEADER
#define CAM_HEADER

#include <xmmintrin.h>
#include <immintrin.h>

#include "utilities.h"

namespace CTR
{

// holds internal camera calibration and boundary information for all scales
// assumes 1) .5 downscaling factor, 2) image sizes restless divisible by 2

  
class CamClass
{
  
public:
  CamClass(const int noscales_in, const float *fc_in, const float *cc_in, const int * wh_in, const int padding_in );
  
  ~CamClass();
  
  inline float getfx(int sc) const {return fx[sc]; };
  inline float getfy(int sc) const {return fy[sc]; };
  inline float getcx(int sc) const {return cx[sc]; };
  inline float getcy(int sc) const {return cy[sc]; };
  inline float getswo(int sc) const {return swo[sc]; };
  inline float getsho(int sc) const {return sho[sc]; };
  inline float getsw(int sc) const {return sw[sc]; };
  inline float getsh(int sc) const {return sh[sc]; };
  
private:

  const int noscales;
  const float fc_org[2];
  const float cc_org[2];
  const int wh_org[2];
  const int padding;
  
  float* fx;   // 1-el. focal length
  float* fy;
  float* cx;
  float* cy;
  float* swo;  
  float* sho;  
  float* sw;  // including padding
  float* sh;  // including padding
  
  // TODO:
//   float* valid_lb;             // lower bound for valid image region, pre-compute for image padding to avoid border check 
//   float* valid_ubw;             // upper width bound for valid image region, pre-compute for image padding to avoid border check 
//   float* valid_ubh;             // upper height bound for valid image region, pre-compute for image padding to avoid border check 


};
















}


#endif
