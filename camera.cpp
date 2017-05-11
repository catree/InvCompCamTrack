#include "camera.h"

#include <iostream>
#include <cstring>
#include <math.h>

using std::cout;
using std::endl;


namespace CTR
{
  
CamClass::CamClass(const int noscales_in, const float *fc_in, const float *cc_in, const int * wh_in, const int padding_in)   
    : noscales(noscales_in),
      fc_org{ fc_in[0], fc_in[1]},
      cc_org{ cc_in[0], cc_in[1]},
      wh_org{ wh_in[0], wh_in[1]},
      padding(padding_in)
      
{
    fx = new float[noscales];
    fy = new float[noscales];
    cx = new float[noscales];
    cy = new float[noscales];
    swo = new float[noscales];
    sho = new float[noscales];
    sw = new float[noscales];
    sh = new float[noscales];
    
    
    for (int i = 0; i < noscales; ++i)
    {
      float sc_fct = 1 / pow(2,i);
      fx[i] = sc_fct * fc_org[0];
      fy[i] = sc_fct * fc_org[1];
      cx[i] = sc_fct * cc_org[0];
      cy[i] = sc_fct * cc_org[1];
      swo[i] = sc_fct * (float)wh_org[0];
      sho[i] = sc_fct * (float)wh_org[1];
      sw[i] = swo[i] + 2*padding;
      sh[i] = sho[i] + 2*padding;
    }

}

CamClass::~CamClass()
{
    delete[] fx;
    delete[] fy;
    delete[] cx;
    delete[] cy;
    delete[] swo;
    delete[] sho;
    delete[] sw;
    delete[] sh;
        
}



}
  