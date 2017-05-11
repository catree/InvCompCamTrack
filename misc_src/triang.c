
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

// project noviews points pt3d into pout using Projection matrices P and subtracts observations pt2d.
// Assumes pout to have size 3*noviews for temp. storage during reprojection.
void comp_residuals(float* pout, float *res_msq, const float* pt2d, const float* pt3d, const float* P, int noviews) {
  (*res_msq) = 0;
  const float *p00=&P[0],*p01=&P[noviews],*p02=&P[2*noviews],
    *p03=&P[3*noviews],*p04=&P[4*noviews],*p05=&P[5*noviews],
    *p06=&P[6*noviews],*p07=&P[7*noviews],*p08=&P[8*noviews],
    *p09=&P[9*noviews],*p10=&P[10*noviews],*p11=&P[11*noviews];
  float *po0=&pout[0], *po1=&pout[noviews], *po2=&pout[2*noviews];
  const float *pi0=&pt2d[0], *pi1=&pt2d[noviews];
  for (int i=noviews; i--;
      ++p00,++p01, ++p02, ++p03,
      ++p04,++p05, ++p06, ++p07,
      ++p08,++p09, ++p10, ++p11,
      ++po0, ++po1, ++po2, ++pi0, ++pi1) { // take advantage of SSE instructions to process blocks of 4/8 floats.
    (*po0) = (*p00)*pt3d[0] + (*p01)*pt3d[1] + (*p02)*pt3d[2] + (*p03);
    (*po1) = (*p04)*pt3d[0] + (*p05)*pt3d[1] + (*p06)*pt3d[2] + (*p07);
    (*po2) = (*p08)*pt3d[0] + (*p09)*pt3d[1] + (*p10)*pt3d[2] + (*p11);
    (*po0) /= (*po2);
    (*po1) /= (*po2);
    (*po0) = (*pi0) - (*po0);
    (*po1) = (*pi1) - (*po1);
    (*res_msq) += (*po0)*(*po0) + (*po1)*(*po1);
	}
  (*res_msq) /= 2*noviews;
}


// computes the jacobian of the projection function of pt3d into noviews cameras
// Wrt full 3d point position
// assumes pout to be of size 3*noviews*2
void comp_jacobian_full_3D(float* pout, const float* pt3d, const float* P, int noviews) {
  const float *p00=&P[0],*p01=&P[noviews],*p02=&P[2*noviews],
    *p03=&P[3*noviews],*p04=&P[4*noviews],*p05=&P[5*noviews],
    *p06=&P[6*noviews],*p07=&P[7*noviews],*p08=&P[8*noviews],
    *p09=&P[9*noviews],*p10=&P[10*noviews],*p11=&P[11*noviews];
  float *po0=&pout[0], *po1=&pout[3*noviews];
  for (int i=noviews; i--;
      ++p00,++p01, ++p02, ++p03,
      ++p04,++p05, ++p06, ++p07,
      ++p08,++p09, ++p10, ++p11,
      ++po0, ++po1) {
    float denom = (*p08)*pt3d[0] + (*p09)*pt3d[1] + (*p10)*pt3d[2] + (*p11);
    denom *= denom;

    float c0n0 =                  (*p01)*pt3d[1] + (*p02)*pt3d[2] + (*p03);
    float c1n0 = (*p00)*pt3d[0] +                  (*p02)*pt3d[2] + (*p03);
    float c2n0 = (*p00)*pt3d[0] + (*p01)*pt3d[1]                  + (*p03);

    float c0n1 =                  (*p05)*pt3d[1] + (*p06)*pt3d[2] + (*p07);
    float c1n1 = (*p04)*pt3d[0] +                  (*p06)*pt3d[2] + (*p07);
    float c2n1 = (*p04)*pt3d[0] + (*p05)*pt3d[1]                  + (*p07);

    float c0n2 =                  (*p09)*pt3d[1] + (*p10)*pt3d[2] + (*p11);
    float c1n2 = (*p08)*pt3d[0] +                  (*p10)*pt3d[2] + (*p11);
    float c2n2 = (*p08)*pt3d[0] + (*p09)*pt3d[1]                  + (*p11);

    (*po0) = ((*p00)*c0n2 - (*p08)*c0n0) / denom; ++po0;
    (*po0) = ((*p01)*c1n2 - (*p09)*c1n0) / denom; ++po0;
    (*po0) = ((*p02)*c2n2 - (*p10)*c2n0) / denom;

    (*po1) = ((*p04)*c0n2 - (*p08)*c0n1) / denom; ++po1;
    (*po1) = ((*p05)*c1n2 - (*p09)*c1n1) / denom; ++po1;
    (*po1) = ((*p06)*c2n2 - (*p10)*c2n1) / denom;
  }
}


// computes the jacobian of the projection function of pt3d into noviews cameras
// Wrt. depth of the 3d point in the first view.
// Viewing direction from first view is fixed.
// Assumes nom to be of size noviews*2, and denom_1, denom_2 of size noviews
// Prepares computation of jacobian which are independent of depth.
void prep_jacobian_depth_only(float* nominator, float* denom1, float* denom2, const float* ptdir, const float* campos, const float* P, int noviews) {
  const float *p00=&P[0],*p01=&P[noviews],*p02=&P[2*noviews],
    *p03=&P[3*noviews],*p04=&P[4*noviews],*p05=&P[5*noviews],
    *p06=&P[6*noviews],*p07=&P[7*noviews],*p08=&P[8*noviews],
    *p09=&P[9*noviews],*p10=&P[10*noviews],*p11=&P[11*noviews];
  float *po0=&nominator[0], *po1=&nominator[noviews];
  for (int i=noviews; i--;
      ++p00,++p01, ++p02, ++p03,
      ++p04,++p05, ++p06, ++p07,
      ++p08,++p09, ++p10, ++p11,
      ++po0, ++po1, ++denom1, ++denom2) {


    (*denom1) = (*p08)*campos[0] + (*p09)*campos[1] + (*p10)*campos[2] + (*p11);
    (*denom2) = (*p08)*ptdir[0] + (*p09)*ptdir[1] + (*p10)*ptdir[2];
    //float denom = ee * depth + dd;
    //denom *= denom;

    float aa0 = (*p00)*ptdir[0] + (*p01)*ptdir[1] + (*p02)*ptdir[2];
    float aa1 = (*p04)*ptdir[0] + (*p05)*ptdir[1] + (*p06)*ptdir[2];

    float bb0 = (*p00)*campos[0] + (*p01)*campos[1] + (*p02)*campos[2] + (*p03);
    float bb1 = (*p04)*campos[0] + (*p05)*campos[1] + (*p06)*campos[2] + (*p07);

    (*po0) = (aa0 * (*denom1) - bb0 * (*denom2));// / denom;
    (*po1) = (aa1 * (*denom1) - bb1 * (*denom2));// / denom;

    //printf("denom1_ : %g\n", (*denom1));
    //printf("denom2_ : %g\n", (*denom2));
  }
}


// Scale the denominator of jacobian, and compute final jacobian.
void comb_jacobian_depth_only(float* jac, float* jactjacinv, const float depth, const float* nominator, const float* denom1, const float* denom2, int noviews) {
  (*jactjacinv) = 0;
  const float *pi0=&nominator[0], *pi1=&nominator[noviews];
  float *po0=&jac[0], *po1=&jac[noviews];
  for (int i=noviews; i--; ++pi0, ++pi1, ++po0, ++po1, ++denom1, ++denom2) {
    float denom = (*denom2) * depth + (*denom1);
    denom *= denom;
    (*po0) = (*pi0) / denom;
    (*po1) = (*pi1) / denom;

    //printf("pi0_ : %g\n", (*pi0));
    //printf("pi1_ : %g\n", (*pi1));
    //printf("denom_ : %g\n", denom);
    //printf("depth_ : %g\n", (depth));

    (*jactjacinv) += (*po0)*(*po0) + (*po1)*(*po1);
  }
  (*jactjacinv) = 1/(*jactjacinv);
}


void comp_matrix_inverse_3x3_symmetric(float* minv, const float *m) {
  minv[0] = m[8] * m[4] - m[5]*m[5];
  minv[1] = m[2] * m[5] - m[8]*m[1];
  minv[2] = m[1] * m[5] - m[2]*m[4];
  minv[4] = m[8] * m[0] - m[2]*m[2];
  minv[5] = m[1] * m[2] - m[0]*m[5];
  minv[8] = m[0] * m[4] - m[1]*m[1];
  minv[3] = minv[1];
  minv[7] = minv[5];
  minv[6] = minv[2];
  float det = m[0] * minv[0] + m[1] * minv[1] + m[2] * minv[2];
  for (int i=0; i < 9; ++i)
    minv[i] /= det;
}

void comp_jactjac(float *jactjac, const float* jac, int noviews) {
  memset(jactjac, 0, 3*3*sizeof(float));
  for (int k=0; k < 2*noviews; ++k) {
    jactjac[0] += jac[k*3]*jac[k*3];
    jactjac[1] += jac[k*3]*jac[k*3+1];
    jactjac[2] += jac[k*3]*jac[k*3+2];

    jactjac[4] += jac[k*3+1]*jac[k*3+1];
    jactjac[5] += jac[k*3+1]*jac[k*3+2];

    jactjac[8] += jac[k*3+2]*jac[k*3+2];
  }
  jactjac[3] = jactjac[1];
  jactjac[6] = jactjac[2];
  jactjac[7] = jactjac[5];

  //for (int i=0; i < 3; ++i) {
  //  for (int j=0; j < 3; ++j) {
  //    printf("%g ", jactjacinv[j+i*3]);
  //  }
  //  printf("\n ");
  //}
  //printf("\n\n ");
}


// Solve (Jac.T * Jac)^-1 * delta_p = Jac.T * residual for delta_p
void compute_update_vector(float* delta_p, float* delta_p_tmp, const float* jac, const float* jactjacinv, const float* residual, int noviews) {
  memset(delta_p_tmp, 0, 3*sizeof(float));
  const float* resptr = residual;
  for (int k=0; k < 2*noviews; ++k, ++resptr) {
    delta_p_tmp[0] += jac[3*k  ] * (*resptr);
    delta_p_tmp[1] += jac[3*k+1] * (*resptr);
    delta_p_tmp[2] += jac[3*k+2] * (*resptr);
  }
  delta_p[0] = jactjacinv[0] * delta_p_tmp[0] + jactjacinv[1] * delta_p_tmp[1] +  jactjacinv[2] * delta_p_tmp[2];
  delta_p[1] = jactjacinv[1] * delta_p_tmp[0] + jactjacinv[4] * delta_p_tmp[1] +  jactjacinv[5] * delta_p_tmp[2];
  delta_p[2] = jactjacinv[2] * delta_p_tmp[0] + jactjacinv[5] * delta_p_tmp[1] +  jactjacinv[8] * delta_p_tmp[2];
}


// Gauss-Newton grad. descent on reprojection error.
// Minimize full 3D position of point
void triangulate_full3D (float* pt3d, float* pt3d_cov, const float* pt2d, const float* P, const int noviews, const int noiter, const float minres)
{
  // Allocate memory for reprojection
  float * residual = malloc(3*noviews*sizeof(float));
  float res_msq=1e300;

  // Allocate memory for jacobian, and its inverse
  float * jac = malloc(2*3*noviews*sizeof(float)); // for full 3d jacobian
  float * jactjac = malloc(3*3*sizeof(float)); // for J.T * J
  float * jactjacinv = pt3d_cov; // (J.T * J)^-1

  // point update vector
  float * delta_p = malloc(3*sizeof(float));
  float * delta_p_tmp = malloc(3*sizeof(float));

  for (int i = 0; i < noiter && res_msq > minres; ++i) {
    // Compute residual at current point estimate
    comp_residuals(residual, &res_msq, pt2d, pt3d, P, noviews);
    printf("Iter %i, mean squared residual: %g \n",i, res_msq);

    // Compute jacobian and its inverse
    comp_jacobian_full_3D(jac, pt3d, P, noviews);
    comp_jactjac(jactjac, jac, noviews);
    comp_matrix_inverse_3x3_symmetric(jactjacinv, jactjac);
    // SPEED-UP tip: only recompute jacobian at every n'th iteration
    //for (int i=0; i < noviews; ++i) {
    //  printf("JacX: %g \t %g \t %g\n",jac[3*i], jac[3*i+1], jac[3*i+2]);
    //  printf("JacY: %g \t %g \t %g\n",jac[3*i + 3*noviews], jac[3*i+1 + 3*noviews], jac[3*i+2 + 3*noviews]);
    //}


    compute_update_vector(delta_p, delta_p_tmp, jac, jactjacinv, residual, noviews);

    // Update 3D point
    pt3d[0] += delta_p[0];
    pt3d[1] += delta_p[1];
    pt3d[2] += delta_p[2];
  }

  // Free memory
  free(residual);
  free(jac);
  free(jactjac);
  free(delta_p);
  free(delta_p_tmp);
}


void comp_LM_update(float* pt3d, float* delta_p, float* delta_p_tmp, float* jactjac_tmp, float* jactjacinv, float* residual_tmp, const float* residual, float* res_msq, const float* jactjac, const float* jac, const float* damp, const float* pt2d, const float* P, const int noviews) {
  memcpy(jactjac_tmp, jactjac, 3*3*sizeof(float));
  jactjac_tmp[0] += (*damp)*jactjac[0];
  jactjac_tmp[4] += (*damp)*jactjac[4];
  jactjac_tmp[8] += (*damp)*jactjac[8];

  comp_matrix_inverse_3x3_symmetric(jactjacinv, jactjac_tmp);

  compute_update_vector(delta_p, delta_p_tmp, jac, jactjacinv, residual, noviews);

  //printf("res %g %g %g\n", residual[0], residual[1], residual[2]);

  // Update 3D point
  pt3d[0] += delta_p[0];
  pt3d[1] += delta_p[1];
  pt3d[2] += delta_p[2];

  comp_residuals(residual_tmp, res_msq, pt2d, pt3d, P, noviews);
}


void triangulate_DLT (float* pt3d, float *AtAinv, const float* pt2d, const float* P, const int noviews)
{
  float *A = malloc(4*2*noviews*sizeof(float));
  float *AtA = malloc(3*3*sizeof(float));
  float *pt3d_tmp = malloc(3*sizeof(float));

  const float *p00=&P[0],*p01=&P[noviews],*p02=&P[2*noviews],
    *p03=&P[3*noviews],*p04=&P[4*noviews],*p05=&P[5*noviews],
    *p06=&P[6*noviews],*p07=&P[7*noviews],*p08=&P[8*noviews],
    *p09=&P[9*noviews],*p10=&P[10*noviews],*p11=&P[11*noviews];
  const float *pi0=&pt2d[0], *pi1=&pt2d[noviews];
  float *po0=&A[0], *po1=&A[4*noviews];
  for (int i=noviews; i--;
      ++p00,++p01, ++p02, ++p03,
      ++p04,++p05, ++p06, ++p07,
      ++p08,++p09, ++p10, ++p11,
      ++po0, ++po1, ++pi0, ++pi1) {
		(*po0) = (*pi0)*(*p08) - (*p00); ++po0;
		(*po0) = (*pi0)*(*p09) - (*p01); ++po0;
		(*po0) = (*pi0)*(*p10) - (*p02); ++po0;
		(*po0) = (*pi0)*(*p11) - (*p03);
		(*po1) = (*pi1)*(*p08) - (*p04); ++po1;
		(*po1) = (*pi1)*(*p09) - (*p05); ++po1;
		(*po1) = (*pi1)*(*p10) - (*p06); ++po1;
		(*po1) = (*pi1)*(*p11) - (*p07);
	}

	// Compute A.T*A and A.T*y
  memset(AtA, 0, 3*3*sizeof(float));
  memset(pt3d_tmp, 0, 3*sizeof(float));
  for (int k=0; k < 2*noviews; ++k) {
    AtA[0] += A[k*4]*A[k*4];
    AtA[1] += A[k*4]*A[k*4+1];
    AtA[2] += A[k*4]*A[k*4+2];

    AtA[4] += A[k*4+1]*A[k*4+1];
    AtA[5] += A[k*4+1]*A[k*4+2];

    AtA[8] += A[k*4+2]*A[k*4+2];

		pt3d_tmp[0] -= A[k*4  ]*A[k*4+3];
		pt3d_tmp[1] -= A[k*4+1]*A[k*4+3];
		pt3d_tmp[2] -= A[k*4+2]*A[k*4+3];
  }
  AtA[3] = AtA[1];
  AtA[6] = AtA[2];
  AtA[7] = AtA[5];

	// Inverse of AtA is used as covariance estimate at solution.
	comp_matrix_inverse_3x3_symmetric(AtAinv, AtA);

  // Solve linear system
	pt3d[0] = AtAinv[0] * pt3d_tmp[0] + AtAinv[1] * pt3d_tmp[1] + AtAinv[2] * pt3d_tmp[2];
	pt3d[1] = AtAinv[3] * pt3d_tmp[0] + AtAinv[4] * pt3d_tmp[1] + AtAinv[5] * pt3d_tmp[2];
	pt3d[2] = AtAinv[6] * pt3d_tmp[0] + AtAinv[7] * pt3d_tmp[1] + AtAinv[8] * pt3d_tmp[2];

	// free allocated memory
	free(A);
	free(AtA);
	free(pt3d_tmp);
}


// LM grad. descent on reprojection error.
// Minimize full 3D position of point
void triangulate_full3D_LM (float* pt3d, float *pt3d_cov, const float* pt2d, const float* P, const int noviews, const int noiter, const float damp_init, const float damp_fct, const float minres, const float maxdamp) // default values: damp_init = 2, damp_fct = 10, minres = 1e-5, maxdamp = 1e10
{
  // Allocate memory for reprojection
  float * residual = malloc(3*noviews*sizeof(float));
  float res_msq=1e300, res_msq_old;
  float damp = damp_init;

  // Allocate memory for jacobian, and its inverse
  float * jac = malloc(2*3*noviews*sizeof(float)); // for full 3d jacobian
  float * jactjac = malloc(3*3*sizeof(float)); // for J.T * J
  float * jactjac_tmp = malloc(3*3*sizeof(float)); // for J.T * J
  float * jactjacinv = pt3d_cov; // (J.T * J)^-1

  // point update vector
  float * delta_p = malloc(3*sizeof(float));
  float * delta_p_tmp = malloc(3*sizeof(float));
  float * pt3d_tmp = malloc(3*sizeof(float));

  comp_residuals(residual, &res_msq_old, pt2d, pt3d, P, noviews);

  for (int i = 0; i < noiter && res_msq > minres && damp < maxdamp; ++i) {
    comp_jacobian_full_3D(jac, pt3d, P, noviews);
    comp_jactjac(jactjac, jac, noviews);

    memcpy(pt3d_tmp, pt3d, 3*sizeof(float));
    comp_LM_update(pt3d_tmp, delta_p, delta_p_tmp, jactjac_tmp, jactjacinv, residual, residual, &res_msq, jactjac, jac, &damp, pt2d, P, noviews);

    if (res_msq < (res_msq_old-minres)) {
      damp /= damp_fct;
      memcpy(pt3d, pt3d_tmp, 3*sizeof(float));
    } else {
      damp *= damp_fct;
      comp_LM_update(pt3d, delta_p, delta_p_tmp, jactjac_tmp, jactjacinv, residual, residual, &res_msq, jactjac, jac, &damp, pt2d, P, noviews);
    }
    printf("Iter %i, mean squared residual: %g at damp %g\n",i, res_msq, damp);
    res_msq_old = res_msq;
  }

  // Free memory
  free(residual);
  free(jac);
  free(jactjac);
  free(jactjac_tmp);
  free(delta_p);
  free(delta_p_tmp);
  free(pt3d_tmp);
}


// Gauss-Newton grad. descent on reprojection error.
// Minimize depth along viewing direction in first frame (first P).
void triangulate_depthonly(float* pt3d, float* depth_cov, const float* campos, const float* ptdir, const float* pt2d, const float* P, const int noviews, const int noiter, const float minres)
{
  // Allocate memory for jacobian, and temporary helper variables
  float * jac = malloc(2*noviews*sizeof(float)); // depth-only jacobian
  float * nominator = malloc(2*noviews*sizeof(float)); // tmp. variable for depth independent jacobian terms
  float * denom1 = malloc(noviews*sizeof(float)); // same
  float * denom2 = malloc(noviews*sizeof(float)); // same

  // Allocate memory for reprojection
  float * residual = malloc(3*noviews*sizeof(float));
  float res_msq=1e300;

  // compute current point depth from first view
  jac[0] = pt3d[0]-campos[0]; // reusing the memory of jac here for depth initialization
  jac[1] = pt3d[1]-campos[1];
  jac[2] = pt3d[2]-campos[2];
  float depth = sqrt(jac[0]*jac[0] + jac[1]*jac[1] + jac[2]*jac[2]);

  // since pt3d may not lie exactly on ptdir at the beginning, we move it there
  pt3d[0] = ptdir[0] * depth + campos[0];
  pt3d[1] = ptdir[1] * depth + campos[1];
  pt3d[2] = ptdir[2] * depth + campos[2];

  // Prepare jacobian terms which are independent of depth
  //float jactjacinv;
  float* jactjacinv = depth_cov;
  prep_jacobian_depth_only(nominator, denom1, denom2, ptdir, campos, P, noviews);

  for (int i = 0; i < noiter && res_msq > minres; ++i) {
    // Compute residual at current depth estimate
    comp_residuals(residual, &res_msq, pt2d, pt3d, P, noviews); // in first iteration
    printf("Iter %i, mean squared residual: %g \n",i, res_msq);

    // Compute jacobian
    comb_jacobian_depth_only(jac, jactjacinv, depth, nominator, denom1, denom2, noviews);

    const float *pi0=jac, *pi1=&jac[noviews];
    const float *pr0=residual, *pr1=&residual[noviews];
    float delta_p = 0;
    for (int i=noviews; i--; ++pi0, ++pi1, ++pr0, ++pr1) {
      delta_p += (*pi0)*(*pr0) + (*pi1)*(*pr1);
    }
    delta_p *= (*jactjacinv);

    // Update 3D point
    depth += delta_p;
    pt3d[0] = ptdir[0] * depth + campos[0];
    pt3d[1] = ptdir[1] * depth + campos[1];
    pt3d[2] = ptdir[2] * depth + campos[2];
  }

  // free memory
  free(nominator);
  free(residual);
  free(jac);
  free(denom1);
  free(denom2);
}
