#include <math.h>
#include <assert.h>
#include <stdlib.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_cdf.h>

#include "ridge.h"
#include "util.h"

#define MIN(a,b) (((a)<(b))?(a):(b))
#define MAX(a,b) (((a)>(b))?(a):(b))
#define EPS 1e-12


void shuffle(size_t array[], const size_t n)
{
	size_t i, j;
	double t;

	// n should be much smaller than RAND_MAX
	assert(n < RAND_MAX/2);

	for (i = 0; i < n-1; i++)
	{
		j = i + rand() / (RAND_MAX / (n - i) + 1);

		t = array[j];
		array[j] = array[i];
		array[i] = t;
	}
}


int permutation_test(
		ridge_workspace *workspace,

		const gsl_matrix *X,
		const gsl_matrix *Y,
		const double lambda,
		const size_t nrand,
		const int mode,
		const int verbose)
{
	size_t n = X->size1, p = X->size2, m = Y->size2, *array_index, i, j, step;
	gsl_matrix *I, *T, *Y_rand, *beta_rand, *aver, *aver_sq, *beta, *zscore, *pvalue;

	if(n != Y->size1) return ERROR_DIMENSION;

	// connect to the workspace
	assert(workspace != NULL);

	I = workspace->I;
	T = workspace->T;
	Y_rand = workspace->Y_rand;
	beta_rand = workspace->beta_rand;
	aver = workspace->aver;
	aver_sq = workspace->aver_sq;
	array_index = workspace->array_index;
	beta = workspace->beta;
	zscore = workspace->zscore;
	pvalue = workspace->pvalue;

	// we assume the programmer should make these fields correct
	assert(I->size1 == p && I->size2 == p);
	assert(T->size1 == p && T->size2 == n);
	assert(Y_rand->size1 == n && Y_rand->size2 == m);
	assert(beta_rand->size1 == p && beta_rand->size2 == m);
	assert(aver->size1 == p && aver->size2 == m);
	assert(aver_sq->size1 == p && aver_sq->size2 == m);
	assert(beta->size1 == p && beta->size2 == m);
	assert(zscore->size1 == p && zscore->size2 == m);
	assert(pvalue->size1 == p && pvalue->size2 == m);

	////////////////////////////////////////////////
	// start computation

	// compute (X'X + lambda*I)^-1
	gsl_matrix_set_identity(I);
	gsl_blas_dsyrk(CblasLower, CblasTrans, 1, X, lambda, I);

	if(gsl_linalg_cholesky_decomp(I) == GSL_EDOM) return ERROR_DECOMPOSITION;

	gsl_linalg_cholesky_invert(I);

	// T = (X'X + lambda)^-1 * X'
	gsl_blas_dgemm(CblasNoTrans, CblasTrans, 1, I, X, 0, T);

	// beta = (X'X)^-1 X'Y
	gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1, T, Y, 0, beta);

	// no randomization request
	if(nrand == 0) return SUCCESS;


	////////////////////////////////////////////////
	// start randomization

	// all random sequences are the same
	srand(0);
	for(i=0;i<n;i++) array_index[i] = i;

	gsl_matrix_set_zero(aver);
	gsl_matrix_set_zero(aver_sq);
	gsl_matrix_set_zero(pvalue);

	step = MAX(1, nrand/10);

	for(i=0;i<nrand;i++)
	{
		if(verbose && i%step == 0) fprintf(stdout, "%lu%%\n", 100*i/nrand);

		shuffle(array_index, n);

		// create a randomized Y
		for(j=0;j<n;j++){
			gsl_vector_const_view t = gsl_matrix_const_row(Y, array_index[j]);
			gsl_matrix_set_row(Y_rand, j, &t.vector);
		}

		gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1, T, Y_rand, 0, beta_rand);

		// p-value comparison
		for(j=0; j< p * m; j++)
		{
			switch (mode) {
				case TWOSIDED:
					if(fabs(beta_rand->data[j]) >= fabs(beta->data[j])) pvalue->data[j]++;
					break;
				case GREATER:
					if(beta_rand->data[j] >= beta->data[j]) pvalue->data[j]++;
					break;
				case LESS:
					if(beta_rand->data[j] <= beta->data[j]) pvalue->data[j]++;
					break;
				default:
					return ERROR_UNRECOGNIZED_MODE;
					break;
			}
		}

		// variation
		gsl_matrix_add(aver, beta_rand);

		gsl_matrix_mul_elements(beta_rand, beta_rand);
		gsl_matrix_add(aver_sq, beta_rand);
	}

	gsl_matrix_scale(aver, 1.0/nrand);
	gsl_matrix_scale(aver_sq, 1.0/nrand);
	gsl_matrix_scale(pvalue, 1.0/nrand);

	// compute z-score
	gsl_matrix_memcpy(zscore, beta);
	gsl_matrix_sub(zscore, aver);

	gsl_matrix_mul_elements(aver, aver);
	gsl_matrix_sub(aver_sq, aver);

	for(i=0;i< aver_sq->size1 * aver_sq->size2; i++)
	{
		// first of all, confirm the computational procedure is right
		if(aver_sq->data[i] < 0) assert(aver_sq->data[i] > -EPS);

		if(aver_sq->data[i] < EPS) return ERROR_VARIANCE;

		aver_sq->data[i] = sqrt(aver_sq->data[i]);
	}

	gsl_matrix_div_elements(zscore, aver_sq);

	return SUCCESS;
}



int t_test(
		ridge_workspace *workspace,

		const gsl_matrix *X,
		const gsl_matrix *Y,
		const double lambda,
		const int mode,
		const int verbose)
{
	size_t n = X->size1, p = X->size2, m = Y->size2, i, j;
	gsl_matrix *I, *T, *Y_rand, *aver_sq, *beta, *zscore, *pvalue;
	double *sigma2, df, t, se;

	if(n != Y->size1) return ERROR_DIMENSION;

	// connect to the workspace
	assert(workspace != NULL);

	sigma2 = workspace->sigma2;
	I = workspace->I;
	T = workspace->T;
	Y_rand = workspace->Y_rand;
	aver_sq = workspace->aver_sq;
	beta = workspace->beta;
	zscore = workspace->zscore;
	pvalue = workspace->pvalue;

	// we assume the programmer should make these fields correct
	assert(I->size1 == p && I->size2 == p);
	assert(T->size1 == p && T->size2 == n);
	assert(Y_rand->size1 == n && Y_rand->size2 == m);
	assert(aver_sq->size1 == p && aver_sq->size2 == m);
	assert(beta->size1 == p && beta->size2 == m);
	assert(zscore->size1 == p && zscore->size2 == m);
	assert(pvalue->size1 == p && pvalue->size2 == m);

	////////////////////////////////////////////////
	// start computation

	// compute (X'X + lambda*I)^-1
	gsl_matrix_set_identity(I);

	gsl_blas_dsyrk(CblasLower, CblasTrans, 1, X, lambda, I);

	if(gsl_linalg_cholesky_decomp(I) == GSL_EDOM) return ERROR_DECOMPOSITION;

	gsl_linalg_cholesky_invert(I);

	// T = (X'X + lambda)^-1 * X'
	gsl_blas_dgemm(CblasNoTrans, CblasTrans, 1, I, X, 0, T);

	if(lambda == 0)
	{
		if(verbose) fprintf(stdout, "Use regular OLS since lambda = 0\n");
		df = n - p;
	}else{
		// degree of freedom as n - trace(H) (H = X*T)
		for(df=n, i=0; i<n; i++)
		{
			gsl_vector_const_view X_i = gsl_matrix_const_row(X, i);
			gsl_vector_const_view T_i = gsl_matrix_const_column(T, i);

			gsl_blas_ddot(&X_i.vector, &T_i.vector, &t);
			df -= t;
		}
	}

	if(verbose) fprintf(stdout, "degree of freedom = %f and n=%lu\n", df, n);

	if(df <= 0) return ERROR_DEGREE_FREEDOM;

	// beta = (X'X)^-1 X'Y
	gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1, T, Y, 0, beta);

	// Y_est = X*beta
	gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1, X, beta, 0, Y_rand);

	// Y_est - Y
	gsl_matrix_sub(Y_rand, Y);

	// compute sigma vector
	for(i=0; i<m; i++)
	{
		gsl_vector_const_view c = gsl_matrix_const_column(Y_rand, i);
		gsl_blas_ddot(&c.vector, &c.vector, sigma2 + i);
		sigma2[i] /= df;
	}

	for(i=0; i<p; i++)
	{
		if(lambda==0){
			// use simplified approach
			t = gsl_matrix_get(I, i, i);
		}else{
			gsl_vector_const_view c = gsl_matrix_const_row(T, i);
			gsl_blas_ddot(&c.vector, &c.vector, &t);
		}

		for(j=0;j<m;j++)
		{
			se = sigma2[j] * t;

			if(se < EPS){
				if(verbose) fprintf(stderr, "Error: standard error %e smaller than %e on %lu, %lu\n", se, EPS, i, j);

				//return ERROR_VARIANCE;
				se = NAN;
			}else{
				se = sqrt(se);
			}

			gsl_matrix_set(aver_sq, i, j, se);
		}
	}

	// compute t-value
	gsl_matrix_memcpy(zscore, beta);
	gsl_matrix_div_elements(zscore, aver_sq);

	// compute t-test p-value
	for(i=0;i<p;i++)
	{
		for(j=0;j<m;j++)
		{
			t = gsl_matrix_get(zscore, i, j);

			if(!isnan(t))
			{
				switch (mode) {
					case TWOSIDED:
						t = 2*gsl_cdf_tdist_Q(fabs(t), df);
						break;
					case GREATER:
						t = gsl_cdf_tdist_Q(t, df);
						break;
					case LESS:
						t = gsl_cdf_tdist_P(t, df);
						break;
					default:
						return ERROR_UNRECOGNIZED_MODE;
						break;
				}
			}

			gsl_matrix_set(pvalue, i, j, t);
		}
	}

	return SUCCESS;
}


// allocate the internal variable space
ridge_workspace * ridge_workspace_alloc(const size_t n, const size_t p, const size_t m)
{
	ridge_workspace *r = (ridge_workspace*)malloc(sizeof(ridge_workspace));

	// intermediate space
	r->array_index = (size_t*)malloc(n*sizeof(size_t));
	r->sigma2 = (double*)malloc(m*sizeof(double));

	r->I = gsl_matrix_alloc(p, p);
	r->T = gsl_matrix_alloc(p, n);
	r->Y_rand = gsl_matrix_alloc(n, m);
	r->beta_rand = gsl_matrix_alloc(p, m);
	r->aver = gsl_matrix_calloc(p, m);
	r->aver_sq = gsl_matrix_calloc(p, m);

	// result section
	r->beta = gsl_matrix_calloc(p, m);
	r->zscore = gsl_matrix_calloc(p, m);
	r->pvalue = gsl_matrix_calloc(p, m);

	// handle errors by non-exit procedure
	gsl_set_error_handler_off();

	return r;
}


void ridge_workspace_free(ridge_workspace *r, const int delete_result)
{
	gsl_matrix_free(r->I);
	gsl_matrix_free(r->T);
	gsl_matrix_free(r->Y_rand);
	gsl_matrix_free(r->beta_rand);
	gsl_matrix_free(r->aver);

	free(r->array_index);
	free(r->sigma2);

	if(delete_result){
		gsl_matrix_free(r->beta);
		gsl_matrix_free(r->aver_sq);
		gsl_matrix_free(r->zscore);
		gsl_matrix_free(r->pvalue);
	}else{
		// still keep the data block
		gsl_matrix_partial_free(r->beta);
		gsl_matrix_partial_free(r->aver_sq);
		gsl_matrix_partial_free(r->zscore);
		gsl_matrix_partial_free(r->pvalue);
	}

	free(r);
}
