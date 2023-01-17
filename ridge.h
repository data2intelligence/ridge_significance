#ifndef RIDGE_H_
#define RIDGE_H_

#include <gsl/gsl_matrix.h>

// alternative hypothesis in permutation test
#define TWOSIDED 0
#define GREATER 1
#define LESS 2

// running errors
#define SUCCESS 0
#define ERROR_DIMENSION 1
#define ERROR_DECOMPOSITION 2
#define ERROR_VARIANCE 3
#define ERROR_UNRECOGNIZED_MODE 4
#define ERROR_DEGREE_FREEDOM 5

// workspace for ridge regression and results
// Input X: n*p, Y:n*m
typedef struct ridge_workspace{
	size_t n, p, m;
	gsl_matrix *I, *T, *Y_rand, *beta_rand, *aver, *aver_sq, *beta, *zscore, *pvalue;
	size_t *array_index;
	double *sigma2;

} ridge_workspace;

ridge_workspace *ridge_workspace_alloc(const size_t n, const size_t p, const size_t m);

void ridge_workspace_free(ridge_workspace *r, const int delete_result);


////////////////////////////////////////////////
// Main functions

int permutation_test(
	ridge_workspace *workspace,

	const gsl_matrix *X,
	const gsl_matrix *Y,
	const double lambda,
	const size_t nrand,
	const int mode,	// permutation test alternative hypothesis, see three modes defined above
	const int verbose);


int t_test(
	ridge_workspace *workspace,
	const gsl_matrix *X,
	const gsl_matrix *Y,
	const double lambda,
	const int mode,
	const int verbose);


#endif /* RIDGE_H_ */
