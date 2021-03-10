#include "util.h"

void print_matrix(const gsl_matrix *X, const char *title, FILE *fp)
{
	size_t i,j;

	if(title != NULL) fprintf(fp,"Matrix %s:\n", title);

	for(i=0;i<X->size1;i++)
	{
		for(j=0;j<X->size2;j++)
		{
			fprintf(fp, "%f%c", gsl_matrix_get(X,i,j), (j<X->size2-1?'\t':'\n'));
		}
	}
}

gsl_matrix *PyArrayObject_to_gsl_matrix(PyArrayObject *x)
{
	gsl_block *b;
	gsl_matrix *r;

	if(x->nd != 2 || x->descr->type_num != NPY_DOUBLE)
	{
		PyErr_SetString(PyExc_ValueError, "Cannot convert non 2D matrix to gsl_matrix");
		return NULL;
	}

	b = (gsl_block*)malloc(sizeof(gsl_block));
	r = (gsl_matrix*)malloc(sizeof(gsl_matrix));

	r->size1 = x->dimensions[0];
	r->tda = r->size2 = x->dimensions[1];
	r->owner = 1;
	b->data = r->data = (double*)x->data;
	r->block = b;
	b->size = r->size1 * r->size2;

	return r;
}

PyArrayObject *gsl_matrix_to_PyArrayObject(gsl_matrix *x, const int new_memory)
{
	npy_intp dimensions[2] = {x->size1, x->size2};
	PyArrayObject *ptr;

	if(new_memory){
		ptr = (PyArrayObject*)PyArray_SimpleNew(2, dimensions, NPY_DOUBLE);
		memcpy(ptr->data, x->data, x->block->size * sizeof(double));
	}else{
		ptr = (PyArrayObject*)PyArray_SimpleNewFromData(2, dimensions, NPY_DOUBLE, (void*)x->data);
	}

	return (PyArrayObject*)PyArray_Return(ptr);
}

void gsl_matrix_partial_free(gsl_matrix *x)
{
	// data fields is not my own
	free(x->block);
	free(x);
}
