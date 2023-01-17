#ifndef UTIL_H_
#define UTIL_H_

#define PY_SSIZE_T_CLEAN
#include <Python.h>

#define NO_IMPORT_ARRAY
#define PY_ARRAY_UNIQUE_SYMBOL Py_Array_API_myext
#include <arrayobject.h>

#include <stdio.h>
#include <gsl/gsl_matrix.h>

// convert numpy 2d matrix to gsl_matrix
gsl_matrix *PyArrayObject_to_gsl_matrix(PyArrayObject *x);

// convert gsl_matrix to numpy 2d, new_memory flag for allocating separate memory space
PyArrayObject *gsl_matrix_to_PyArrayObject(gsl_matrix *x, const int new_memory);

//free the converted gsl_matrix, which is just a wrap without full memory allocation
void gsl_matrix_partial_free(gsl_matrix *x);

void print_matrix(const gsl_matrix *X, const char *title, FILE *fp);


#endif /* UTIL_H_ */
