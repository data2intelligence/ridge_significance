#define PY_SSIZE_T_CLEAN
#include <Python.h>

#define PY_ARRAY_UNIQUE_SYMBOL Py_Array_API_myext
#include <arrayobject.h>

#include <time.h>
#include <string.h>
#include "ridge.h"
#include "util.h"

int assert_2darray(const PyArrayObject *X, const char title[])
{
	char error_message[1000];
	int flag = 1;

	strcpy(error_message, title);

	if(X->nd != 2 || X->descr->type_num != PyArray_DOUBLE)
	{
		strcat(error_message, " must be two-dimensional double");
		flag = 0;
	}

	if(X->strides[1] != sizeof(double) || X->strides[0] != X->dimensions[1] * sizeof(double))
	{
		strcat(error_message, " array stride is wrong in the memory");
		flag = 0;
	}

	if(!flag) PyErr_SetString(PyExc_ValueError, error_message);
	return flag;
}


static PyObject *fit(PyObject *self, PyObject *args)
{
	// result resuls in a list
	PyObject *result = NULL;

	PyArrayObject *X, *Y;
	gsl_matrix *Xg, *Yg;
	ridge_workspace *space;

	const char *alternative;
	int mode, nrand, verbose, flag;
	double lambda;

	if (!PyArg_ParseTuple(args, "OOdsii", &X, &Y, &lambda, &alternative, &nrand, &verbose))
		return NULL;

	if (!assert_2darray(Y, "X") || !assert_2darray(Y, "Y")) return NULL;

	if (X->dimensions[0] != Y->dimensions[0]){
		PyErr_SetString(PyExc_ValueError, "X and Y must have same number of rows");
		return NULL;
	}

	// convert numpy matrix to gsl matrix, exception handling inside these functions
	Xg = PyArrayObject_to_gsl_matrix(X);
	Yg = PyArrayObject_to_gsl_matrix(Y);

	if(Xg == NULL || Yg == NULL) return NULL;

	if(strcmp(alternative, "greater") == 0){
		mode = GREATER;
	}else if(strcmp(alternative, "less") == 0){
		mode = LESS;
	}else if(strcmp(alternative, "two-sided") == 0){
		mode = TWOSIDED;
	}else{
		PyErr_SetString(PyExc_ValueError, "alternative mode can only be greater, less, two-sided");
		return NULL;
	}

	if(nrand < 0){
		PyErr_SetString(PyExc_ValueError, "nrand < 0");
		return NULL;
	}

	space = ridge_workspace_alloc(X->dimensions[0], X->dimensions[1], Y->dimensions[1]);

	// core computation
	if(nrand == 0){
		if(verbose) fprintf(stdout, "Use t-test since nrand = 0\n");
		flag = t_test(space, Xg, Yg, lambda, mode, verbose);
	}else{
		if(verbose) fprintf(stdout, "Use permutation test with nrand = %d\n", nrand);
		flag = permutation_test(space, Xg, Yg, lambda, nrand, mode, verbose);
	}

	// evaluate return flag and potential errors
	switch(flag)
	{
		case ERROR_DIMENSION:
			PyErr_SetString(PyExc_AssertionError, "X Y row name dimension mismatch, impossible!");
			break;
		case ERROR_UNRECOGNIZED_MODE:
			PyErr_SetString(PyExc_AssertionError, "Alternative mode is unknown, impossible!");
			break;
		case ERROR_DECOMPOSITION:
			PyErr_SetString(PyExc_ArithmeticError, "Cholesky decomposition failure, X'X is singular");
			break;
		case ERROR_VARIANCE:
			PyErr_SetString(PyExc_ArithmeticError, "Coefficient variation is almost zero.");
			break;
		case ERROR_DEGREE_FREEDOM:
			PyErr_SetString(PyExc_ArithmeticError, "Degree of freedom is negative.");
			break;
		default:
			break;
	}

	if(flag == SUCCESS)
	{
		// beta, z-score, p-value
		result = PyList_New(4);

		PyList_SetItem(result, 0, (PyObject *)gsl_matrix_to_PyArrayObject(space->beta, 0));
		PyList_SetItem(result, 1, (PyObject *)gsl_matrix_to_PyArrayObject(space->aver_sq, 0));
		PyList_SetItem(result, 2, (PyObject *)gsl_matrix_to_PyArrayObject(space->zscore, 0));
		PyList_SetItem(result, 3, (PyObject *)gsl_matrix_to_PyArrayObject(space->pvalue, 0));

		// keep result space
		ridge_workspace_free(space, 0);
	}else{
		// failure, remove all allocated space since no results will be returned
		ridge_workspace_free(space, 1);
	}

	// free allocated space
	gsl_matrix_partial_free(Xg);
	gsl_matrix_partial_free(Yg);

	return result;
}


static PyMethodDef ridge_methods[] = {
	{"fit",  fit, METH_VARARGS, "Ridge regression with significance test"},
	{NULL, NULL, 0, NULL}  // Sentinel
};

static struct PyModuleDef ridge_module = {
	PyModuleDef_HEAD_INIT,
	"ridge_significance",   // name of module
	NULL, // module documentation
	-1,       // size of per-interpreter state of the module, or -1 if the module keeps state in global variables.
	ridge_methods
};

PyMODINIT_FUNC PyInit_ridge_significance(void)
{
	import_array();
    return PyModule_Create(&ridge_module);
}


void standalone_run()
{
	// for debugging evaluation of parameter input
	FILE *fp;
	gsl_matrix *X = gsl_matrix_alloc(12740, 10), *Y = gsl_matrix_alloc(12740, 3556);

	fp = fopen("/Users/jiangp4/Desktop/X", "r+");
	gsl_matrix_fscanf(fp, X);
	fclose(fp);

	fp = fopen("/Users/jiangp4/Desktop/Y", "r+");
	gsl_matrix_fscanf(fp, Y);
	fclose(fp);

	ridge_workspace *space = ridge_workspace_alloc(X->size1, X->size2, Y->size2);

	clock_t begin = clock();
	permutation_test(space, X, Y, 1, 1000, GREATER, 1);

	fprintf(stdout, "Time elapsed is %f seconds\n", (double)(clock() - begin)/CLOCKS_PER_SEC);

	fp = fopen("/Users/jiangp4/Desktop/zscore", "w+");
	print_matrix(space->zscore, NULL, fp);
	fclose(fp);

	ridge_workspace_free(space, 1);
}


int main(int argc, char *argv[])
{
	wchar_t *program = Py_DecodeLocale(argv[0], NULL);

	if (program == NULL) {
		fprintf(stderr, "Fatal error: cannot decode argv[0]\n");
		exit(1);
	}

	// Add buildin module
	PyImport_AppendInittab("ridge_significance", PyInit_ridge_significance);

	// Pass argv[0] to the Python interpreter
	Py_SetProgramName(program);

	Py_Initialize();

	// Optionally import the module; alternatively, import can be deferred until the embedded script imports it
	PyImport_ImportModule("ridge_significance");

	PyMem_RawFree(program);

	// Debug area
	//standalone_run();

	return 0;
}
