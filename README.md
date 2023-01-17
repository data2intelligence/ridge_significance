# ridge_significance
Ridge regression with a fast implementation of statistical significance test.

Prerequisite:  
1, python >= 3.6 developer version. We suggest install anaconda (https://www.anaconda.com) to include all required packages.  
2, numpy >= 1.19;  
3, pandas >= 1.1.4;  
4, gcc >= 4.2;  
5, gsl-2.6: https://ftp.gnu.org/gnu/gsl (please don't use other versions, such as 2.7)  

Install:  
python setup.py install  

Test:  
python -m unittest tests.regression  

Usage:  
Call the regression function in python code as follows:  
beta, se, zscore, pvalue = ridge_significance.fit(X, Y, alpha, alternative, nrand, verbose). 

Input:  
X: explanatory matrix, numpy matrix in C-contiguous order (last-index varies the fastest).  
Y: response variable, numpy matrix in C-contiguous order (last-index varies the fastest).  

The row dimension of X and Y should be the same. Y input allows multiple columns, with each column as one input response variable. The function will return the same number of output variables in output matrices (beta, se, zscore, pvalue).

alpha: penalty factor in ridge regression (>= 0). If alpha is 0, we will use regular ordinary least square.  
alternative: one-tailed or two-tailed statistical test, with three options: 1, two-sided; 2, greater; 3, less.  
nrand: number of randomizations (>=0). If nrand = 0, we will use student t-test for the statistical test. Otherwise, we will use permutation test.  
verbose: 1 or 0. Report intermediate results.  


Output:  
beta: regression coefficient.  
se: standard error.  
zscore: beta/se.  
pvalue: statistical significance from permutation test (nrand>0) or student t-test (nrand=0).  
