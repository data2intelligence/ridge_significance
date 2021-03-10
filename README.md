# ridge_significance
Ridge regression with a fast implementation of statistical significance test

Prerequisite:
1, python >= 3.6 developer version
2, numpy >= 1.19

For simplicity, you may get both from https://www.anaconda.com/

3, gcc >= 4.2
4, gsl-2.6: https://ftp.gnu.org/gnu/gsl


Install:
python setup.py install

Test:
python -m unittest tests.regression

Usage:
Call the function in python code as follows:

beta, se, zscore, pvalue = ridge_significance.fit(X, Y, alpha, alternative, nrand, 1)

X: explanatory matrix
Y: response variable
alternative: 
