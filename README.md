# ridge_significance
Ridge regression with a fast implementation of statistical significance test

Prerequisite:  
1, python >= 3.6 developer version;  
2, numpy >= 1.19;  

For simplicity, you may install anaconda (https://www.anaconda.com) to include all required python packages.

3, gcc >= 4.2;  
4, gsl-2.6: https://ftp.gnu.org/gnu/gsl  

Install:  
python setup.py install. 

Test:  
python -m unittest tests.regression. 

Usage:  
Call the regression function in python code as follows:  
beta, se, zscore, pvalue = ridge_significance.fit(X, Y, alpha, alternative, nrand, verbose). 

Input:  
X: explanatory matrix  
Y: response variable  
alpha: penalty factor in ridge regression (>= 0). If alpha is 0, we will use regular ordinary least square.  
alternative: one-tailed or two-tailed statistical test, with three options: 1, two-sided; 2, greater; 3, less.  
nrand: number of randomizations (>=0). If nrand = 0, we will use student t-test for the statistical test. Otherwise, we will use permutation test.  
verbose: 1 or 0. Report intermediate results.  

Output:  
beta: regression coefficient. 
se: standard error. 
zscore: beta/se. 
pvalue: statistical significance from permutation test (nrand>0) or student t-test (nrand=0). 
Prerequisite:
