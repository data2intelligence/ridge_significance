import unittest
import os
import pandas
import numpy
import time
import pathlib
import random

import ridge_significance

random.seed(0)

eps = 1e-8
nrand = 1000
alpha = 10000
alternative = "two-sided"

fpath = pathlib.Path(__file__).parent.absolute()

output = os.path.join(fpath, 'data', 'output')

def dataframe_to_array(x, dtype = None):
    """ convert data frame to numpy matrix in C order, in case numpy and gsl use different matrix order """
    
    x = x.to_numpy(dtype=dtype)
    if x.flags.f_contiguous: x = numpy.array(x, order='C')
    return x


def difference(x, y, max_mode=True):
    """ difference between two numpy matrix """
    
    diff = numpy.abs(x-y)
    
    if max_mode:
        return diff.max()
    else:
        return diff.mean()



def save_results(beta, se, zscore, pvalue, out):
    """ a tool function in development, save results """
    
    for title, mat in [
        ['beta', beta],
        ['se', se],
        ['zscore', zscore],
        ['pvalue', pvalue],
        ]:
        numpy.save(out + '.' + title, mat)


def load_results(out):
    """ load pre-computed results for comparison """
    
    result = []
    
    for title in ['beta', 'se', 'zscore', 'pvalue']:
        result.append(numpy.load(out + '.' + title + '.npy'))
    
    return result


class TestRegressionMethods(unittest.TestCase):
    
    def test_bulk(self):
        # an example differential logFC vector from cell lines infected with different viruses
        Y = os.path.join(fpath, 'data', 'infection_GSE147507.gz')
        Y = pandas.read_csv(Y, sep='\t', index_col=0)
        
        # a sub matrix of CellSig prediction model
        X = os.path.join(fpath, 'data', 'signaling_signature.gz')
        X = pandas.read_csv(X, sep='\t', index_col=0)
        
        common = Y.index.intersection(X.index)
        Y, X = Y.loc[common], X.loc[common]
        
        # correct background differential change, but please don't do this if you normalized all gene values by zero-mean
        X['background'] = Y.mean(axis=1)
        
        # adjust variables in a reasonable range
        Y = (Y - Y.mean())/Y.std()
        X = (X - X.mean())/X.std()
    
        Y = dataframe_to_array(Y)
        X = dataframe_to_array(X)
        
        # Test A: permutation test
        start_time = time.time()
        beta_p, se_p, zscore_p, pvalue_p = ridge_significance.fit(X, Y, alpha, alternative, nrand, 1)
        print("permutation test %s seconds" % (time.time() - start_time))
        
        #save_results(beta_p, se_p, zscore_p, pvalue_p, output + '.permutation')
        beta, se, zscore, pvalue = load_results(output + '.permutation')
        
        self.assertTrue(difference(beta_p, beta) < eps)
        
        # permutation test results might fluctuate in different platforms
        self.assertTrue(difference(se_p, se, False) < 1e-2)
        self.assertTrue(difference(zscore_p, zscore, False) < 1)
        self.assertTrue(difference(pvalue_p, pvalue, False) < 1e-2)
        
        # Test B: student t-test
        start_time = time.time()
        beta_t, se_t, zscore_t, pvalue_t = ridge_significance.fit(X, Y, alpha, alternative, 0, 1)
        print("t test %s seconds" % (time.time() - start_time))
        
        #save_results(beta_t, se_t, zscore_t, pvalue_t, output + '.t')
        beta, se, zscore, pvalue = load_results(output + '.t')
        
        self.assertTrue(difference(beta_t, beta) < eps)
        self.assertTrue(difference(se_t, se) < eps)
        self.assertTrue(difference(zscore_t, zscore) < eps)
        self.assertTrue(difference(pvalue_t, pvalue) < eps)
        
        

if __name__ == '__main__':
    unittest.main()
