import pandas
import numpy
import time
import ridge_significance
import statsmodels.api as sm

def dataframe_to_array(x):
    x = x.to_numpy()
    if x.flags.f_contiguous: x = numpy.array(x, order='C')
    return x

def difference(x, y):
    return numpy.abs(x-y)#/(x+y)

nrand = 1000
alpha = 5000
alternative = "two-sided"

Y = pandas.read_pickle('~/workspace/Data/Cancer/Single_Cell/Melanoma.GSE115978.post.pickle.gz')
#Y = pandas.read_csv('~/workspace/Data/TCGA/GDC/RNASeq/Output/SKCM.Primary.self_subtract.gz', sep='\t')
X = pandas.read_csv('~/workspace/Data/Cancer/Immune/Signaling/Output/diff.centroid', sep='\t')
X['background'] = Y.mean(axis=1)

#Y = pandas.read_csv('~/workspace/Data/Cancer/Output/CCLE.drug', sep='\t')
#X = pandas.read_csv('~/workspace/Data/Cancer/Output/CCLE.features', sep='\t')

Y = Y.iloc[:, range(5)].dropna()

common = Y.index.intersection(X.index)
Y, X = Y.loc[common], X.loc[common]



Y = (Y - Y.mean())/Y.std()
X = (X - X.mean())/X.std()


def OLS(X, Y, data_type):
    common = X.index.intersection(Y.index)
    X = X.loc[common]
    Y = Y.loc[common]
    #X = sm.add_constant(X)
    
    if data_type == 'pvalue':
        result = Y.apply(lambda y: sm.OLS(y, X).fit().pvalues)
    else:
        result = Y.apply(lambda y: sm.OLS(y, X).fit().tvalues)
    
    #result.drop('const', axis=0, inplace=True)
    
    return result


if alpha == 0:
    pvalue = dataframe_to_array(OLS(X, Y, 'pvalue'))
    tvalue = dataframe_to_array(OLS(X, Y, 'tvalue'))

Y = dataframe_to_array(Y)
X = dataframe_to_array(X)

start_time = time.time()
beta_p, se_p, zscore_p, pvalue_p = ridge_significance.fit(X, Y, alpha, alternative, nrand, 1)
print("permutation test %s seconds" % (time.time() - start_time))

start_time = time.time()
beta_t, se_t, zscore_t, pvalue_t = ridge_significance.fit(X, Y, alpha, alternative, 0, 1)
print("t test %s seconds" % (time.time() - start_time))

# should be extractly the same
#print("beta_p/beta_t")
#print(beta_p/beta_t)

if alpha == 0:
    print("zscore_t/tvalue")
    print(zscore_t/tvalue)
    
    print("pvalue_t/pvalue")
    print(pvalue_t/pvalue)

# should be roughly the same
#print(zscore_p/zscore_t)
#print(difference(pvalue_p, pvalue_t))

print(se_p/se_t)
