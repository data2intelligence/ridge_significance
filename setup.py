from distutils.core import setup, Extension
import os, numpy

module = Extension('ridge_significance', sources = ['main.c', 'ridge.c', 'util.c'], libraries = ['gsl', 'gslcblas'])

setup(name = 'ridge_significance',
      version = '1.0',
      description = 'Ridge regression with signficance test through either permutation or t-test',
      ext_modules = [module],
      include_dirs = [os.path.join(numpy.get_include(), 'numpy')],
      
      install_requires=[
        'numpy',
        ],
      )
