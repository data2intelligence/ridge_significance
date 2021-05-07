from setuptools import setup, Extension
import os, numpy

with open("README.md", "r") as fh:
    long_description = fh.read()

module = Extension('ridge_significance', sources = ['main.c', 'ridge.c', 'util.c'], libraries = ['gsl', 'gslcblas'])

setup(
    name = 'ridge_significance',
    version = '1.0',
    author="Peng Jiang",
    author_email="peng.jiang@nih.gov",
    description = 'Ridge regression with signficance test through either permutation or t-test',
    
    long_description=long_description,
    long_description_content_type="text/markdown",
    
    url="https://github.com/data2intelligence/ridge_significance",
    
    include_dirs = [os.path.join(numpy.get_include(), 'numpy')],
    
    install_requires=['numpy', 'pandas'],
    
    ext_modules = [module],
    )
