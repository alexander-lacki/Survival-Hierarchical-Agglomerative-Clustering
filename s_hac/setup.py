from setuptools import setup, Extension
from Cython.Build import cythonize


extensions = [
    Extension("s_hac", ["./s_hac.pyx"]), 
    Extension("s_hac", ["./s_hac.cpp"]),
    Extension("s_hac", ["./fast_logrank.cpp"]),
]

s = cythonize(extensions, language_level=3)

setup(name='s_hac',
    version='0.0.1',
    description="Perform survival hierarchical agglomerative clustering",
    author="Alexander Lacki, Antonio Martinez-Millana, ITACA Institute, Universitat Politecnica de Valencia",
    author_email='alacki@upvnet.upv.es', 
    license="BSD",
    ext_modules=s)
