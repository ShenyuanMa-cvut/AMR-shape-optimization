import dolfinx.cpp
from setuptools import setup,Extension
from Cython.Build import cythonize
import numpy as np
import dolfinx

ext = [
    Extension(
        "AMR_TO._interpolation_helper",
        ["AMR_TO/_interpolation_helper.pyx"],
        language="c++",
        include_dirs=[np.get_include()]
    )
]

setup(name='AMR_TO',
      version='0.1',
      description='An experimental implementation of the adaptive mesh refinement technique in topology optimization',
      author='Shenyuan Ma',
      author_email='shenyma@fel.cvut.cz',
      license='MIT',
      packages=["AMR_TO"],
      ext_modules=cythonize(ext),
    )
