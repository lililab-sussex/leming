__version__ = '0.0.0'

from setuptools import setup
from setuptools import find_packages

setup(name=               'leming',
      version=            __version__,
      description=        'LEMING code',
      url=                'https://github.com/lililab-sussex/leming',
      classifiers=	  ['Intended Audience :: Science/Research',
                   	   'Programming Language :: Python',
                   	   'Topic :: Scientific/Engineering',
                   	   'Programming Language :: Python :: 3.9'],
      maintainer=         'Peter Wijeratne',
      maintainer_email=   'p.wijeratne@sussex.ac.uk',
      license=		  'MIT',
      packages=           find_packages('src'),
      package_dir=        {"": "src"},
      python_requires=    '>=3.9',
      install_requires=   ['numpy==1.24',
                           'scipy==1.9',
                           'scikit-learn==1.4',
                           'torch==2.2',
                           'matplotlib==3.7'],
      entry_points=	  {},
      zip_safe=		  False)
