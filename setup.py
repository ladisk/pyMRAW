desc = """\
pyMRAW
======

Module for reading Photron MRAW image sequences.
-----------------------------------------------------------
We developed this module while working on this publication:

J. Javh, J. Slavič and M. Boltežar: The Subpixel Resolution of Optical-Flow-Based Modal Analysis,
Mechanical Systems and Signal Processing, Vol. 88, p. 89–99, 2017
 
Our recent research effort can be found here: http://lab.fs.uni-lj.si/ladisk/?what=incfl&flnm=research_filtered.php&keyword=optical%20methods

If you find our research useful, consider to cite us.

"""

#from distutils.core import setup, Extension
from setuptools import setup, Extension
from pyMRAW import __version__
setup(name='pyMRAW',
      version=__version__,
      author='Jaka Javh, Janko Slavič, Domen Gorjup',
      author_email='jaka.javh@fs.uni-lj.si,janko.slavic@fs.uni-lj.si, domen.gorjup@fs.uni-lj.si',
      description='Module for reading and writing Photron MRAW image sequences.',
      url='https://github.com/ladisk/pyMRAW',
      py_modules=['pyMRAW'],
      #ext_modules=[Extension('lvm_read', ['data/short.lvm'])],
      long_description=desc,
      install_requires=['numpy>=1.10.0', 'xmltodict>=0.12.0', 'numba>=0.56.4']
      )