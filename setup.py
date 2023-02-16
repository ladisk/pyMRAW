# www.ladisk.si
# 
# pyMRAW is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3 of the License.
# 
# pyFRF is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with pyMRAW.  If not, see <http://www.gnu.org/licenses/>.


desc = """\
pyMRAW
======

Module for reading Photron MRAW image sequences.
-----------------------------------------------------------
We developed this module while working on this publication:

J. Javh, J. Slavič and M. Boltežar: The Subpixel Resolution of Optical-Flow-Based Modal Analysis,
Mechanical Systems and Signal Processing, Vol. 88, p. 89–99, 2017
 
If you find it useful, consider to cite us.

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