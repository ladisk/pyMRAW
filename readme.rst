pyMRAW
======

Photron MRAW File Reader.
-------------------------

`pyMRAW` is an open-source package, enabling the efficient use of the Photron MRAW video files in Python workflows.

It's main feature is the use of memory-mapped (`np.memmap`) arrays to create memory maps to locally stored raw video files and avoid loading large amounts of data into RAM. 

.. warning::
    To take advantage of pyMRAW's memory-mapping functionality, make sure to save MRAW files either in 8-bit or 16-bit formats, corresponding to standard data types `uint8` and `uint16`! Using pyMRAW to read 12-bit MRAW files is possible, but requires loading the complete image data into RAM to produce standard Numpy arrays.

To load `.mraw` - `.cihx` files, simply use the `pymraw.load_video` function::

    import pyMRAW
    images, info = pyMRAW.load_video('data/beam.cihx')

For more info, please refer to the `Showcase.ipynb` notebook.

We developed this module while working on this publication:
J. Javh, J. Slavič and M. Boltežar: The Subpixel Resolution of Optical-Flow-Based Modal Analysis,
Mechanical Systems and Signal Processing, Vol. 88, p. 89–99, 2017

Our recent research effort can be found here: http://lab.fs.uni-lj.si/ladisk/?what=incfl&flnm=research_filtered.php&keyword=optical%20methods

If you find our research useful, consider to cite us.


|pytest|

.. |pytest| image:: https://github.com/ladisk/pyMRAW/actions/workflows/python-package.yml/badge.svg
    :target: https://github.com/ladisk/pyMRAW/actions



