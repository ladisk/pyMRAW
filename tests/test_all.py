"""
Unit test for pyMRAW.py
"""
import numpy as np
import sys, os
myPath = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, myPath + '/../')

import pyMRAW

def test():
    filename = './data/sample_60k_16bit.cih'
    cih = pyMRAW.get_cih(filename)
    N = cih['Total Frame']
    h = cih['Image Height']
    w = cih['Image Width']

    np.testing.assert_equal(N, 12)
    np.testing.assert_equal(h, 368)
    np.testing.assert_equal(w, 896)
    #if N > 12:
    #    N = 12
    mraw = open(filename[:-4] + '.mraw', 'rb')
    mraw.seek(0, 0)  # find the beginning of the file
    image_data = pyMRAW.load_images(mraw, h, w, N)  # load N images
    #np.memmap in load_images loads enables reading an array from disc as if from RAM. If you want all the images to load on RAM imediatly use load_images(mraw, h, w, N).copy()
    mraw.close()
    np.testing.assert_allclose(image_data[0,0,0],1889, atol=1e-8)



if __name__ == '__mains__':
    np.testing.run_module_suite()