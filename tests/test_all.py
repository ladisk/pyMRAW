"""
Unit test for pyMRAW.py
"""
import pytest
import numpy as np
import sys, os
import tempfile

myPath = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, myPath + '/../')

import pyMRAW

@pytest.mark.filterwarnings('ignore')
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


@pytest.mark.filterwarnings('ignore')
def test_cihx():
    filename = './data/beam.cihx'
    cih = pyMRAW.get_cih(filename)
    N = cih['Total Frame']
    h = cih['Image Height']
    w = cih['Image Width']
    mraw = pyMRAW.load_images(filename[:-5] + '.mraw', h, w, N, bit=16, roll_axis=False)
    np.testing.assert_equal(mraw.shape, (4, 80, 1024))

@pytest.mark.filterwarnings('ignore')
def test_12bit_cihx():
    filename = './data/ball_12bit.cihx'
    cih = pyMRAW.get_cih(filename)
    N = cih['Total Frame']
    h = cih['Image Height']
    w = cih['Image Width']
    mraw = pyMRAW.load_images(filename[:-5] + '.mraw', h, w, N, bit=12, roll_axis=False)
    np.testing.assert_equal(mraw.shape, (15, 384, 384))
    np.testing.assert_equal(mraw.dtype, np.dtype(np.uint16))


@pytest.mark.filterwarnings('ignore')
def test_save_mraw():
    root_dir = './tests'
    with tempfile.TemporaryDirectory(dir=root_dir) as tmpdir:

            images8 = np.ones((3, 4, 5), dtype=np.uint8)
            images16 = np.ones((2, 3, 4), dtype=np.uint16) * (2**12-1)

            mraw8, cih8 = pyMRAW.save_mraw(images8, os.path.join(tmpdir, 'test8.cih'), bit_depth=8, ext='mraw', info_dict={'Record Rate(fps)':10})
            mraw8_16, cih8_16 = pyMRAW.save_mraw(images8, os.path.join(tmpdir, 'test8_16.cih'), bit_depth=16, ext='mraw', info_dict={'Shutter Speed(s)':0.001})
            mraw16, cih16 = pyMRAW.save_mraw(images16, os.path.join(tmpdir, 'test16.cih'), bit_depth=16, ext='mraw', info_dict={'Comment Text':'Test saving 16 bit images.'})

            loaded_images8, info8 =  pyMRAW.load_video(cih8)
            loaded_images8_16, info8_16 =  pyMRAW.load_video(cih8_16)
            loaded_images16, info16 =  pyMRAW.load_video(cih16)

            assert loaded_images8.shape == images8.shape
            assert loaded_images8_16.shape == images8.shape
            assert loaded_images16.shape == images16.shape
            assert loaded_images8.dtype == np.uint8
            assert loaded_images8_16.dtype == np.uint16
            assert loaded_images16.dtype == np.uint16
            assert info8['Record Rate(fps)'] == 10
            assert info8_16['Shutter Speed(s)'] == 0.001
            assert info16['Comment Text'] == 'Test saving 16 bit images.'

            loaded_images8._mmap.close()
            loaded_images8_16._mmap.close()
            loaded_images16._mmap.close()


if __name__ == '__main__':
    np.testing.run_module_suite()