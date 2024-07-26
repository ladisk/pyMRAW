# www.ladisk.si
#
# pyMRAW is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3 of the License.
#
# pyMRAW is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with pyMRAW.  If not, see <http://www.gnu.org/licenses/>.
"""
This module is reads Photron MRAW image sequences.

Author: Jaka Javh (jaka.javh@fs.uni-lj.si), Janko Slavič (janko.slavic@fs.uni-lj.si) www.ladisk.si

We developed this module while working on this publication:
J. Javh, J. Slavič and M. Boltežar: The Subpixel Resolution of Optical-Flow-Based Modal Analysis,
Mechanical Systems and Signal Processing, Vol. 88, p. 89–99, 2017
 
If you find it useful, consider to cite us.
"""

import os
from os import path
import numpy as np
import numba as nb
import warnings
import xmltodict

__version__ = '0.32'

SUPPORTED_FILE_FORMATS = ['mraw', 'tiff']
SUPPORTED_EFFECTIVE_BIT_SIDE = ['lower', 'higher']


def get_cih(filename):
    name, ext = path.splitext(filename)
    if ext == '.cih':
        cih = dict()
        # read the cif header
        with open(filename, 'r') as f:
            for line in f:
                if line == '\n': #end of cif header
                    break
                line_sp = line.replace('\n', '').split(' : ')
                if len(line_sp) == 2:
                    key, value = line_sp
                    try:
                        if '.' in value:
                            value = float(value)
                        else:
                            value = int(value)
                        cih[key] = value
                    except:
                        cih[key] = value

    elif ext == '.cihx':
        with open(filename, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
            first_last_line = [ i for i in range(len(lines)) if '<cih>' in lines[i] or '</cih>' in lines[i] ]
            xml = ''.join(lines[first_last_line[0]:first_last_line[-1]+1])

        raw_cih_dict = xmltodict.parse(xml)
        cih = {
            'Date': raw_cih_dict['cih']['fileInfo']['date'], 
            'Camera Type': raw_cih_dict['cih']['deviceInfo']['deviceName'],
            'Record Rate(fps)': float(raw_cih_dict['cih']['recordInfo']['recordRate']),
            'Shutter Speed(s)': float(raw_cih_dict['cih']['recordInfo']['shutterSpeed']),
            'Total Frame': int(raw_cih_dict['cih']['frameInfo']['totalFrame']),
            'Original Total Frame': int(raw_cih_dict['cih']['frameInfo']['recordedFrame']),
            'Image Width': int(raw_cih_dict['cih']['imageDataInfo']['resolution']['width']),
            'Image Height': int(raw_cih_dict['cih']['imageDataInfo']['resolution']['height']),
            'File Format': raw_cih_dict['cih']['imageFileInfo']['fileFormat'],
            'EffectiveBit Depth': int(raw_cih_dict['cih']['imageDataInfo']['effectiveBit']['depth']),
            'EffectiveBit Side': raw_cih_dict['cih']['imageDataInfo']['effectiveBit']['side'],
            'Color Bit': int(raw_cih_dict['cih']['imageDataInfo']['colorInfo']['bit']),
            'Comment Text': raw_cih_dict['cih']['basicInfo'].get('comment', ''),
        }

    else:
        raise Exception('Unsupported configuration file ({:s})!'.format(ext))

    # check exceptions
    ff = cih['File Format']
    if ff.lower() not in SUPPORTED_FILE_FORMATS:
        raise Exception('Unexpected File Format: {:g}.'.format(ff))
    bits = cih['Color Bit']
    if bits < 12:
        warnings.warn('Not 12bit ({:g} bits)! clipped values?'.format(bits))
                # - may cause overflow')
                # 12-bit values are spaced over the 16bit resolution - in case of photron filming at 12bit
                # this can be meanded by dividing images with //16
    if cih['EffectiveBit Depth'] != 12:
        warnings.warn('Not 12bit image!')
    ebs = cih['EffectiveBit Side']
    if ebs.lower() not in SUPPORTED_EFFECTIVE_BIT_SIDE:
        raise Exception('Unexpected EffectiveBit Side: {:g}'.format(ebs))
    if (cih['File Format'].lower() == 'mraw') & (cih['Color Bit'] not in [8, 12, 16]):
        raise Exception('pyMRAW only works for 8-bit, 12-bit and 16-bit files!')
    if cih['Original Total Frame'] > cih['Total Frame']:
        warnings.warn('Clipped footage! (Total frame: {}, Original total frame: {})'.format(cih['Total Frame'], cih['Original Total Frame'] ))

    return cih


def load_images(mraw, h, w, N, bit=16, roll_axis=True):
    """
    loads the next N images from the binary mraw file into a numpy array.
    Inputs:
        mraw: an opened binary .mraw file
        h: image height
        w: image width
        N: number of sequential images to be loaded
        roll_axis (bool): whether to roll the first axis of the output 
            to the back or not. Defaults to True
    Outputs:
        images: array of shape (h, w, N) if `roll_axis` is True, or (N, h, w) otherwise.
    """

    if int(bit) == 16:
        images = np.memmap(mraw, dtype=np.uint16, mode='r', shape=(N, h, w))
    elif int(bit) == 8:
        images = np.memmap(mraw, dtype=np.uint8, mode='r', shape=(N, h, w))
    elif int(bit) == 12:
        warnings.warn("12bit images will be loaded into memory!")
        #images = _read_uint12_video(mraw, (N, h, w))
        images = _read_uint12_video_prec(mraw, (N, h, w))
    else:
        raise Exception(f"Unsupported bit depth: {bit}")


    #images=np.fromfile(mraw, dtype=np.uint16, count=h * w * N).reshape(N, h, w) # about a 1/3 slower than memmap when loading to RAM. Also memmap doesn't need to read to RAM but can read from disc when needed.
    if roll_axis:
        return np.rollaxis(images, 0, 3)
    else:
        return images


def load_video(cih_file):
    """
    Loads and returns images and a cih info dict.
    
    Inputs:
    cih_filename: path to .cih or .cihx file, with a .mraw file 
        with the same name in the same folder.
    Outputs:
        images: image data array of shape (N, h, w)
        cih: cih info dict.

    """
    cih = get_cih(cih_file)
    mraw_file = path.splitext(cih_file)[0] + '.mraw'
    N = cih['Total Frame']
    h = cih['Image Height']
    w = cih['Image Width']
    bit = cih['Color Bit']
    images = load_images(mraw_file, h, w, N, bit, roll_axis=False)
    return images, cih


def save_mraw(images, save_path, bit_depth=16, ext='mraw', info_dict={}):
    """
    Saves given sequence of images into .mraw file.

    Inputs:
    sequence : array_like of shape (n, h, w), sequence of `n` grayscale images
        of shape (h, w) to save.
    save_path : str, path to saved cih file. 
    bit_depth: int, bit depth of the image data. Currently supported bit depths are 8 and 16.
    ext : str, generated file extension ('mraw' or 'npy'). If set to 'mraw', it can be viewed in
        PFV. Defaults to '.mraw'.
    info_dict : dict, mraw video information to go into the .cih file. The info keys have to match
        .cih properties descriptions exactly (example common keys: 'Record Rate(fps)', 
        'Shutter Speed(s)', 'Comment Text' etc.).

    Outputs:
    mraw_path : str, path to output or .mraw (or .npy) file.
    cih_path : str, path to generated .cih file
    """

    filename, extension = path.splitext(save_path)
    mraw_path = '{:s}.{:s}'.format(filename, ext)
    cih_path = '{:s}.{:s}'.format(filename, '.cih')

    directory_path = path.split(save_path)[0]
    if not path.exists(directory_path):
        os.makedirs(directory_path)

    bit_depth_dtype_map = {
        8: np.uint8,
        16: np.uint16
    }
    if bit_depth not in bit_depth_dtype_map.keys():
        raise ValueError('Currently supported bit depths are 8 and 16.')
    
    if bit_depth < 16:
        effective_bit = bit_depth
    else:
        effective_bit = 12
    if np.max(images) > 2**bit_depth-1:
        raise ValueError(
            'The input image data does not match the selected bit depth. ' +
            'Consider normalizing the image data before saving.')

    # Generate .mraw file
    with open(mraw_path, 'wb') as file:
        for image in images:
            image = image.astype(bit_depth_dtype_map[bit_depth])
            image.tofile(file)
    file_shape = (int(len(images)), image.shape[0], image.shape[1])
    file_format = 'MRaw'

    image_info = {'Record Rate(fps)': '{:d}'.format(1),
                'Shutter Speed(s)': '{:.6f}'.format(1),
                'Total Frame': '{:d}'.format(file_shape[0]),
                'Original Total Frame': '{:d}'.format(file_shape[0]),
                'Start Frame': '{:d}'.format(0),
                'Image Width': '{:d}'.format(file_shape[2]),
                'Image Height': '{:d}'.format(file_shape[1]),
                'Color Type': 'Mono', 
                'Color Bit': bit_depth,
                'File Format' : file_format,
                'EffectiveBit Depth': effective_bit,
                'Comment Text': 'Generated sequence. Modify measurement info in created .cih file if necessary.',
                'EffectiveBit Side': 'Lower'}

    image_info.update(info_dict)

    cih_path = '{:s}.{:s}'.format(filename, 'cih')
    with open(cih_path, 'w') as file:
        file.write('#Camera Information Header\n')
        for key in image_info.keys():
            file.write('{:s} : {:s}\n'.format(key, str(image_info[key])))
    
    return mraw_path, cih_path

def _read_uint12_video(data, shape):
    """Utility function to read 12bit packed mraw files into uint16 array
    Will store entire array in memory!

    Adapted from https://stackoverflow.com/a/51967333/9173710
    """
    data = np.memmap(data,  dtype=np.uint8, mode="r")
    fst_uint8, mid_uint8, lst_uint8 = np.reshape(data, (data.shape[0] // 3, 3)).astype(np.uint16).T
    fst_uint12 = (fst_uint8 << 4) + (mid_uint8 >> 4)
    snd_uint12 = ((mid_uint8 % 16) << 8) + lst_uint8
    return np.reshape(np.concatenate((fst_uint12[:, None], snd_uint12[:, None]), axis=1), shape)

def _read_uint12_video_prec(data, shape):
    """Utility function to read 12bit packed mraw files into uint16 array
    Will store entire array in memory!

    Adapted from https://stackoverflow.com/a/51967333/9173710
    """
    data = np.memmap(data,  dtype=np.uint8, mode="r")
    return nb_read_uint12(data).reshape(shape)


@nb.njit(nb.uint16[::1](nb.types.Array(nb.types.uint8, 1, 'C', readonly=True)), fastmath=True, parallel=True, cache=True)
def nb_read_uint12(data_chunk):
  """ precompiled function to efficiently covnert from 12bit packed video to 16bit video  
  it splits 3 bytes into two 16 bit words  
  data_chunk is a contigous 1D array of uint8 data, e.g. the 12bit video loaded as 8bit array
  """
  
  #ensure that the data_chunk has the right length
  assert np.mod(data_chunk.shape[0],3)==0
  out = np.empty(data_chunk.size//3*2, dtype=np.uint16)

  for i in nb.prange(data_chunk.shape[0]//3):
    fst_uint8=np.uint16(data_chunk[i*3])
    mid_uint8=np.uint16(data_chunk[i*3+1])
    lst_uint8=np.uint16(data_chunk[i*3+2])
    
    out[i*2] =   (fst_uint8 << 4) + (mid_uint8 >> 4)
    out[i*2+1] = ((mid_uint8 % 16) << 8) + lst_uint8
    
  return out

def show_UI():
    from tkinter import Tk
    from tkinter.filedialog import askopenfilename
    from matplotlib import pyplot as plt
    from matplotlib import animation


    window = Tk()  # open window
    filename = askopenfilename(parent=window, title='Select the .cih file', filetypes=[
        ("Photron cih file", "*.cih"), ("Photron cihx file", "*.cihx")])  # open window to load the camera and files info
    window.destroy()  # close the tk window

    cih = get_cih(filename)
    N = cih['Total Frame']
    h = cih['Image Height']
    w = cih['Image Width']

    #if N > 12:
    #    N = 12
    name, ext = path.splitext(filename)
    mraw = open(name + '.mraw', 'rb')
    mraw.seek(0, 0)  # find the beginning of the file
    image_data = load_images(mraw, h, w, N)  # load N images
    #np.memmap in load_images loads enables reading an array from disc as if from RAM. If you want all the images to load on RAM imediatly use load_images(mraw, h, w, N).copy()
    mraw.close()

    fig = plt.figure()
    ax = plt.subplot()
    ms = ax.matshow(image_data[:, :, 0], cmap=plt.get_cmap('gray'), vmin=0,
                    vmax=2 ** 12)  # display data for first image


    def animate(i):
        ms.set_data(image_data[:, :, i])
        return [ms]


    anim = animation.FuncAnimation(fig, animate, frames=N, interval=1, blit=True)
    plt.show()


if __name__ == '__main__':
    show_UI()
    #a = get_cih('data/sample_60k_16bit.cih')
    #print(a)