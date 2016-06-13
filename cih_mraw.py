__author__ = 'Jaka'
import numpy as np
from tkinter import Tk
from tkinter.filedialog import askopenfilename

def open_CIH_file():
    """
    open the .cih file and the function reads the acqusition parameters
    Outputs:
        fps,shutter_speed,N,sir,vis,filename,FileFormat,bits,EfBitSide
    """
    okno=Tk()#open window
    filename=askopenfilename(parent=okno,title='Select the .cih file',filetypes=[("Photron cih file","*.cih")])#open window to load the camera and files info
    okno.destroy()#close the tk window
    cih=open(filename)
    for line in cih:
        if line[:16]=='Record Rate(fps)':
            fps=int(line[19:])
        elif line[:16]=='Shutter Speed(s)':
            shutter_speed=float(int(line[19])/int(line[21:]))
        elif line[:20]=='Original Total Frame':
            N_total=int(line[23:])
        elif line[:11]=='Total Frame':
            N=int(line[13:])
        elif line[:11]=='Image Width':
            w=int(line[13:])
        elif line[:12]=='Image Height':
            h=int(line[14:])
        elif line=='File Format : MRaw\n':
            FileFormat='.mraw'
        elif line=='File Format : TIFF\n':
            FileFormat='.tif'
        elif line[:11]=='File Format':
            raise Exception('unexpected File Format. Read: '+line)
        elif (line[:9]=='Color Bit'):
            bits=int(line[12:])
            if (bits<12): print('not 12bit! clipped values?')
            elif (bits>12): print('not 12bit but 16! values spaced - values may/will be divided by /16->12bit (during operation)')# - may cause overflow')
                #12-bit values are spaced over the 16bit resolution - in case of photron filming at 12bit
                #this can be meanded by dividing images with //16
        elif (line[:18]=='EffectiveBit Depth')&(line[21:]!='12\n'):
            print('not 12bit!')
        elif (line=='EffectiveBit Side : Lower\n'):
            EfBitSide='l'
        elif (line=='EffectiveBit Side : Higher\n'):
            EfBitSide='h'
        elif (line[:17]=='EffectiveBit Side'):
            raise Exception('unexpected EffectiveBit Side. Read: '+line)
    if (FileFormat=='.mraw')&(bits!=16):
        raise Exception('not a 16bit file!. load .mraw only works for 16bit files')#sicer zelooooo počas
    if N_total!=N:
        print('Clipped footage!')
    return fps,shutter_speed,N,w,h,filename[:-4],FileFormat,bits,EfBitSide

def load_images(mraw,h,w,N):
    """
    loads the next N images from the binary mraw file into a numpy array.
    Inputs:
        mraw - an opened binary .mraw file
        h - image height
        w - image width
        N - number of sequential images to be loaded
    Outputs:
        images[h, w, N]
    """
    images=np.fromfile(mraw,dtype=np.uint16,count=h*w*N).reshape(N,h,w)
    return np.rollaxis(images,0,3)

if __name__ == '__main__':
    from matplotlib import pyplot as plt
    from matplotlib import animation

    _,_,N,w,h,filename,_,_,_=open_CIH_file()
    if N>12:
        N=12
    mraw=open(filename+'.mraw','rb')
    mraw.seek(0,0)#find the beginning of the file
    image_data=load_images(mraw,h,w,N)#load N images
    mraw.close()

    fig = plt.figure()
    ax=plt.subplot()
    ms=ax.matshow(image_data[:,:,0],cmap=plt.get_cmap('gray'),vmin=0,vmax=2**12)#display data for first image
    def animate(i):
        ms.set_data(image_data[:,:,i])
        return [ms]
    anim = animation.FuncAnimation(fig, animate,frames=N, interval=1, blit=True)
    plt.show()

