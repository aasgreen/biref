import numpy as np
import sys
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pandas as pd
import glob as glob
import pims
import skimage
import colour as colour
from colour.plotting import *
from matplotlib.widgets import RectangleSelector
import numpy as np
import matplotlib.pyplot as plt
import re
import sys


def line_select_callback(eclick, erelease):
    'eclick and erelease are the press and release events'
    x1, y1 = eclick.xdata, eclick.ydata
    x2, y2 = erelease.xdata, erelease.ydata
    print("(%3.2f, %3.2f) --> (%3.2f, %3.2f)" % (x1, y1, x2, y2))
    print(" The button you used were: %s %s" % (eclick.button, erelease.button))


def toggle_selector(event):
    print(' Key pressed.')
    if event.key in ['Q', 'q'] and toggle_selector.RS.active:
        print(' RectangleSelector deactivated.')
        toggle_selector.RS.set_active(False)
    if event.key in ['A', 'a'] and not toggle_selector.RS.active:
        print(' RectangleSelector activated.')
        toggle_selector.RS.set_active(True)


#First, import michel-levy chart and get it in usuable form
plt.ion()
mL = mpimg.imread('./Picture1.png')
mL = skimage.img_as_ubyte(mL)
mL = mL[:,:,0:3]
#imgplot = plt.imshow(mL)
mLConvert = .05/mL.shape[1]

#Need to do own michel-levy chart
def mlEQ(lam,ret):
    '''lam: wavelength in nanometers
        ret: retardence in microns'''
    return np.sin(np.pi*ret/lam*10**3)**2.
lamL = np.arange(360,830,5)
retL = np.arange(.05,0.3,.0001)*5
I = np.array([[mlEQ(lam,ret) for lam in lamL] for ret in retL])
XYZ = [colour.spectral_to_XYZ(colour.SpectralPowerDistribution({lam:i for (lam,i) in zip(lamL,iL)}),illuminant=colour.ILLUMINANTS_RELATIVE_SPDS["D65"]) for iL in I]
ilA = colour.ILLUMINANTS['CIE 1931 2 Degree Standard Observer']['D65'] #don't really know what this is, but I think I'm using a halogen lamp
M = np.array([[2.04414, -.5649, -0.3447],[-0.9693 , 1.8760, 0.0416],[0.0134, -0.1184, 1.0154]])
#RGB = [list(colour.XYZ_to_RGB(xyz/100,ilA,ilA,M)) for xyz in XYZ]
sRGB = [list(colour.XYZ_to_sRGB(xyz/100)) for xyz in XYZ]
#psd = {lam:i for (lam,i) in zip(lamL,I)}
#psd = colour.SpectralPowerDistribution(psd)
#XYZ = colour.spectral_to_XYZ(psd)
#mlChart = np.array([[i for i in RGB] for j in np.arange(0,1000)])
mlChart2 = np.array([[i for i in sRGB] for j in np.arange(0,200)])
fig10,ax10 = plt.subplots()
ax10.axis('off')
ax10.imshow(mlChart2)
fig10.savefig('mlChart.png')
sys.exit()
#Now, we can calculate a pretty fair michel-levy chart. Now, set it up so that we can find the colour difference in XYZ between the average colour read in from the image


#Now, import voltage data so we can match it up
voltData = pd.read_csv('./DC Field PLM/10-11-2018_DCField_FileName_to_VoltsApplied.csv')


#Now, import images
print('begin import')


print('\nROI identification')
plt.ion()
fileNames = glob.glob('W586/')
ims = {}
#roi = {'w586':[
for name in fileNames:
    images = pims.ImageSequence(name+"/W*.jpg")
    fig,ax=plt.subplots()
    ax.imshow(images[0])

    print("\n      click  -->  release")

    # drawtype is 'box' or 'line' or 'none'
    toggle_selector.RS = RectangleSelector(ax, line_select_callback,
                                           drawtype='box', useblit=True,
                                           button=[1, 3],  # don't use middle button
                                           minspanx=5, minspany=5,
                                           spancoords='pixels',                                       interactive=True)
    plt.connect('key_press_event', toggle_selector)
    input("Choose ROI")
    roi = toggle_selector.RS.extents #xmin,xmax,ymin,ymax
    roi = [int(r) for r in roi]
    
    


    color = np.array([i[roi[2]:roi[3],roi[0]:roi[1]].mean((0,1))/255 for i in images])
    ##now, convert this into the XYZ domain    
    cXYZ = [colour.sRGB_to_XYZ(col) for col in color] 
    delE = np.array([[colour.delta_E(c1*100,c2) for c2 in XYZ] for c1 in cXYZ])
    bif = np.array([retL[i.argmin()]/5 for i in delE])

    #Now, extract the voltage from the data files

    #first, pull the appropriate columsn

    nameRegex = re.compile('[wW](\d{3}).*')
    compound = nameRegex.search(name)[1]

    cmpGex = re.compile('^W'+compound+'.*')

    col1 = [i for i,el in enumerate(voltData.columns) if cmpGex.search(el) ][0]
    col2 = col1+1
    bif = pd.Series(bif,dtype='float')
    imageNum = 'W'+compound+'-'+voltData.iloc[:,col1].astype('str')
    voltage = voltData.iloc[:,col2].replace('DC OFF', 0).astype('float')
    imageNum = imageNum[~voltage.isnull()]
    voltage = voltage[~voltage.isnull()]
    dataTemp = pd.concat([imageNum,voltage,bif],axis=1)
    dataTemp.columns=['Im. No.','Applied Field (V)','Del N']
    dataTemp.to_csv('W'+compound+'bVv-r2.csv',index=False)

    #now, assuming that the biref is still in the ordered sequence of images, we can simply match up the biref and the voltage info







