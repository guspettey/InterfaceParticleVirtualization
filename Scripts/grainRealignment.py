from pathlib import Path
import numpy as np
import cv2
from matplotlib.widgets import RectangleSelector
import matplotlib.pyplot as plt
from skimage.filters import threshold_otsu as otsu
import serial
from harvesters.core import Harvester
from skimage.feature import match_template
import pandas as pd
import time


def line_select_callback(eclick, erelease):
    'eclick and erelease are the press and release events'
    # global x1, y1, x2, y2
    x1, y1 = eclick.xdata, eclick.ydata
    x2, y2 = erelease.xdata, erelease.ydata
    # print("Rectangle Coordinates: (%3.2f, %3.2f) --> (%3.2f, %3.2f)" % (x1, y1, x2, y2))
    # print(" The button you used were: %s %s" % (eclick.button, erelease.button))


def toggle_selector(event):
    print(' Key pressed.')
    if event.key in ['Q', 'q'] and toggle_selector.RS.active:
        print(' RectangleSelector deactivated.')
        toggle_selector.RS.set_active(False)
    if event.key in ['A', 'a'] and not toggle_selector.RS.active:
        print(' RectangleSelector activated.')
        toggle_selector.RS.set_active(True)

def draw_box(image_or_series,alpha=1,**kwargs):
    fig, ax = plt.subplots()
    title= kwargs.get('title',None)
    aspect=kwargs.get('aspect','equal')
    if title:
        ax.set_title(title)
    
    if isinstance(image_or_series,list):
        ax.imshow( (cv2.imread(str(image_or_series[0]),0)), alpha=alpha)
        ax.imshow( (cv2.imread(str(image_or_series[-1]),0)), alpha=alpha)
    elif isinstance(image_or_series,Path):
        ax.imshow( (cv2.imread(str(image_or_series),0)), alpha=1)
    elif isinstance(image_or_series,np.ndarray):
        ax.imshow( image_or_series, alpha=1,aspect=aspect)
        # print('No valid image or image series given to plot')

    toggle_selector.RS = RectangleSelector(ax, line_select_callback,
                                        useblit=True,
                                        button=[1, 3],  # don't use middle button
                                        minspanx=5, minspany=5,
                                        spancoords='pixels',
                                        interactive=True)
    plt.connect('key_press_event', toggle_selector)
    #to draw the rectangle, advance to next stage by closing plot
    plt.show(block=True)
    rect_selection_coords = [round(num) for num in toggle_selector.RS.extents]
    x1, x2, y1, y2 = rect_selection_coords


    return x1, x2, y1, y2

def snapImage(acquirer):
    for i in range(4):
       with acquirer.fetch() as buffer:
                component = buffer.payload.components[-1]
    buffer=acquirer.fetch()
    component = buffer.payload.components[-1] #the buffer fills up, we want the latest entry, i.e. the last...
    img = component.data.reshape(component.height, component.width)
    buffer.queue()

    return img

def query(serialPort,string):
    command="#1"+string+"\r"
    serialPort.write(bytes(command, 'utf-8'))
    time.sleep(0.1)
    stri = serialPort.read_until(b"\r").decode('UTF-8')[1:]
    # print(stri)
    return stri


def detectAngle(stackedFolder,img,crop=False,binarise=False,rotate=False):
    if isinstance(img,(str,Path)):
        newImage = cv2.imread(str(img),0)
    elif isinstance(img,np.ndarray):
        newImage=img

    stackedFolder=Path(stackedFolder)
    imageList = [x for x in stackedFolder.glob('*.tif*') if x.is_file()]
    imageList.sort()

    if crop==True:
        X1, X2, Y1, Y2 = draw_box(newImage,title='Crop the feature to search for')
        newImage = newImage[Y1:Y2,X1:X2]
        # crop_bounds = [Y1,Y2,X1,X2]

        x1, x2, y1, y2 = draw_box(imageList,alpha=0.1,title='Crop the stacked image series')
        crop_bounds = [y1,y2,x1,x2]
        
    else:
        crop_bounds=None
    
    if binarise==True:
        thresh_val=otsu(newImage)
        _, newImage = cv2.threshold(newImage,thresh_val,255,cv2.THRESH_BINARY_INV)
    else:
        thresh_val=None

    if rotate==True:
        newImage=np.fliplr(np.flipud(newImage))

    comparison = {}

    for idx,oldImage in enumerate(imageList):
        print('comparing angle {} of {}'.format(idx,len(imageList)),flush=True,end='\r')
        oldimg = cv2.imread(str(oldImage),0)
        if crop==True:
            oldimg = oldimg[y1:y2,x1:x2]
        if binarise==True:
            _, oldimg = cv2.threshold(oldimg,thresh_val,255,cv2.THRESH_BINARY_INV)
        
        match = match_template(oldimg,newImage)
        maxcorr = match.max()

        # dh, _, _ = directed_hausdorff(img,newImage)
        comparison[str(oldImage)] = maxcorr

    # maxIndex = np.argmax(comparison)
    maxMatch = max(comparison, key=comparison.get)

    return Path(maxMatch), max(comparison.values()),thresh_val, crop_bounds

def improveAngle(bestImage,serial,acquirer,searchRange=12,crop_bounds=None,thresh_val=None):
    if isinstance(bestImage,(str,Path)) and crop_bounds and thresh_val:
        bestImage = cv2.imread(str(bestImage),0)
    elif isinstance(bestImage,np.ndarray):
        bestImage=bestImage
    else:
        print('No valid image passed')
        return

    x1, x2, y1, y2 = draw_box(bestImage,title='Crop the search image')
    bestImage=bestImage[Y1:Y2,X1:X2]
    if thresh_val:
        _, bestImage = cv2.threshold(bestImage,thresh_val,255,cv2.THRESH_BINARY_INV)


    #move back by searchrange/2 degrees
    query(serial,"s+"+str(int(12800*(searchRange/2)/360))) # step by total steps per rev/n
    query(serial,"d0") # set rotation to left
    query(serial,"A")

    #set to move by degree increments
    query(serial,"s+"+str(int(12800/360))) # step by total steps per rev/n
    query(serial,"d1") # set rotation to right

    comparison = np.zeros((1,searchRange))
    for n in range(searchRange):
        print('Analysing similarity at point {}'.format(n+1))
        img = snapImage(acquirer)
        if crop_bounds:
            Y1,Y2,X1,X2=crop_bounds
            img=img[Y1:Y2,X1:X2]
        if thresh_val:
            _, img = cv2.threshold(img,thresh_val,255,cv2.THRESH_BINARY_INV)
        
        match = match_template(bestImage,img)
        maxcorr = max(match)

        comparison[n] = maxcorr

        #advance right
        query(serial,"A")

    maxIndex = np.argmax(comparison)
    
    return maxIndex


def main():
    """
    This script is used to compare a new single image to a stack of images, and find the best match.
    This allows for the realignment of the grain to the same angle as the stack, prior to revirtualization.
    After realignment, the grain can be revirtualized using CaptureImages.
    """
    testCode = input('Enter the code for this particle')

    USBC_LOCATION=Path('/media/usb/')
    stackedPath = USBC_LOCATION/testCode/'Stacked'
    imageList = [x for x in stackedPath.glob('*.tif') if x.is_file()]

    print('opening gigE-vis platform')
    # open the GeniCam API
    h = Harvester()
    # import the GenTL cti file
    h.add_file('/opt/mvIMPACT_Acquire/lib/x86_64/mvGenTLProducer.cti')
    # update Harvester to use the GenTL file
    h.update()
    # display the list of detected GigE cameras
    print(h.device_info_list)

    ia = h.create(0)
    n=ia.remote_device.node_map

    n.acquisitionFrameRateControlMode.value = 'Programmable'
    n.ExposureMode.value = 'Timed'
    n.ExposureTime.value = 4004
    n.AcquisitionFrameRate.value = 4
    n.exposureAlignment.value = 'Synchronous'
    n.Width.value = 4112
    n.Height.value = 3008
    n.autoBrightnessMode.value = 'Off'
    n.devicePacketResendBufferSize.value = 0.1
    n.DeviceLinkThroughputLimitMode.value	= 'On'
    n.DeviceLinkThroughputLimit.value = 55000000 # I removed a 0 now it works
    n.GevSCPSPacketSize.value	= 9000
    n.GevStreamChannelSelector.value = 0

    ser = serial.Serial('/dev/ttyS4',baudrate=115200,
                         bytesize=serial.EIGHTBITS,
                         parity=serial.PARITY_NONE,
                         stopbits=serial.STOPBITS_ONE,
                         timeout=2000)

    #set motor properties
    print('opening serial connection,setting properties')
    query(ser,"J0") # open connection
    query(ser,"g32") # set 32nd step
    query(ser,"p1") # set relative positioning
    query(ser,":gn+200") # gear numerator
    query(ser,"o+200") # set speed HZ
    query(ser,":ramp_mode+2") # 
    query(ser,":accel+50800") # set acceleration HZ/S
    query(ser,":decel+50800") # set acceleration

    query(ser,"s+"+str(int(12800/360))) # step by total steps per rev/n

    query(ser,"A")
    print('Motor connected.')
    time.sleep(3)
    print(query(ser,"C"))
    query(ser,"I")

    ia.start(run_as_thread=True)
    print('Camera ready.')

    img=snapImage(ia)

    maxMatch, thresh_val, crop_bounds = detectAngle(stackedPath,img,crop=True,binarise=True)

    #return the amount of degrees to move by for the degree accurate max match
    maxIndex = improveAngle(maxMatch,ser,ia,crop_bounds=crop_bounds,thresh_val=thresh_val)

    #set retreat angle for best image. 12 is the searchRange
    query(ser,"s+"+str(int(12800*(12-maxIndex)/360))) # step by total steps per rev/n
    query(serial,"d0") # set rotation to left
    query(serial,"A") # advance to best fit

    baseAngle = int(maxMatch.stem[-3:])*(len(imageList)+1)/360
    query(ser,"s+"+str(int(12800*(baseAngle)/360))) # step by total steps per rev/n
    query(serial,"A") # advance to base image 


