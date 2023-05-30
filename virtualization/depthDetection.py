from matplotlib.widgets import RectangleSelector
import numpy as np
import matplotlib.pyplot as plt
import cv2
from pathlib import Path
from skimage.filters import threshold_otsu as otsu
from skimage.feature import match_template
from tqdm import tqdm
from scipy.optimize import curve_fit,minimize_scalar


def line_select_callback(eclick, erelease):
    'eclick and erelease are the press and release events'
    # global x1, y1, x2, y2
    x1, y1 = eclick.xdata, eclick.ydata
    x2, y2 = erelease.xdata, erelease.ydata
    print("Rectangle Coordinates: (%3.2f, %3.2f) --> (%3.2f, %3.2f)" % (x1, y1, x2, y2))
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
    """
    Used to draw a box on an image or image series, and return the coordinates of the box.
    Inputs:
        image_or_series: either a Path object to an image, a list of Path objects to images, or a numpy array of an image
        alpha: transparency of the image, default 1
        kwargs: optional arguments, for title and aspect ratio
    Returns:
        x1, x2, y1, y2: coordinates of the box
    """
    fig, ax = plt.subplots()
    title= kwargs.get('title',None)
    aspect=kwargs.get('aspect','equal')
    if title:
        ax.set_title(title)
    
    if isinstance(image_or_series,list):
        ax.imshow( (cv2.imread(str(image_or_series[int(len(image_or_series)/4)]),0)), alpha=alpha)
        ax.imshow( (cv2.imread(str(image_or_series[int(2*len(image_or_series)/4)]),0)), alpha=alpha)
        ax.imshow( (cv2.imread(str(image_or_series[int(3*len(image_or_series)/4)]),0)), alpha=alpha)
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


def find_patch_offset(ImageA,ImageB,patch):
    """
    Used to find the offset of a search patch of an image between two images.
    Inputs:
        ImageA: numpy array of the first image
        ImageB: numpy array of the second image
        patch: numpy array of the patch to search for
    Returns:
        y_diff, x_diff: the y and x offset of the patch between the two images
    """
    matchA = match_template(ImageA,patch)
    matchB = match_template(ImageB,patch)

    maxA = np.unravel_index(matchA.argmax(), matchA.shape)
    maxB = np.unravel_index(matchB.argmax(), matchB.shape)

    y_diff = maxA[0]-maxB[0]
    x_diff = maxA[1]-maxB[1]
    
    return y_diff, x_diff

def find_patch_offset_subpixel_y(ImageA,ImageB,patch,fit_window=10):
    """
    Used to find the offset of a search patch of an image between two images.
    Subpixel accuracy in the y direction is achieved by fitting a parabola to the maxima of the cross correlation.
        Inputs:
        ImageA: numpy array of the first image
        ImageB: numpy array of the second image
        patch: numpy array of the patch to search for
        fit_window: the window size to use for the parabolic fit, default 10
    Returns:
        subpx_max_A-subpx_max_B: the y offset of the patch between the two images
    """
    matchA = match_template(ImageA,patch)
    matchB = match_template(ImageB,patch)

    #Use the standard max location as a lookup range for the interpolation, indexes must be maintained
    maxA = np.unravel_index(matchA.argmax(), matchA.shape)
    maxB = np.unravel_index(matchB.argmax(), matchB.shape)

    #take the y direction only
    Ay_corr = np.amax(matchA,axis=1)
    By_corr = np.amax(matchB,axis=1)

    A_bounds = [maxA[0]-fit_window,maxA[0]+fit_window]
    B_bounds = [maxB[0]-fit_window,maxB[0]+fit_window]
    
    def f(x, a, b, c):
        return a * x + b * x**2 + c

    A_popt,_ = curve_fit(f,np.arange(A_bounds[0],A_bounds[1]),Ay_corr[A_bounds[0]:A_bounds[1]])
    B_popt,_ = curve_fit(f,np.arange(B_bounds[0],B_bounds[1]),By_corr[B_bounds[0]:B_bounds[1]])

    A_fm = lambda x: -f(x, *A_popt)
    B_fm = lambda x: -f(x, *B_popt)

    A_r = minimize_scalar(A_fm, bounds=(A_bounds[0],A_bounds[1]))
    B_r = minimize_scalar(B_fm, bounds=(B_bounds[0],B_bounds[1]))

    subpx_max_A = A_r['x']
    subpx_max_B = B_r['x']

    return subpx_max_A-subpx_max_B


def img_track(list_of_files: list,leapfrog: int=5):
    """
    Used to track a patch of an image series, returning the offset of the patch between each image.
    Inputs:
        list_of_files: list of Path objects to the images
        leapfrog: the number of images to skip between tracking, default 5
    Returns:
        delta: numpy array of the offset of the patch between each image
    """
    #select the extents of the tracking, this is the crop bounds of the entire image series
    x1, x2, y1, y2=draw_box(list_of_files,title="Select the crop box for the image series")
    
    #get the threshold value from otsu
    baseImage = cv2.imread(str(list_of_files[0]),0)[y1:y2,x1:x2]
    thresh_val=otsu(baseImage)
    _, baseImage = cv2.threshold(baseImage,thresh_val,255,cv2.THRESH_BINARY_INV)

    #now select the patch to track in the cropped area
    X1, X2, Y1, Y2=draw_box(baseImage,title="Select the tracking patch")
    patch = baseImage[Y1:Y2,X1:X2]

    delta = np.zeros((len(list_of_files),2))
    res = np.zeros((len(list_of_files),2))

    pbar = tqdm(  (Q for Q in list_of_files[1:]),total=len(list_of_files[1:]) ,position=0, leave=True )
    for idx,image in enumerate(pbar):
        pbar.set_description('Tracking image {} of {}'.format(idx,len(list_of_files)))
        currentImage = cv2.imread(str(image),0)[y1:y2,x1:x2]
        _, currentImage = cv2.threshold(currentImage,thresh_val,255,cv2.THRESH_BINARY_INV)

        res[idx,:] = find_patch_offset(baseImage,currentImage,patch)
        delta[idx,:] = res[idx,:] - res[idx-1,:] 
        # delta[idx+1,:] -= delta[idx,:]

        if (idx % leapfrog == 0):
            baseImage = cv2.imread(str(image),0)[y1:y2,x1:x2]
            _, baseImage = cv2.threshold(baseImage,thresh_val,255,cv2.THRESH_BINARY_INV)
            res = np.zeros((len(list_of_files),2))
    
    plt.close()

    return delta

def img_track_subpixel_y(list_of_files: list,leapfrog: int=5):
    """
    Used to track a patch of an image series, returning the offset of the patch between each image,
    using the subpixel y tracking method.
    Inputs:
        list_of_files: list of Path objects to the images
        leapfrog: the number of images to skip between tracking, default 5
    Returns:
        delta: numpy array of the offset of the patch between each image
    """
    #select the extents of the tracking, this is the crop bounds of the entire image series
    x1, x2, y1, y2=draw_box(list_of_files,title="Select the crop box for the image series")
    
    #get the threshold value from otsu
    baseImage = cv2.imread(str(list_of_files[0]),0)[y1:y2,x1:x2]
    thresh_val=otsu(baseImage)
    _, baseImage = cv2.threshold(baseImage,thresh_val,255,cv2.THRESH_BINARY_INV)

    #now select the patch to track in the cropped area
    X1, X2, Y1, Y2=draw_box(baseImage,title="Select the tracking patch")
    patch = baseImage[Y1:Y2,X1:X2]

    delta = np.zeros((len(list_of_files),1))
    res = np.zeros((len(list_of_files),1))

    pbar = tqdm(  (Q for Q in list_of_files[1:]),total=len(list_of_files[1:]) ,position=0, leave=True )
    for idx,image in enumerate(pbar):
        pbar.set_description('Tracking image {} of {}'.format(idx,len(list_of_files)))
        currentImage = cv2.imread(str(image),0)[y1:y2,x1:x2]
        _, currentImage = cv2.threshold(currentImage,thresh_val,255,cv2.THRESH_BINARY_INV)

        res[idx] = find_patch_offset_subpixel_y(baseImage,currentImage,patch)
        delta[idx] = res[idx] - res[idx-1] 

        if (idx % leapfrog == 0):
            baseImage = cv2.imread(str(image),0)[y1:y2,x1:x2]
            _, baseImage = cv2.threshold(baseImage,thresh_val,255,cv2.THRESH_BINARY_INV)
            res = np.zeros((len(list_of_files),1))
    
    plt.close()

    return delta

def findShearAngle(particleFolder: Path, seedImage: np.ndarray or Path):
    """
    Find the shear angle of a particle from a seed image.
    Inputs:
        particleFolder: the path to the folder of the particle
        seedImage: the seed image to use
    Returns:
        angle: the shear angle equal to angle of the seed image - 90 degrees
    """
    if isinstance(seedImage,Path):
        seedImage = cv2.imread(str(seedImage),0)
    
    #get the list of stacked images
    imageList = sorted([x for x in (particleFolder/'Stacked').glob('*.tif') if x.is_file()])
    
    #crop the seed image to an in focus area with draw_box
    x1, x2, y1, y2=draw_box(seedImage,title="Select the crop box for the seed image")
    seedImage = np.flipud(np.fliplr(seedImage[y1:y2,x1:x2]))


    maxCorr = np.zeros((len(imageList),2))
    pbar = tqdm(  (Q for Q in imageList),total=len(imageList) ,position=0, leave=True )
    for idx,imagePath in enumerate(pbar):
        image = cv2.imread(str(imagePath),0)
        output = match_template(image,seedImage)
        maxCorr[idx,:] = [idx,output.max()]

    bestMatch = np.argmax(maxCorr[:,1])
    seed_angle = (bestMatch/(len(imageList)+1))*360
    print('Seed angle is {} degrees'.format(seed_angle))

    return seed_angle-90, maxCorr
        