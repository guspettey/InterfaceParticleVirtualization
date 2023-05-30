from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import cv2
import skimage.filters as skf
from skimage.measure import label, regionprops_table, marching_cubes
from skimage.feature import match_template
import pandas as pd
from scipy.ndimage import  rotate, binary_erosion
import mcubes
import sys
from depthDetection import draw_box

def yesno(question):
    """Simple Yes/No Function."""
    prompt = f'{question} ? (y/n): '
    ans = input(prompt).strip().lower()
    if ans not in ['y', 'n','q']:
        print(f'{ans} is invalid, please try again...')
        return yesno(question)
    if ans == 'y':
        return True
    elif ans == 'n':
        return False
    print('Quitting')
    sys.exit(1)


def poll_repo_folder(RepoPath):
    """
    Checks the status of the image series in the particle repository and updates the .json file.
    Used to manage the current status of a series of images, and allows for discontinuous or distributed processing.
    inputs: 
        RepoPath: string of path to the particle repository for all image folders
    returns:
        ImageSeriesStatus: updated .json file
    outputs:
        ImageSeriesStatus: updated .json file (saved in place)
    """
    RepoPath = Path(RepoPath)
    if (RepoPath/'ImageSeriesStatus.json').is_file():
        ImageSeriesStatus = pd.read_json((RepoPath/'ImageSeriesStatus.json'),orient='index',convert_dates=False,convert_axes=False)
        p=RepoPath.glob('*/')
        ImageFolderList = [x.stem for x in p if x.is_dir()]
        statusList=ImageSeriesStatus.index.to_list()
        
        newFolderList = [x for x in ImageFolderList if x not in statusList]
        for folder in newFolderList:
            ImageSeriesStatus.append(pd.Series(name=folder))
            ImageSeriesStatus.loc[folder,ImageSeriesStatus.columns]=False

    else: #create new dataframe
        column_names = ['Stacked','Cropped','Binarised','Corr_Alignment','Under_Construction','Reconstructed','Meshed']
        p=RepoPath.glob('*/')
        ImageFolderList = [x.stem for x in p if x.is_dir()]
        ImageSeriesStatus = pd.DataFrame(index=ImageFolderList,columns=column_names)

    for ImgSeries in ImageSeriesStatus.index.values:
        SeriesFolder=(RepoPath/str(ImgSeries))
        if (SeriesFolder/'stacked').is_dir():
            ImageSeriesStatus.loc[ImgSeries,'Stacked']=True
        else:
            ImageSeriesStatus.loc[ImgSeries,ImageSeriesStatus.columns]=False
        if (SeriesFolder/'cropped').is_dir():
            ImageSeriesStatus.loc[ImgSeries,'Cropped']=True
        else:
            ImageSeriesStatus.loc[ImgSeries,ImageSeriesStatus.columns.difference(['Stacked'])]=False
            continue

        if (SeriesFolder/'cropped'/'image_properties.json').is_file():
            ImageSeriesStatus.loc[ImgSeries,'Binarised']=True
        else:
            ImageSeriesStatus.loc[ImgSeries,ImageSeriesStatus.columns.difference(['Stacked','Cropped'])]=False
            continue

        if (SeriesFolder/"rescaled_corr").is_dir(): #this needs manipulating if operating on rescanned grains for the rescaled folders
            ImageSeriesStatus.loc[ImgSeries,'Corr_Alignment']=True
        else:
            ImageSeriesStatus.loc[ImgSeries,ImageSeriesStatus.columns.difference(['Stacked','Cropped','Binarised'])]=False
            continue

        if ImageSeriesStatus.loc[ImgSeries,'Under_Construction']==True:
            ImageSeriesStatus.loc[ImgSeries,'Under_Construction']=True
            continue
        else:
            ImageSeriesStatus.loc[ImgSeries,'Under_Construction']=False


        if len(list(SeriesFolder.glob("*.npy")))>0:
            ImageSeriesStatus.loc[ImgSeries,'Reconstructed']=True
        else:
            ImageSeriesStatus.loc[ImgSeries,'Reconstructed']=False

        if len(list(SeriesFolder.glob("*.obj")))>0:
            ImageSeriesStatus.loc[ImgSeries,'Meshed']=True
        else:
            ImageSeriesStatus.loc[ImgSeries,'Meshed']=False

    ImageSeriesStatus.sort_index()
    ImageSeriesStatus.to_json((RepoPath/'ImageSeriesStatus.json'),orient='index',indent = 4)
   
    return ImageSeriesStatus

def onclick(event):
    x1,y1 = event.xdata, event.ydata
    print(x1,y1)


def overlay_stacked_crop_save(path):
    """
    Takes an image series and crops all to consistent new canvas.
    Crop bounds are graphically selected by overlaying all images with alpha.
    input:  
        path_str: path to folder as a string
    returns:
        None
    output:
        saves the cropped images to a child folder "cropped"
    """
        
    stacked_path = Path(path/'Stacked')
    globbed = stacked_path.glob('*.tif')
    imageList = [x for x in globbed if x.is_file()]

    x1, x2, y1, y2 = draw_box(imageList,alpha=0.1,title='Draw the cropping box')

    Path((path/"cropped")).mkdir(parents=True,exist_ok=True)
    pbar = tqdm((file for file in imageList),total=len(imageList),position=0, leave=True)
    for idx,file in enumerate(pbar):
        img = cv2.imread(str(file),0).astype('uint8')[y1:y2,x1:x2]
        cv2.imwrite(str(path/"cropped"/file.name), img)

    print("Cropped images saved.")

def binarise_otsu(pathName):
    """
    Takes a cropped image series and binarises using Otsu's method.
    input:
        path_str: path to folder as a string
    returns:
        None
    output:
        saves the binarised images to a child folder "cropped"
    """

    pathName = Path(str(pathName))
    croppedPath = pathName/"cropped"
    globbed = croppedPath.glob('*.tif')
    croppedList = [x for x in globbed if x.is_file()]
    image_name_list = [x.stem for x in globbed if x.is_file()]

    img = cv2.imread(str(croppedList[0]),0).astype('uint8')
    thresh_val = skf.threshold_otsu(img)

    properties = pd.DataFrame(columns=['left_idx','right_idx','top_idx','centroid_0','centroid_1','area'],index=image_name_list)
    
    pbar = tqdm(     (file for file in croppedList)  ,total=len(croppedList)   )
    for idx, file in enumerate( pbar  ):
        pbar.set_description("binarising and writing {} to child directory".format(file.name)) 
        img = cv2.imread(str(file),0).astype('uint8')

        _, binaryimg = cv2.threshold(img,thresh_val,255,cv2.THRESH_BINARY_INV)

        #get some debug info of the particle outline
        contours, _ = cv2.findContours(binaryimg, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        longest_list = max(contours,key=len) #find the longest contour
        points=np.column_stack(([longest_list[:,0,0],longest_list[:,0,1]])) #convert the list into a two col np array
        binaryimg[binaryimg==255]=1 #convert the binary image to 1s and 0s

        regprop = regionprops_table(binaryimg,properties=['centroid','area'])

        rows, cols = binaryimg.shape
        frame_centreline = int(cols/2)

        cent0 = float(regprop['centroid-0'][0])
        cent1 = float(regprop['centroid-1'][0])
        binarea = int(regprop['area'][0])
        proplist = list([min(points[:,0]),max(points[:,0]),min(points[:,1]),cent0,cent1,binarea])
        properties.loc[file.stem] = proplist

        cv2.imwrite(str(pathName/"cropped"/file.name), binaryimg)

    properties['x_offset'] = frame_centreline-properties['centroid_1'] # calc the bx value for transforming
    properties['y_offset'] = properties['top_idx']-properties['top_idx'].min()

    #save the properties to json for later use
    properties.to_json((pathName/'cropped'/'image_properties.json'),orient="index")


def rescale_to_tip(path_str):
    """
    Rescales the cropped and binarised image set to align the tips of the grains (assuming this point is fixed).
    The rescaled images are then realigned based on the x-centroid being constant, and cropped tightly.
    This is done to account for parallax in the stacked images where the particle moves into and away from the camera.
    inputs:
        path_str: path to folder as a string
    returns:
        None
    output: 
        saves the rescaled and realigned images to child folder "rescaled_corr"
    """
    pathName = Path(str(path_str))
    imageList = [x for x in Path((pathName/'cropped')).glob('*.tif') if x.is_file()]
    cropped_img = cv2.imread(str(imageList[0]),0)
    properties = pd.read_json((pathName/'cropped'/'image_properties.json'),orient="index",convert_dates=False,convert_axes=False)
    properties.sort_index(inplace=True)
    #create the scale factors required
    properties['rescale_factor'] = 1+properties['y_offset']/cropped_img.shape[0]
    properties['max_width']=properties['right_idx']-properties['left_idx']

    Path(pathName/'rescaled_corr').mkdir(parents=True,exist_ok=True)
    for image in properties.index.values:
        cropped_img = cv2.imread(str((pathName/'cropped'/(image+'.tif'))),0)
        rfactor = properties.loc[image,'rescale_factor']
        rescaled_img = np.flipud(cv2.resize(cropped_img,None,fx=rfactor,fy=rfactor))

        regprop = regionprops_table(cropped_img,properties=['centroid','area'])

        properties.loc[image,'centroid_0']=float(regprop['centroid-0'][0])
        properties.loc[image,'centroid_1']=float(regprop['centroid-1'][0])
        properties.loc[image,'area']=int(regprop['area'][0])

        contours, _ = cv2.findContours(rescaled_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        longest_list = max(contours,key=len) #find the longest contour
        points=np.column_stack(([longest_list[:,0,0],longest_list[:,0,1]])) #convert the list into a two col np array

        properties.loc[image,'top_idx']=max(points[:,1])
        properties.loc[image,'left_idx']=min(points[:,0])
        properties.loc[image,'right_idx']=max(points[:,0])
        
        cv2.imwrite(str(pathName/'rescaled_corr'/(image+'.tif')),rescaled_img)

    for image in properties.index.values:
        rescaled_img = cv2.imread(str((pathName/'rescaled_corr'/(image+'.tif'))),0)

        #move images using recomputed centroids after scaling
        rows,cols = rescaled_img.shape
        frame_centreline = int(cols/2)

        properties['x_offset'] = frame_centreline-properties['centroid_1'] # calc the bx value for transforming
        properties.loc[image,'x_offset'] = properties.loc[image,'centroid_1']-properties['centroid_1'].mean()
        properties.loc[image,'y_offset'] = properties.loc[image,'top_idx']-properties['top_idx'].mean()
        
        M = np.float32([[1,0,-properties.loc[image,'x_offset']],[0,1,-properties.loc[image,'y_offset']]])
        rescaled_img = cv2.warpAffine(rescaled_img,M,(cols,rows))

        contours, _ = cv2.findContours(rescaled_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        longest_list = max(contours,key=len) #find the longest contour
        points=np.column_stack(([longest_list[:,0,0],longest_list[:,0,1]])) #convert the list into a two col np array

        properties.loc[image,'top_idx']=max(points[:,1])
        properties.loc[image,'left_idx']=min(points[:,0])
        properties.loc[image,'right_idx']=max(points[:,0])

        cv2.imwrite(str(pathName/'rescaled_corr'/(image+'.tif')),rescaled_img)    

    rightidx = properties['right_idx'].max()+20
    leftidx = properties['left_idx'].min()-20
    topidx= properties['top_idx'].max()+20
    for image in properties.index.values:
        rescaled_img = cv2.imread(str((pathName/'rescaled_corr'/(image+'.tif'))),0)
        #recrop images using max width, flipud before writing to file
        output = rescaled_img[0:int(topidx),int(leftidx):int(rightidx)]
        cv2.imwrite(str(pathName/'rescaled_corr'/(image+'.tif')),np.flipud(output))    
    properties.to_json(str(pathName/'rescaled_corr'/'image_properties.json'),orient='index',indent = 4)

def autoRescale(seedFeature,image,search=[0.9,1.2],convergenceLim=0.0001,returnFactors=True,reverseMatch=False):
    """
    Simple binary search recursive function to find the best scale factor for a given seed feature using match_template.
    Images must be loaded as numpy arrays.
    inputs:
        seedFeature: the image to be matched
        image: the image to be matched against
        search: the initial search range
        convergenceLim: the limit for the difference between the two search values
        returnFactors: if True, returns the scale factor and the match location
        reverseMatch: if True, the the seedFeature is matched against the image
    returns:
        bestMatch: the best scale factor
        match: the location of the best match
        factors: the list of scale factors used in the search
    """

    if reverseMatch==False:
        factors=[]
        fitA = match_template(cv2.resize(image,None,fx=search[0],fy=search[0]),seedFeature)
        scoreA = fitA.max()
        fitB = match_template(cv2.resize(image,None,fx=search[1],fy=search[1]),seedFeature)
        scoreB = fitB.max()
    else:
        factors=[]
        fitA = match_template(seedFeature,cv2.resize(image,None,fx=search[0],fy=search[0]))
        scoreA = fitA.max()
        fitB = match_template(seedFeature,cv2.resize(image,None,fx=search[1],fy=search[1]))
        scoreB = fitB.max()

    bestMatch = search[0] if scoreA>scoreB else search[1]
    factors.append(bestMatch)
    print('Fit Scores: {}, {}. Current Best Scale Factor: {}'.format(scoreA,scoreB,bestMatch),flush=True,end='\r')
    if abs(scoreA-scoreB)<convergenceLim:
        match = np.unravel_index(fitA.argmax(), fitA.shape) if scoreA>scoreB else np.unravel_index(fitB.argmax(), fitB.shape)

        if returnFactors==True:
            return bestMatch, match, factors
        else:
            return bestMatch, match

    else:
        delta=(search[1]-search[0])/2
        newsearch = [bestMatch,bestMatch+delta] if scoreA>scoreB else [bestMatch,bestMatch-delta]
        if reverseMatch==False:
            return autoRescale(seedFeature,image,search=sorted(newsearch),reverseMatch=False)
        else:
            return autoRescale(seedFeature,image,search=sorted(newsearch),reverseMatch=True)

def realign_rescanned(path_str,crop_search_area=False):
    """
    Realigns the revirtualised images to the original images using template matching and autorescaling.
    The images are then cropped to the original size, keeping the canvas size consistent.
    A properties dataframe is created to store the centroid and top index of each image.
    The pre-test images are given a border and the post-test image autorescaled.
    A folder must exist in the parent directory called "RescannedGrains" which contains the rescanned image series.
    inputs:
        path_str: path to particle folder as a string
        crop_search_area: if True, the user is prompted to select the search area in the pre-test images
    returns:
        None
    output:
        saves the realigned rescanned images to child folder "rescaled_corr"
    """
    folderPath = Path(str(path_str))
    rescannedPath = folderPath.parent/'RescannedGrains'/folderPath.stem
    print('Rescaling {} post-test images'.format(folderPath.stem),flush=True,end='\r')
    Path((folderPath/'rescaled_corr')).mkdir(parents=True,exist_ok=True)

    preImgList = sorted([x for x in (folderPath/'rescaled_corr').glob('*.tif*') if x.is_file()])
    postImgList = sorted([x for x in (rescannedPath/'cropped_corr').glob('*.tif*') if x.is_file()])
    image_name_list = [x.stem for x in preImgList]
    
    preImgShape = cv2.imread(str(preImgList[0]),0).shape
    postImgShape = cv2.imread(str(postImgList[0]),0).shape
    
    if crop_search_area==True:
        x1, x2, y1, y2=draw_box(preImgList,title='Select the search feature in the pre images, crop out the tip',alpha=0.25)
    
    properties = pd.DataFrame(columns=['left_idx','right_idx','top_idx','centroid_0','centroid_1','x_offset'],index=image_name_list)

    scaleRecord = pd.DataFrame(columns=[x.stem[-3:] for x in postImgList])
    for idx,angle in enumerate(preImgList):
        print('Scaling on {}\n'.format(angle.name),flush=True,end='\r')

        preImg = plt.imread(folderPath/'rescaled_corr'/angle.name)
        postImg = plt.imread(str(rescannedPath/'cropped_corr'/angle.name))

        if crop_search_area==True:
            padding = 250
            searchPatch=cv2.copyMakeBorder(preImg[y1:y2,x1:x2],padding,padding,padding,padding,cv2.BORDER_CONSTANT,0)
        else:
            padding = 250
            searchPatch=cv2.copyMakeBorder(preImg,padding,padding,padding,padding,cv2.BORDER_CONSTANT,0)

        scaleFactor,match,factors= autoRescale(searchPatch,postImg,search=[1.0,1.22],convergenceLim=0.00001,reverseMatch=True)
        scaleRecord[angle.stem[-3:]]=factors

        rescaled = cv2.resize(postImg,None,fx=scaleFactor,fy=scaleFactor)
        rows,cols = rescaled.shape
        M = np.float32([[1,0,match[1]-padding],[0,1,match[0]-padding]])
        corrected = cv2.warpAffine(rescaled,M,(cols,rows))

        #get the properties dataframe and vertically realign the tips using the largest value
        output = np.zeros_like(preImg) 
        if cols<preImgShape[1] and rows>preImgShape[0]:
            output[:,:]=np.hstack((corrected[:preImgShape[0],:],np.zeros((preImgShape[0],preImgShape[1]-cols),dtype=np.uint8)))
        elif rows<preImgShape[0] and cols>preImgShape[1]:
            output[:,:]=np.vstack((corrected[:,:preImgShape[1]],np.zeros((preImgShape[0]-rows,preImgShape[1]),dtype=np.uint8)))
        elif cols<preImgShape[1] and rows<preImgShape[0]:
            output[:rows,:cols]=corrected
        else:
            output[:,:]=corrected[:preImgShape[0],:preImgShape[1]]

        regprop = regionprops_table(output,properties=['centroid'])

        properties.loc[angle.stem,'centroid_0']=float(regprop['centroid-0'][0])
        properties.loc[angle.stem,'centroid_1']=float(regprop['centroid-1'][0])

        contours, _ = cv2.findContours(output, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        longest_list = max(contours,key=len) #find the longest contour
        points=np.column_stack(([longest_list[:,0,0],longest_list[:,0,1]])) #convert the list into a two col np array

        properties.loc[angle.stem,'top_idx']=min(points[:,1])
        properties.loc[angle.stem,'left_idx']=min(points[:,0])
        properties.loc[angle.stem,'right_idx']=max(points[:,0])

        cv2.imwrite(str(rescannedPath/'rescaled_corr'/angle.name),output)
        print("\033c", end="")

    properties['y_offset'] = properties['top_idx']-properties['top_idx'].mean()
    properties['x_offset'] = properties['centroid_1']-properties['centroid_1'].mean()
    
    #realign the centroids to the mean x value
    for idx,angle in enumerate(preImgList):
        print('Realigning on {}'.format(angle.name),flush=True,end='\r')
        rescaled = cv2.imread(str(rescannedPath/'rescaled_corr'/angle.name),0)
        rows,cols = rescaled.shape
        M = np.float32([[1,0,-properties.loc[angle.stem,'x_offset']],[0,1,-properties.loc[angle.stem,'y_offset']]])
        corrected = cv2.warpAffine(rescaled,M,(cols,rows))
        cv2.imwrite(str(rescannedPath/'rescaled_corr'/angle.name),corrected)
        contours, _ = cv2.findContours(corrected, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        longest_list = max(contours,key=len) #find the longest contour
        points=np.column_stack(([longest_list[:,0,0],longest_list[:,0,1]])) #convert the list into a two col np array

        properties.loc[angle.stem,'top_idx']=min(points[:,1])
        properties.loc[angle.stem,'left_idx']=min(points[:,0])
        properties.loc[angle.stem,'right_idx']=max(points[:,0])
    

    scaleRecord.to_csv(str(rescannedPath/'rescaled_corr'/'scaleFactors.csv'))
    properties.to_json(str(rescannedPath/'rescaled_corr'/'image_properties.json'),orient='index',indent = 4)


def extrude_rotate(path_str, use_full_series=True):
    """
    Takes the binarised images and extrudes by greater dimension size
    3D arrays are rotated and multiplied to keep the intersection (multiplication of ones)
    finally binary erosion is conducted to return a binary shell
    inputs:
        path_str: path to root image folder as a string
        use_full_series: if False only half series will be used
    returns:
        None
    output:
        saves the filled and shell binary array to root image folder
    """

    pathName = Path(str(path_str))

    corr_pathName = (pathName/"rescaled_corr")
    globbed = corr_pathName.glob('*.tif')
    corrList = [x for x in globbed if x.is_file()]
    corrList.sort()

    #get a sorted list of the file endings
    idx_list =  [str(item).zfill(3) for item in list(np.arange(0,len(corrList)))]

    corr_img= cv2.imread(str(corrList[0]),0)
    corr_img=np.fliplr(np.flipud(corr_img))

    rows,cols = corr_img.shape 

    numStops = len(corrList)
    angleIncrement = 360/(numStops+1) #as the number of files is n-1
    stack_height = 1

    #create the array of ones
    combined = np.ones([rows+2*stack_height,cols,max(rows,cols)],dtype=np.uint8)

    if use_full_series==True:
        corrList=corrList
        filledArrayName = 'fullSeries_combined_arr_filled'
        shellArrayName = 'fullSeries_combined_arr_eroded'

    elif use_full_series==False:
        half_list_index = int(len(corrList)//2)
        corrList=corrList[0:half_list_index]
        filledArrayName = 'halfSeries_combined_arr_filled'
        shellArrayName = 'halfSeries_combined_arr_eroded'

    pbar = tqdm((file for file in corrList),total=len(corrList))
    for idx,file in enumerate(   pbar   ):

        pbar.set_description("extruding and rotating image {}".format(idx))
        
        img = cv2.imread(str(file),0).astype('uint8')
        img = np.fliplr(np.flipud(img))
        #using the last three chars in the filename, find the 
        rot_angle = int(file.stem[-3:].lstrip("0") or "0")*angleIncrement
        #need to add a layer of 0s to the top of the array (base of particle) to allow marching cubes to close mesh
        stack = np.zeros((stack_height,int(cols)))

        img = np.vstack((stack,img,stack)).astype('uint8')

        threed_img = np.repeat(img[:, :, np.newaxis],max(rows,cols),axis=2).astype('uint8')
        threed_img = rotate(threed_img,rot_angle,axes=(1,2),output='uint8', reshape=False, order=3, mode='constant', cval=0.0, prefilter=False)

        combined = (combined*threed_img).astype('uint8')

    reshaped = np.transpose(combined,(1,2,0)) # better orientation of final array for stl generation
    np.save(str( pathName /  filledArrayName),reshaped.astype('uint8'))

    # erosion = binary_erosion(reshaped).astype(reshaped.dtype)
    # shell = (reshaped != erosion).astype(reshaped.dtype)
    # np.save(str( pathName /  shellArrayName),shell.astype('uint8'))

def mesh_binary_array(path_str,step_size):
    """
    carries out marching cubes meshing on the filled binary array with a given step_size
    inputs:
        pathName: path to root image folder as a string 
        step_size: step for marching cubes
    returns:
        None
    outputs:
        .obj file saved in root image folder
    """
    pathName = Path(str(path_str))

    globbed = pathName.glob('*filled.npy')
    array_list = [x for x in globbed if x.is_file()]

    if str(array_list[0].name).startswith('full'):
        meshName = 'fullSeries_particleMesh.obj'
    elif str(array_list[0].name).startswith('half'): 
        meshName = 'halfSeries_particleMesh.obj'

    filledArray = np.load(array_list[0]).astype(np.uint8)
    verts, faces, _, _ = marching_cubes(filledArray.astype(np.uint8), level=0.5, spacing=(1,1,1), allow_degenerate=False, step_size=step_size,gradient_direction='ascent')

    mcubes.export_obj(verts, faces,str( pathName /  meshName))
