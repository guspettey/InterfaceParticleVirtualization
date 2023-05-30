from pathlib import Path
import numpy as np
import cv2
from virtualization.depthDetection import draw_box
from skimage.feature import match_template
from virtualization.Virtualization import autoRescale,yesno
from scipy.ndimage import zoom
from skimage.measure import marching_cubes
import mcubes
import pandas as pd
from pick import pick

statusPath = Path('/Users/gus/Library/CloudStorage/OneDrive-SharedLibraries-TheUniversityofNottingham/Research - Soil-Structure Interface - Particles/shearStatus.json')
def allCalibrated(path_to_status):
        status = pd.read_json(path_to_status,orient='index',convert_dates=False,convert_axes=False)
        return status.loc[status['rescanned']==True].index.to_list()

repoPath = Path('/Users/gus/Library/CloudStorage/OneDrive-SharedLibraries-TheUniversityofNottingham/Research - Soil-Structure Interface - Particles')

primeList = sorted(allCalibrated(statusPath))


if yesno('Do you want to use the full list?'):
    primeList = sorted(allCalibrated(statusPath))
else:
    option,index=pick(primeList,'Select a particle to load')
    primeList = [option]


for particle in primeList[:]:
    print('Loading particle {}...\n'.format(particle))
    array   = np.load(str(repoPath/particle/'fullSeries_combined_arr_filled.npy')).astype(np.uint8)

    flattenedPreAxis0 = np.sum(array,axis=0).astype(np.uint16).T
    flattenedPreAxis1 = np.sum(array,axis=1).astype(np.uint16).T

    array   = np.load(str(repoPath/'RescannedGrains'/particle/'fullSeries_combined_arr_filled.npy')).astype(np.uint8)

    flattenedPostAxis0 = np.sum(array,axis=0).astype(np.uint16).T
    flattenedPostAxis1 = np.sum(array,axis=1).astype(np.uint16).T

    x1, x2, y1, y2 = draw_box(flattenedPostAxis0,title='Select a template region to match to the post array',cmap='gray')
    templateAxis0 = flattenedPostAxis0[y1:y2,x1:x2]

    X1, X2, Y1, Y2 = draw_box(flattenedPostAxis1,title='Select a template region to match to the post array',cmap='gray')
    templateAxis1 = flattenedPostAxis1[y1:y2,X1:X2]

    factorAxis0,matchAxis0,_ = autoRescale(templateAxis0,cv2.copyMakeBorder(flattenedPreAxis0,250,250,250,250,cv2.BORDER_CONSTANT,value=0),reverseMatch=True,search=[0.9,1.15],convergenceLim=0.00001)
    factorAxis1,matchAxis1,_ = autoRescale(templateAxis1,cv2.copyMakeBorder(flattenedPreAxis1,250,250,250,250,cv2.BORDER_CONSTANT,value=0),reverseMatch=True,search=[0.9,1.15],convergenceLim=0.00001)

    finalFactor = np.mean([factorAxis0,factorAxis1])
    print('\nAxis 0 factor: {}, Axis 1 factor: {}\n'.format(factorAxis0,factorAxis1))

    if not yesno('Use a final factor of {}?'.format(finalFactor)):
        finalFactor = float(input('Enter a new final factor: '))


    fitAxis0 = match_template(flattenedPreAxis0,cv2.resize(templateAxis0,None,fx=finalFactor,fy=finalFactor))
    fitAxis1 = match_template(flattenedPreAxis1,cv2.resize(templateAxis1,None,fx=finalFactor,fy=finalFactor))

    matchAxis0 = np.unravel_index(fitAxis0.argmax(), fitAxis0.shape)
    matchAxis1 = np.unravel_index(fitAxis1.argmax(), fitAxis1.shape)

    offsetYAxis0 = y1-matchAxis0[0]
    offsetXAxis0  = x1-matchAxis0[1]

    offsetYAxis1 = Y1-matchAxis1[0]
    offsetXAxis1  = X1-matchAxis1[1]

    ypad = round(np.abs(np.mean([offsetYAxis0,offsetYAxis1])))
    print('Axis 0 fit score: {}, Axis 1 fit score: {}\nAxis 0 Y-offset: {}, Axis 1 Y-offset: {}'.format(np.max(fitAxis0),np.max(fitAxis1),offsetYAxis0,offsetYAxis1))

    if not yesno('Use a final Y-offset of {}?'.format(ypad)):
        ypad = int(input('Enter a new final Y-offset: '))

    print('Resizing by a factor of {}'.format(finalFactor))
    resizedPost = zoom(array,finalFactor,output=np.uint8,order=1)

    if offsetYAxis0 < 0:
        resizedPost = np.concatenate((np.zeros((resizedPost.shape[0],resizedPost.shape[1],ypad),dtype=np.uint8),resizedPost,np.zeros((resizedPost.shape[0],resizedPost.shape[1],5),dtype=np.uint8)),axis=2)
    else:
        resizedPost = resizedPost[:,:,ypad:]
        resizedPost = np.concatenate((np.zeros((resizedPost.shape[0],resizedPost.shape[1],1),dtype=np.uint8),resizedPost),axis=2)


    print('Meshing with marching cubes')
    verts, faces, _, _ = marching_cubes(resizedPost.astype(np.uint8), level=0.5, spacing=(1,1,1), allow_degenerate=False, step_size=5,gradient_direction='ascent')

    mcubes.export_obj(verts, faces,str( repoPath/'RescannedGrains'/particle/'translated_particleMesh.obj'))
    print('Mesh saved to {}'.format(str( (repoPath/'RescannedGrains'/particle/'translated_particleMesh.obj').parts[-3:])))