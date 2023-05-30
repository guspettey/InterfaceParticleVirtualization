import subprocess
from pathlib import Path
import Virtualization as virt
from datetime import datetime, timezone
import time
from typing import Union
import pandas as pd
from depthDetection import draw_box
from tqdm import tqdm
import cv2
import numpy as np

def overlayCropAllAngles(folderPath):
    """
    Used to crop all images in a folder to the same size, using the first image as a reference.
    The first image is displayed, and the user is asked to draw a box around the particle.
    The box is then applied to all images in the folder, and saved in a new subfolder named "cropped".
    Used to enhance the focus stacking outputs by removing irrelevant information.
    """
    folderPath = Path(folderPath)

    globbed = folderPath.rglob('*0*/*fs04.tif*')
    firstItemList = [x for x in globbed if x.is_file()]
    
    globbed = folderPath.rglob('*0*/*.tif*')
    allImageList = [x for x in globbed if x.is_file()]
    
    #get the cropping bounds of approximately the particle
    x1, x2, y1, y2 = draw_box(firstItemList,alpha=0.1,title='Draw the cropping box')

    Path((folderPath/"cropped")).mkdir(parents=True,exist_ok=True)
    pbar = tqdm((file for file in allImageList),total=len(allImageList),position=0, leave=True)
    for idx,file in enumerate(pbar):
        pbar.set_description("Cropping image {}".format(file.stem))
        # print(file.stem,flush=True,end='\r')
        img = cv2.imread(str(file),0).astype('uint8')[y1:y2,x1:x2]

        Path((file.parent/"cropped")).mkdir(parents=True,exist_ok=True)
        cv2.imwrite(str(file.parent/"cropped"/file.name), img)
    
    print("Cropped images saved.")

def main(heliconPath:Union[Path,str],particlesRepo:Union[Path,str]) -> None:
    """
    Used to run a recursive operation on a repository, containing subfolders of particle image sets.
    Each particle set contains N subfolders at different angles which contain images at different focal lengths.
    Each subfolder will have images focus-stacked by heliconFocus, returning a new subfolder named "Stacked" containing N images with names formatted as PARTICLECODE_ANGLEINDEX.tif
    
    New folders in the repository will be processed automatically, whilst any existing folders will ask for permission to overwrite. 

    Inputs:
        heliconPath: path to the heliconFocus executable, as a string or Path
        particlesRepo: path to the particle repository
    Returns:
        None
    Outputs:
        Image folder named "Stacked" at same level as angle subfolders
    """
    heliconPath=Path(heliconPath)
    particlesRepo=Path(particlesRepo)

    #get all folders in repo as a sorted list
    globbed = particlesRepo.glob("*/")
    folderList = [x for x in globbed if x.is_dir()]
    folderList.sort(reverse=True)

    for folder in folderList:
        print("\033c", end="")
        print('Operating on folder:{}'.format(folder.stem))
        stackedFolder = Path(folder/"Stacked")
        #check if the stacked folder exists, if not create one
        if stackedFolder.is_dir():
            globbed = stackedFolder.glob("*.tif*")
            stackedImgList = [x for x in globbed if x.is_file()]
            # if there are images already present, ask to overwrite
            if stackedImgList:
                #for info, get the last modified datetime of the stacked file
                ts_epoch=stackedImgList[0].stat().st_mtime
                strtime=datetime.fromtimestamp(ts_epoch,timezone.utc).strftime('%Y-%m-%d %H:%M:%S')
                
                print("Images present in directory: {} \nLast modified at {}.".format(folder.stem,strtime))
                
                #ask to overwrite, else continue to next image series
                ans = virt.yesno("Do you want to overwrite folder contents")
                # ans=True
                if ans:
                    [x.unlink() for x in stackedImgList if x.is_file()]
                else:
                    print("Leaving folder alone")
                    time.sleep(5)
                    continue      
        else:
            stackedFolder.mkdir(parents=True)
        
        overlayCropAllAngles(folder)
        #Find the folder paths to each angle containing images to be stacked
        globbed=folder.glob("*0*/")
        angleDirList = [x for x in globbed if x.is_dir()]
        angleDirList.sort()

        for angleFolder in angleDirList:
            savePath = str(Path(stackedFolder/(folder.stem+"_"+angleFolder.stem+".tif"))) # YYYYmmDDHHSS_0XX.tif
            croppedPath = Path(angleFolder/"cropped")
            # command = [str(heliconPath),"-silent",'"{}"'.format(str(angleFolder)),str("-save:"+'"{}"'.format(str(savePath))),"-rp:35","-sp:7","-mp:2"]#string formatting messes up helicon
            command = [str(heliconPath),"-silent",str(croppedPath),str("-save:"+str(savePath)),"-sp:7","-mp:2","-va:10","-ha:10","-ra:0","-ba:0","-ma:0","-im:3"]
        
            co = subprocess.run(command,capture_output=True)

            print(f"{'Stacking folder:':<30}{angleFolder.stem:<40}",sep=' ', end='\r', flush=True)


if __name__ == "__main__":
    heliconPath = Path("/Applications/HeliconFocus.app/Contents/MacOS/HeliconFocus")
    particlesRepo = Path("YOUR PATH TO PARTICLES REPOSITORY")
    main(heliconPath,particlesRepo)
