# InterfaceParticleVirtualization
A pythonic framework to image and process particles as small as 1mm for reconstruction as a meshed obj file. Abraded grains can be revirtualized for direct comparison. 

## Dependencies
For collecting images I used a GigE camera connected to an Ubuntu OS machine, utilising the Harvesters python package to interface with the camera. 
Alternatively, use another image collection method.
Harvesters is Linux-only, although can be run through WSL on windows.
At present this package does not run on python 3.10.x so for this reason python 3.9.x is required.
All other packages work with python 3.9.x so a pyenv using this is recommended.

The standard data analysis stack is used extensively, with a few other helper packages, namely tqdm (for progress bars), PyMCubes (for meshing) and cv2 (for image wrangling).
For mesh visualisation, pyVista is recommended.

See `requirements.txt` for more, this can be used by pip for installation also.

## Folder Structure
For consistent methodology a folder structure must be followed, for easy management of the various files created. 
The structure is as follows:
```
.
└── ParticleRepository/
    ├── ImageSeriesStatus.json
    ├── Particle/
    │   ├── 000/
    │   ├── 001/
    │   │   ├── Particle_001_fs00.tif
    │   │   ├── Particle_001_fs01.tif
    │   │   ├── Particle_001_fs02.tif
    │   │   ├── ...
    │   │   └── Particle_001_fsXX.tif
    │   ├── ...
    │   ├── ...
    │   ├── 0XX/
    │   ├── Stacked/
    │   │   ├── Particle_001.tif
    │   │   ├── ...
    │   │   └── Particle_0XX.tif
    │   ├── cropped/
    │   │   ├── Particle_001.tif
    │   │   ├── ...
    │   │   └── Particle_0XX.tif
    │   ├── rescaled_corr/
    │   │   ├── Particle_001.tif
    │   │   ├── ...
    │   │   └── Particle_0XX.tif
    │   └── fullSeries_particleMesh.obj
    └── RescannedGrains/
        └── Particle/
            ├── 000/
            ├── 001/
            │   ├── Particle_001_fs00.tif
            │   ├── Particle_001_fs01.tif
            │   ├── Particle_001_fs02.tif
            │   ├── ...
            │   └── Particle_001_fsXX.tif
            ├── ...
            ├── ...
            ├── 0XX/
            ├── Stacked/
            │   ├── Particle_001.tif
            │   ├── ...
            │   └── Particle_0XX.tif
            ├── cropped/
            │   ├── Particle_001.tif
            │   ├── ...
            │   └── Particle_0XX.tif
            ├── rescaled_corr/
            │   ├── Particle_001.tif
            │   ├── ...
            │   └── Particle_0XX.tif
            ├── fullSeries_particleMesh.obj
            └── translated_particleMesh.obj
```
The particle repository can be stored anywhere and contains the particles, which should be named by the `CaptureImages.py` script as a timestamp when the images were created.
A folder at the same level named `RescannedGrains` should store the rescanned version of the same particle, allowing for consistent name convention to be used.

## Sample Usage
Initially, collect images using the `CaptureImages.py` script, changing the serial connection and camera information to suit (Harvesters requires an appropriate GenTL producer).

Image focus stacking can be achieved in any application, for ease of use, and to take full advantage of command-line interface, Helicon Focus is used. 

The `batchFocusStacking.py` script manages stacking of each angle folder in a particle directory, producing a sub-folder `Stacked` containing an image for each rotation point.

If `batchFocusStacking.py` is not used, images need to be cropped to just the region containing the particle, using `Virtualization.overlay_stacked_crop_save`.

The images can then be binarised using Otsu thresholding with `Virtualization.binarise_otsu`.

To account for parallax in the stacked images where the particle moves into and away from the camera, use `Virtualization.rescale_to_tip` to align the tips of the grains and centroids.

Finally, use `Virtualization.extrude_rotate` to create the reconstructed volume as a 3D binary array, and `Virtualization.mesh_binary_array` to create a `.obj` mesh.

Each of these functions simply uses a path to the particle folder as a string, allowing for multiple operations to be chained together with ease.

To realign the meshes for direct comparison, use the `meshRelignment.py` script to create a `translated_particleMesh.obj` mesh in the RescannedGrains version of the particle.
