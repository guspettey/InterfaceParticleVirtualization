from pathlib import Path
import sys
import pyvista as pv

try:
    pathName = Path(sys.argv[1]) #gets second argv in script call , i.e. python3 Plotting\Interactive - BasicView.py [filename]
    if pathName.is_file():
        print('Mesh {} exists, plotting'.format(pathName.parent))
    else:
        print('Mesh {} does not exist, exiting'.format(pathName.parent))
        sys.exit(1)
except:
    print('no path given in script call, exiting...')
    sys.exit(1)

mesh=pv.read(str(pathName))

smooth_w_taubin = mesh.smooth_taubin(n_iter=50, pass_band=0.05)


p=pv.Plotter(window_size=([720, 720]), off_screen=False)
#plot the smoothed mesh
smooth_w_taubin.plot_curvature(curv_type='mean',clim=[-0.025, 0.025])

#plot the original mesh
# p.add_mesh(mesh, cmap="jet",smooth_shading=False)

p.show(auto_close=False)