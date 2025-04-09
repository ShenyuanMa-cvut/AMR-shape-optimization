import sys
import pyvista
import numpy as np

def main(*args):
    filename = args[0]

    grid = pyvista.read(filename)

    p = pyvista.Plotter(off_screen=True)
    screenshot_name = "final"

    try:
        active_field = args[1]
        screenshot_name += "_"+active_field
        log_scale = int(args[2])
    except IndexError:
        p.add_mesh(grid, style='wireframe',color='k')
    else:
        grid.set_active_scalars(active_field)
        p.add_mesh(grid, cmap='gray', flip_scalars=True, log_scale=log_scale,show_edges=False)

    length = 2.    
    arrow_num = 10
    center1 = np.zeros((arrow_num,3))
    directions = np.zeros((arrow_num,3))
    directions[:,1] = -0.1
    center1[:,0] = np.linspace(length/6,length/3,arrow_num)

    center2 = np.zeros((arrow_num,3))
    center2[:,0] = np.linspace(2*length/3,5*length/6,arrow_num)

    p.add_arrows(center1,directions,color='red')
    p.add_arrows(center2,directions,color='blue')

    p.view_xy()
    #p.show()
    p.screenshot(f"{screenshot_name}.png")   


if __name__ == '__main__':
    main(*sys.argv[1:])