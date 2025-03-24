import sys
import pyvista

def main(*args):
    filename = args[0]

    grid = pyvista.read(filename)

    p = pyvista.Plotter()
    
    try:
        active_field = args[1]
    except IndexError:
        p.add_mesh(grid, style='wireframe',color='k')
    else:
        grid.set_active_scalars(active_field)
        p.add_mesh(grid, cmap='gray', flip_scalars=True)
        
    
    p.view_xy()
    p.show()


if __name__ == '__main__':
    main(*sys.argv[1:])