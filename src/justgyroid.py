# Install dependencies
!pip install trimesh numpy-stl ipywidgets scikit-image --quiet

import trimesh
import numpy as np
import matplotlib.pyplot as plt
from stl import mesh
from scipy.interpolate import RegularGridInterpolator
from skimage import measure
from ipywidgets import interact, FloatSlider, IntSlider, widgets, ToggleButtons
from scipy.ndimage import binary_fill_holes, binary_dilation, binary_erosion
import time

# ---------- Mesh Processing ----------
def load_and_prepare_mesh(stl_path, resolution=64):
    print("Loading and preparing mesh...")
    try:
        mesh_obj = trimesh.load_mesh(stl_path, process=True)
        mesh_obj.apply_translation(-mesh_obj.centroid)
        mesh_obj.apply_scale(1.0/np.max(mesh_obj.extents))

        print("Voxelizing...")
        voxel = mesh_obj.voxelized(pitch=1.0/resolution)
        volume = voxel.matrix.astype(np.float32)

        # Properly fill the interior
        print("Filling interior...")
        volume = binary_fill_holes(volume)
        return volume
    except Exception as e:
        print(f"Error: {e}")
        return None

# ---------- Shape Operations ----------
class Shape:
    def __init__(self, f):
        self.f = f

    def intersect(self, other):
        return Shape(lambda x,y,z: np.maximum(self.f(x,y,z), other.f(x,y,z)))

    def difference(self, other):
        return Shape(lambda x,y,z: np.maximum(self.f(x,y,z), -other.f(x,y,z)))

# ---------- Gyroid Functions ----------
def gyroid_surface(period, thickness):
    """Creates a gyroid surface with specified thickness"""
    base_gyroid = Shape(lambda x,y,z:
        np.sin(2*np.pi*x/period)*np.cos(2*np.pi*y/period) +
        np.sin(2*np.pi*y/period)*np.cos(2*np.pi*z/period) +
        np.sin(2*np.pi*z/period)*np.cos(2*np.pi*x/period))

    # Create thick surface
    return Shape(lambda x,y,z: np.abs(base_gyroid.f(x,y,z)) - thickness/2)

def gyroid_volume(period, density):
    """Creates a solid gyroid infill"""
    return Shape(lambda x,y,z:
        np.sin(2*np.pi*x/period)*np.cos(2*np.pi*y/period) +
        np.sin(2*np.pi*y/period)*np.cos(2*np.pi*z/period) +
        np.sin(2*np.pi*z/period)*np.cos(2*np.pi*x/period) - density)

# ---------- Visualization ----------
def visualize(verts, faces, gen_time):
    plt.figure(figsize=(10,8))
    ax = plt.axes(projection='3d')
    if verts is not None and faces is not None:
        ax.plot_trisurf(verts[:,0], verts[:,1], faces, verts[:,2],
                      color='lightblue', edgecolor='none', alpha=0.8)
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_zlim(-1, 1)
    ax.set_title(f"Generated in {gen_time:.1f} seconds")
    plt.show()

# ---------- Main Pipeline ----------
def run_gyroid_generator(stl_path):
    volume = load_and_prepare_mesh(stl_path)
    if volume is None:
        return

    # Create model SDF (proper interior)
    nz, ny, nx = volume.shape
    z, y, x = np.linspace(-1, 1, nz), np.linspace(-1, 1, ny), np.linspace(-1, 1, nx)
    interp = RegularGridInterpolator((z, y, x), volume, bounds_error=False, fill_value=0)

    def model_sdf(x,y,z):
        points = np.vstack([z.ravel(), y.ravel(), x.ravel()]).T
        return 0.5 - interp(points).reshape(x.shape)

    model_shape = Shape(model_sdf)

    def generate_mesh(shape, resolution):
        try:
            # Create grid with proper dimensions
            x = y = z = np.linspace(-1, 1, resolution)
            X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

            # Evaluate SDF
            vol = shape.f(X, Y, Z)

            # Generate mesh
            verts, faces, _, _ = measure.marching_cubes(vol, level=0)
            # Normalize verts to [-1, 1] space
            scale = 2.0 / (resolution - 1)
            verts = verts * scale - 1.0

            return verts, faces
        except Exception as e:
            print(f"Mesh generation error: {e}")
            return None, None

    def export_stl(verts, faces, filename):
        try:
            stl_mesh = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
            for i, f in enumerate(faces):
                stl_mesh.vectors[i] = verts[f]
            stl_mesh.save(filename)
            return True
        except Exception as e:
            print(f"Export error: {e}")
            return False

    # Interactive controls
    style = {'description_width': '150px'}

    def update(mode='volume', period=1.5, thickness=0.2, density=0.3, resolution=60, export=False):
        start_time = time.time()

        # Generate appropriate gyroid
        gyroid = gyroid_volume(period, density) if mode == 'volume' else gyroid_surface(period, thickness)

        # CORRECTED OPERATION: Use difference to get gyroid INSIDE the model
        combined = model_shape.difference(gyroid)

        verts, faces = generate_mesh(combined, resolution)

        if export and verts is not None:
            if export_stl(verts, faces, f"gyroid_{mode}.stl"):
                print(f"Exported {mode} STL successfully!")

        visualize(verts, faces, time.time()-start_time)

    interact(update,
             mode=ToggleButtons(options=[('Volume Infill', 'volume'),
                                       ('Surface Pattern', 'surface')],
                              description='Gyroid Mode:',
                              style=style),
             period=FloatSlider(min=0.5, max=3.0, step=0.1, value=1.5,
                              description='Pattern Scale:',
                              style=style),
             thickness=FloatSlider(min=0.05, max=0.5, step=0.01, value=0.2,
                                 description='Surface Thickness:',
                                 style=style),
             density=FloatSlider(min=-1.0, max=1.0, step=0.05, value=0.3,
                               description='Infill Density:',
                               style=style),
             resolution=IntSlider(min=40, max=100, step=5, value=60,
                                description='Preview Quality:',
                                style=style),
             export=widgets.ToggleButton(value=False,
                                       description='Export STL',
                                       icon='save'))

# Run with your model
run_gyroid_generator("/content/residual.stl")