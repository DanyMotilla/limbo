# Install dependencies (including TorchIO)
!pip install trimesh numpy-stl ipywidgets scikit-image torchio nibabel --quiet

import trimesh
import numpy as np
import matplotlib.pyplot as plt
from stl import mesh
from scipy.interpolate import RegularGridInterpolator
from skimage import measure
from ipywidgets import interact, FloatSlider, IntSlider, widgets, ToggleButtons
from scipy.ndimage import binary_fill_holes, binary_dilation, binary_erosion
import time
import os
import torchio as tio
import tempfile
from google.colab import files

# ---------- Tomography Data Loading ----------
def load_brain_sample():
    """Download and load a sample brain MRI dataset using TorchIO"""
    print("Downloading sample brain data...")

    # Download a sample brain MRI dataset
    subject = tio.datasets.Colin27()

    # Get the T1-weighted image
    t1_image = subject.t1

    print(f"Loaded sample brain MRI with shape: {t1_image.shape}")

    # Convert to numpy array and normalize
    volume_data = t1_image.data.numpy().squeeze()

    # Normalize to [0, 1] range
    volume_data = (volume_data - volume_data.min()) / (volume_data.max() - volume_data.min())

    return volume_data

def get_brain_slices(volume_data, axis=0):
    """Get slices from the brain volume along specified axis"""
    if axis == 0:
        slices = [volume_data[i, :, :] for i in range(volume_data.shape[0])]
    elif axis == 1:
        slices = [volume_data[:, i, :] for i in range(volume_data.shape[1])]
    else:
        slices = [volume_data[:, :, i] for i in range(volume_data.shape[2])]

    return slices

def visualize_tomography_slices(volume_data, axis=0, num_slices=9):
    """Visualize sample slices from the tomography data"""
    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    axes = axes.flatten()

    # Get indices for evenly spaced slices
    if axis == 0:
        indices = np.linspace(0, volume_data.shape[0] - 1, num_slices, dtype=int)
        title_prefix = "Axial"
        for i, idx in enumerate(indices):
            if i < len(axes):
                axes[i].imshow(volume_data[idx, :, :], cmap='gray')
    elif axis == 1:
        indices = np.linspace(0, volume_data.shape[1] - 1, num_slices, dtype=int)
        title_prefix = "Coronal"
        for i, idx in enumerate(indices):
            if i < len(axes):
                axes[i].imshow(volume_data[:, idx, :], cmap='gray')
    else:
        indices = np.linspace(0, volume_data.shape[2] - 1, num_slices, dtype=int)
        title_prefix = "Sagittal"
        for i, idx in enumerate(indices):
            if i < len(axes):
                axes[i].imshow(volume_data[:, :, idx], cmap='gray')

    # Set titles and turn off axes
    for i, idx in enumerate(indices):
        if i < len(axes):
            axes[i].set_title(f'{title_prefix} Slice {idx}')
            axes[i].axis('off')

    plt.tight_layout()
    plt.show()

# ---------- Mesh Processing (Original Functionality) ----------
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

# ---------- 3D Processing from Tomography ----------
def tomography_to_mesh(volume_data, threshold=0.5):
    """Convert tomography volume to mesh using marching cubes"""
    start_time = time.time()

    print("Generating 3D mesh from tomography data...")
    try:
        # Use marching cubes to create a mesh from the tomography data
        verts, faces, _, _ = measure.marching_cubes(volume_data, level=threshold)

        # Normalize verts to [-1, 1] space
        for i in range(3):
            verts[:, i] = 2.0 * (verts[:, i] / volume_data.shape[i]) - 1.0

        print(f"Mesh generated in {time.time() - start_time:.2f} seconds")
        print(f"Mesh has {len(verts)} vertices and {len(faces)} faces")

        return verts, faces
    except Exception as e:
        print(f"Error generating mesh: {e}")
        return None, None

def export_stl(verts, faces, filename):
    """Export mesh as STL file"""
    try:
        stl_mesh = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
        for i, f in enumerate(faces):
            stl_mesh.vectors[i] = verts[f]
        stl_mesh.save(filename)
        files.download(filename)  # Trigger download in Colab
        print(f"STL file '{filename}' created and download started")
        return True
    except Exception as e:
        print(f"Export error: {e}")
        return False

# ---------- Run Tomography Analysis ----------
def run_tomography_analysis():
    print("Loading brain tomography data...")
    brain_volume = load_brain_sample()

    # Create interface for tomography visualization
    def on_view_slices(axis):
        visualize_tomography_slices(brain_volume, axis=axis)

    def on_create_3d(threshold, export_stl_file):
        verts, faces = tomography_to_mesh(brain_volume, threshold)
        if verts is not None and faces is not None:
            visualize(verts, faces, 0)
            if export_stl_file:
                export_stl(verts, faces, "brain_model.stl")

    print("\nBrain Tomography Visualization")
    print("--------------------------------")

    # Create interactive controls
    interact(
        on_view_slices,
        axis=widgets.RadioButtons(
            options=[('Axial (X)', 0), ('Coronal (Y)', 1), ('Sagittal (Z)', 2)],
            description='Slice Direction:',
            disabled=False
        )
    )

    interact(
        on_create_3d,
        threshold=FloatSlider(
            min=0.1, max=0.9, step=0.05, value=0.5,
            description='3D Threshold:',
            style={'description_width': 'initial'}
        ),
        export_stl_file=widgets.Checkbox(
            value=False,
            description='Export as STL',
            disabled=False
        )
    )

# ---------- Main Pipeline (Original) ----------
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

    def update(mode='volume', period=1.5, thickness=0.2, density=0.3, resolution=60, export=False):
        start_time = time.time()

        # Generate appropriate gyroid
        gyroid = gyroid_volume(period, density) if mode == 'volume' else gyroid_surface(period, thickness)

        # CORRECTED OPERATION: Use difference to get gyroid INSIDE the model
        combined = model_shape.difference(gyroid)

        verts, faces = generate_mesh(combined, resolution)

        if export and verts is not None:
            filename = f"gyroid_{mode}.stl"
            if export_stl(verts, faces, filename):
                print(f"Exported {mode} STL successfully!")

        visualize(verts, faces, time.time()-start_time)

    style = {'description_width': '150px'}
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

# ---------- Main Selection ----------
def main():
    mode = widgets.RadioButtons(
        options=['Tomography Analysis', 'STL Mesh Processing'],
        description='Select mode:',
        disabled=False
    )

    stl_upload = widgets.FileUpload(
        accept='.stl',
        multiple=False,
        description='Upload STL (for Mesh Processing)',
        layout={'visibility': 'hidden'}
    )

    run_button = widgets.Button(description='Run Selected Mode')
    output = widgets.Output()

    def update_visibility(_):
        if mode.value == 'STL Mesh Processing':
            stl_upload.layout.visibility = 'visible'
        else:
            stl_upload.layout.visibility = 'hidden'

    mode.observe(update_visibility, 'value')

    def on_run_button_clicked(_):
        with output:
            output.clear_output()
            if mode.value == 'Tomography Analysis':
                run_tomography_analysis()
            else:
                if not stl_upload.value:
                    print("Please upload an STL file first")
                    return

                # Save the uploaded file temporarily
                for filename, data in stl_upload.value.items():
                    with open(filename, 'wb') as f:
                        f.write(data)
                    run_gyroid_generator(filename)

    run_button.on_click(on_run_button_clicked)

    # Display widgets
    display(widgets.VBox([
        widgets.HTML("<h2>3D Tomography and Mesh Processing Tool</h2>"),
        mode,
        stl_upload,
        run_button,
        output
    ]))

# Run the main program
main()