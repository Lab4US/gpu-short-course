import panel as pn
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import h5py
import numpy as np


def pretty_print_h5_group(group):
    """
    Pretty print the content of input h5py Group.
    """
    def print_values(name, node):
        if isinstance(node, h5py.Dataset):
            if node.shape == (1, ):
                # print scalar value
                print(f"{name}: {float(node[0]):.4f}")
            else:
                print(f"{name}: {node.shape}, {node.dtype}")
        else:
            print(f"Group: {name}")
    group.visititems(print_values) 


def view_medium_and_probe_3d(medium, device, probe_c, **kwargs):
    array = medium["speed_of_sound_array"][:]
    probe = device["probe_mask"][:]
    array[probe == 1] = probe_c
    volume = pn.pane.VTKVolume(array, sizing_mode='stretch_both', **kwargs)
    return pn.Row(volume.controls(jslink=True), volume)


def view_medium_and_probe_2d(medium, device, probe_c, figsize=(10, 10)):
    array = medium["speed_of_sound_array"][:]
    probe = device["probe_mask"][:]
    dx = medium["dx"][:].item()
    array[probe == 1] = probe_c
    fig, (ax_oxz, ax_oyz) = plt.subplots(1, 2)
    fig.subplots_adjust(left=0.02, bottom=0.06, right=0.95, top=0.94, wspace=0.2)
    fig.set_size_inches(figsize)
    nx, ny, nz = array.shape
    extent_oxz = np.array([-nx//2, nx//2, nz, 0])*dx*1e3  # (mm)
    extent_oyz = np.array([-ny//2, ny//2, nz, 0])*dx*1e3  # (mm)
    img = ax_oxz.imshow(array[:, ny//2, :].T, extent=extent_oxz, cmap="plasma")
    ax_oxz.set_title("$y=0$")
    ax_oxz.set_xlabel("OX (mm)")
    ax_oxz.set_ylabel("OZ (mm)")
    cbar = fig.colorbar(img, ax=ax_oxz)
    cbar.ax.set_ylabel("Speed of sound (m/s)")
    img = ax_oyz.imshow(array[nx//2, :, :].T, extent=extent_oyz, cmap="plasma")
    ax_oyz.set_xlabel("OY (mm)")
    ax_oyz.set_ylabel("OZ (mm)")
    ax_oyz.set_title("$x=0$")
    cbar = fig.colorbar(img, ax=ax_oyz)
    cbar.ax.set_ylabel("Speed of sound (m/s)")
    
    
def view_volume_3d(volume, **kwargs):
    volume = pn.pane.VTKVolume(volume, sizing_mode='stretch_both', **kwargs)
    return pn.Row(volume.controls(jslink=True), volume)


def view_volume_2d(volume, x_grid, y_grid, z_grid, figsize=(10, 10)):
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    fig, (ax_oxz, ax_oyz) = plt.subplots(1, 2)
    fig.subplots_adjust(left=0.02, bottom=0.06, right=0.95, top=0.94, wspace=0.2)
    fig.set_size_inches(figsize)
    nz, nx, ny = volume.shape
    extent_oxz = np.array([np.min(x_grid), np.max(x_grid), np.max(z_grid), np.min(z_grid)])*1e3  # (mm)
    extent_oyz = np.array([np.min(y_grid), np.max(y_grid), np.max(z_grid), np.min(z_grid)])*1e3  # (mm)
    img = ax_oxz.imshow(volume[:, :, ny//2], extent=extent_oxz, cmap="gray")
    ax_oxz.set_title("$y=0$")
    ax_oxz.set_xlabel("OX (mm)")
    ax_oxz.set_ylabel("OZ (mm)")
    cbar = fig.colorbar(img, ax=ax_oxz)
    cbar.ax.set_ylabel("Amplitude (dB)")
    img = ax_oyz.imshow(volume[:, nx//2, :], extent=extent_oyz, cmap="gray")
    ax_oyz.set_xlabel("OY (mm)")
    ax_oyz.set_ylabel("OZ (mm)")
    ax_oyz.set_title("$x=0$")
    cbar = fig.colorbar(img, ax=ax_oyz)
    cbar.ax.set_ylabel("Amplitude (dB)")