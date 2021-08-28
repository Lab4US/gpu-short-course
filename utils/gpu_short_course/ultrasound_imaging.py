import os
import math
from pathlib import Path
import numpy as np
import cupy as cp
import cupyx.scipy.ndimage
import scipy.signal as signal
import matplotlib.pyplot as plt
from matplotlib import animation
from cupyx.scipy import fftpack


def create_grid(x_mm, z_mm, nx=128, nz=128):
    xgrid = np.linspace(x_mm[0] * 1e-3, x_mm[1] * 1e-3, nx)
    zgrid = np.linspace(z_mm[0] * 1e-3, z_mm[1] * 1e-3, nz)
    return xgrid, zgrid


# DEFAULT GRID
X_MM = [-15, 15]
Z_MM = [20, 50]
X_GRID, Z_GRID = create_grid(X_MM, Z_MM)


# -------------------------------------------- B-mode image reconstruction
class Beamformer:
    def __init__(self, grid_x, grid_z, context, frame_shape):
        n_transmits, n_channels, n_samples = frame_shape
        const_memory_required = 4 * (256 + 256)
        device_props = cp.cuda.runtime.getDeviceProperties(0)
        if device_props['totalConstMem'] < const_memory_required:
            raise ValueError("Not enough constant memory!")
        current_dir = os.path.dirname(os.path.join(os.path.abspath(__file__)))
        kernel_src = Path(os.path.join(current_dir, "beamformer.cc")).read_text()
        self.beamform_module = cp.RawModule(code=kernel_src)
        self.beamform_kernel = self.beamform_module.get_function("beamform")
        self.block = (16, 16)
        self.nx, self.nz = len(grid_x), len(grid_z)
        self.grid = (int(math.ceil(self.nz / self.block[0])), int(math.ceil(self.nx / self.block[1])))
        self.grid_x_gpu = cp.asarray(grid_x, dtype=cp.float32)
        self.grid_z_gpu = cp.asarray(grid_z, dtype=cp.float32)

        probe_width = (n_channels - 1) * context["pitch"]
        elements_x = np.linspace(-probe_width / 2, probe_width / 2, n_channels).astype(np.float32)
        self.elements_x_gpu = self.create_const_array(elements_x, name="elements_x", dtype=cp.float32)
        self.angles_gpu = self.create_const_array(context["angles"], name="angles", dtype=cp.float32)

        self.speed_of_sound = cp.float32(context["speed_of_sound"])
        self.sampling_frequency = cp.float32(context["sampling_frequency"])
        self.output = cp.zeros((self.nx, self.nz), dtype=cp.float32)

    def create_const_array(self, host_data, name, dtype):
        const_mem_ptr = self.beamform_module.get_global(name)
        gpu_data = cp.ndarray(shape=host_data.shape, dtype=dtype, memptr=const_mem_ptr)
        gpu_data.set(host_data)
        return gpu_data

    def process(self, rf):
        rf_gpu = cp.asarray(rf, dtype=cp.float32)
        n_transmits, n_elements, n_samples = rf_gpu.shape
        self.beamform_kernel(self.grid, self.block,
                             args=(
                                 self.output, rf_gpu,
                                 self.grid_z_gpu, self.nz,
                                 self.grid_x_gpu, self.nx,
                                 n_transmits, n_elements, n_samples,
                                 self.speed_of_sound, self.sampling_frequency))
        return self.output


class ToEnvelope:
    def __init__(self):
        self.h = None

    def prepare_if_necessary(self, data):
        if self.h is None:
            n = data.shape[-1]
            ndim = 2
            axis = -1
            h = cp.zeros(n).astype(cp.float32)
            if n % 2 == 0:
                h[0] = h[n // 2] = 1
                h[1:n // 2] = 2
            else:
                h[0] = 1
                h[1:(n + 1) // 2] = 2
            indices = [cp.newaxis] * ndim
            indices[axis] = slice(None)
            self.h = h[tuple(indices)]

    def process(self, data):
        self.prepare_if_necessary(data)
        data = cp.asarray(data)
        xf = fftpack.fft(data, axis=-1)
        result = fftpack.ifft(xf*self.h, axis=-1)
        return cp.abs(result)


class ToBmode:
    def process(self, envelope):
        maximum = cp.max(envelope)
        envelope = envelope / maximum
        envelope = 20 * cp.log10(envelope)
        return envelope


class UltrasoundImaging:

    def __init__(self, context, filter_order, frame_shape):
        transmit_frequency = context["transmit_frequency"]
        sampling_frequency = context["sampling_frequency"]
        filter_order = 64
        filter_coeffs = signal.firwin(
            filter_order, np.array([0.5, 1.5]) * transmit_frequency,
            pass_zero=False, fs=sampling_frequency)
        self.filter_coeffs = cp.asarray(filter_coeffs)
        self.beamformer = Beamformer(X_GRID, Z_GRID, context, frame_shape)
        self.to_envelope = ToEnvelope()
        self.to_bmode = ToBmode()

    def process(self, frame):
        filtered_rf = cupyx.scipy.ndimage.convolve1d(frame, self.filter_coeffs, axis=-1)
        img = self.beamformer.process(filtered_rf)
        envelope = self.to_envelope.process(img)
        bmode = self.to_bmode.process(envelope)
        return bmode


# -------------------------------------------- Display utilities
def display_bmode(img, x=None, z=None):
    if x is None:
        x = X_MM
    if z is None:
        z = Z_MM
    fig, ax = plt.subplots()
    ax.set_xlabel('OX [mm]')
    ax.set_ylabel('OZ [mm]')
    ax.imshow(img, extent=x + z[::-1], cmap='gray', vmin=-30, vmax=0)


def show_cineloop(imgs, value_range=None, cmap=None, figsize=None,
                  interval=50, xlabel="Azimuth (mm)", ylabel="Depth (mm)",
                  extent=None):
    def init():
        img.set_data(imgs[0])
        return (img,)

    def animate(frame):
        img.set_data(imgs[frame])
        return (img,)

    fig, ax = plt.subplots()
    if figsize is not None:
        fig.set_size_inches(figsize)
    img = ax.imshow(imgs[0], cmap=cmap, extent=extent)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if value_range is not None:
        img.set_clim(*value_range)

    return animation.FuncAnimation(
        fig, animate, init_func=init, frames=len(imgs),
        interval=interval, blit=True)


def to_bmode_cpu(rf_beamformed):
    envelope = np.abs(signal.hilbert(rf_beamformed, axis=0))
    res = 20*np.log10(envelope/np.max(envelope))
    return res
