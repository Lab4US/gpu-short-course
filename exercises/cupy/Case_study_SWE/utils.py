import cupy as cp
import cupyx.scipy.ndimage
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from matplotlib import animation


def show_frame(
    data, 
    dx=1e-4, dy=1e-4, 
    title=None, 
    vmin=None, vmax=None,
    cmap='jet',
    nxticks=5,
    nyticks=5,
    row0=64,
    show=True
):
    
    nrow, ncol = data.shape
    xticks = np.linspace(0, ncol, nxticks)
    xticklabels = np.round((xticks-ncol/2)*dx*1e3, 0)
    
    y0 = row0*dy*1e3
    yticks = np.linspace(0, nrow, nyticks)
    yticklabels = np.round((row0 + yticks)*dy*1e3, 0) 
    
    fig = plt.figure()
    im = plt.imshow(
       data, 
       cmap=cmap, 
       aspect='equal', 
       vmin=vmin, 
       vmax=vmax,
    )
    plt.xticks(xticks, xticklabels)
    plt.yticks(yticks, yticklabels)
    plt.xlabel('[mm]')
    plt.ylabel('[mm]')
    plt.colorbar()
    if show:
        plt.show()
    return fig, im


def show_waves(data, vmin=None, vmax=None):
    """
    """
    
    def init():
        im.set_data(data[0, :, :].T)
        return im
    
    def animate(frame):
        im.set_data(data[frame, :, :].T)
        return im

    fig, im = show_frame(
        data[0,:,:].T, 
        vmin=vmin, 
        vmax=vmax,
        show=False,
    )
    # fig = plt.figure()
    fig.set_facecolor('white')
    # im = plt.imshow(
    #     data[0,:,:].T, 
    #     cmap='jet',
    #     aspect='equal',
    #     vmin=vmin, vmax=vmax
    # )
    # fig.colorbar(im)

    return animation.FuncAnimation(
        fig, animate,
        init_func=init,
        frames=data.shape[0],
        interval=100, blit=False)


def import_data_from_matfile(file):
    from scipy.io import loadmat
    matdata = loadmat(file)
    print(matdata.keys())
    output = {
        "iq_hri":        matdata["HRI_data"],
        "nchan":     matdata["N_channels"],
        "nframe":    matdata["N_frames"],
        "nsamp":     matdata["N_samples"],
        "c":         matdata["c"],
        "pri":       matdata["frame_pri"],
        "fs":        matdata["fs"],
        "pitch":     matdata["probe_pitch"],
        "tx_angles": matdata["tx_angles"],
        "tx_freq":   matdata["tx_freq"],
        "grid_dx": matdata["grid_pitch"],
    }
    return output

def import_data_from_matfile2(file):
    from scipy.io import loadmat
    matdata = loadmat(file)
    # print(matdata.keys())
    output = {
        "iq_hri":        matdata["HRI_data"],
        "pri":       matdata["frame_pri"],
        "dx":     matdata["grid_pitch"],
    }
    return output

def anim2gif(anim, filename="demo.gif"):
    writer = animation.PillowWriter(fps=10) 
    anim.save(filename, writer=writer)
    
    
def filter_gpu(input_signal, Wn=0.2, N=33, axis=0, pass_zero=False, output_type=None):
    '''
    Filter signal on gpu.
    
    :param input_signal: input signal. Should be of cupy.ndarray type
    :param Wn: cut-off frequency or band edges (from 0 to fs/2)
    :param N: filter length
    :param axis: axis along which filtration will be performed
    :param pass_zero: one of the following True, False, ‘bandpass’, ‘lowpass’, ‘highpass’, ‘bandstop’
    :param output_type: output type
    '''
    if N % 2 == 0:
        N = N+1
    b = signal.firwin(N, Wn, pass_zero=pass_zero)
    b = cp.array(b)
    output_signal = cupyx.scipy.ndimage.convolve1d(
        input_signal,
        b,
        axis=axis)
    if output_type is None:
        output_type = input_signal.dtype
    return output_signal.astype(output_type)


def wavefront_separation(frames):
    '''
    Separates wavefronts propagating in oposite directions (left-right).
    
    :params frames: array of frames (cupy.ndarray or numpy.ndarray)
    '''
    if isinstance(frames, np.ndarray):
        xp = np
    elif isinstance(frames, cp.ndarray):
        xp = cp
    else:
        raise TypeError('Input variable must be numpy.ndarray or cupy.ndarray type.')
    # some reorganization for more convenient use of fft2
    frames = xp.moveaxis(frames, (0,1,2), (2,1,0))
    
    # create spectral masks for wavefront separation
    ncol, nrow = frames.shape[-2:]
    r_mask = xp.ones((ncol, nrow))
    r_mask[:int(nrow/2), :int(ncol/2)] = 0
    r_mask[(xp.ceil(nrow/2).astype(int) + 1):, (xp.ceil(nrow/2).astype(int) + 1):] = 0
    l_mask = xp.fliplr(r_mask)

    # separation in spectral domain
    f_frames = xp.fft.fft2(frames, axes=(1,2))
    l_frames = xp.fft.ifft2(f_frames*l_mask).real
    r_frames = xp.fft.ifft2(f_frames*r_mask).real

    # return to 'proper' dimension order
    l_frames = xp.moveaxis(l_frames, (2,1,0), (0,1,2))
    r_frames = xp.moveaxis(r_frames, (2,1,0), (0,1,2))
    
    # get real
    l_frames = xp.real(l_frames)
    r_frames = xp.real(r_frames)
    
    return l_frames, r_frames


def estimate_lag(
    data, 
    kernel, 
    d=50, 
    block_size_x=32,
    block_size_z=32,
):
    '''
    Estimate shear wave propagation time of flight.
    '''
       
    # define grid
    nframes, nx, nz = data.shape
    grid_size_x = np.ceil(nx/block_size_x).astype(int)
    grid_size_z = np.ceil(nz/block_size_z).astype(int)
    block = (block_size_z, block_size_x)
    grid = (grid_size_z, grid_size_x)
    
    # allocate gpu memory
    lags = cp.zeros((nx-d, nz)).astype(cp.float32)
    cors = cp.zeros((nx-d, nz)).astype(cp.float32)
    data = cp.array(data, order='C').astype(cp.float32)
    
    # use kernel
    kernel_args = (lags, cors, data, d, nframes, nx, nz)
    kernel(grid, block, kernel_args)
    
    return lags, cors


def estimate_phase_shift(
    data,
    phase_shift_kernel,
    nbatchframes=6,
    step=1,
    block_size_x=16,
    block_size_z=16,
    block_size_t=4,    
):
    '''
    Estimate phase shift in 'slow-time'.
    '''

    # calculate number of batches
    nframes, nx, nz = data.shape
    nstep = np.floor((nframes - nbatchframes)/step).astype(int)
    nbatch = nstep + 1

    # specify block and grid sizes
    grid_size_x = np.ceil(nx/block_size_x).astype(int)
    grid_size_z = np.ceil(nz/block_size_z).astype(int)
    grid_size_t = np.ceil(nbatch/block_size_t).astype(int)
    
    # define block and grid
    block = (block_size_z, block_size_x, block_size_t)
    grid = (grid_size_z, grid_size_x, grid_size_t)

    # allocate memory on gpu for output data
    ps = cp.zeros((nbatch, nx, nz)).astype(cp.float32)

    # use kernel
    kernel_args = (ps, data, nbatch, nbatchframes, nx, nz, step)
    phase_shift_kernel(grid, block, kernel_args)
    
    return ps


def process_all(iq, d=40, dx=0.1*1e-3, pri=2e-4):
    # create kernels
    phase_shift_kernel_src = open("phase_shift.cc").read()
    phase_shift_kernel = cp.RawKernel(phase_shift_kernel_src, 'phase_shift')
    
    lag_kernel_src = open("correlation.cc").read()
    lag_kernel = cp.RawKernel(lag_kernel_src, 'correlation')
    
    # move input data to gpu
    iq_gpu = cp.array(iq, order='C').astype(cp.complex64)
    
    # initial band-pass filtration
    iq_gpu = filter_gpu(iq_gpu, Wn=(0.1, 0.2), N=32, axis=0, pass_zero='bandpass')
    
    # slow time phase shift estimation
    ps = estimate_phase_shift(iq_gpu, phase_shift_kernel)
    
    # left and right propagating wavefronts separation
    ps_l, ps_r = wavefront_separation(ps)
    
    ps_l = ps_l[:48,:,:]
    ps_r = ps_r[:48,:,:]
    
    # estimate lag maps
    lag_l, cor_l = estimate_lag(ps_l, lag_kernel, d)
    lag_r, cor_r = estimate_lag(ps_r, lag_kernel, d)
    lag = (lag_l + lag_r)/2
    
    # gaussian filtration
    lag = cupyx.scipy.ndimage.gaussian_filter(lag, 11)
    
    # scaling
    distance = d*dx
    tof = lag*pri
    swe = distance/tof
    return swe.get()
