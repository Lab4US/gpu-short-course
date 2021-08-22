import numpy as np
import cupy as cp
import math
import scipy.signal as signal
import matplotlib.pyplot as plt
from scipy.io import loadmat
import pickle
from cupyx.scipy import fftpack
import cupyx.scipy.ndimage
from matplotlib import animation


def dB(data, mx=1):
    return 20*np.log10(data/mx)


def iq2bmode(iq):
    env = np.abs(iq)
    mx = np.nanmax(env)
    env = dB(env, mx=mx)
    return env


def iq2bmode_gpu(iq, mx=1):
    env = cp.abs(iq)
    mx = cp.nanmax(env)
    env = 20*cp.log10(env/mx)
    return env


def power_mask(data, power_dB, power_limit):
    mask = (power_dB < power_limit[0]) | (power_dB >= power_limit[1])
    img = np.copy(data)
    img[mask] = None
    return img


def scale_doppler(doppler_data, doppler_type):

    if doppler_type == 'color':
        doppler_scaled = doppler_data

    elif doppler_type == 'doppler frequency':
        doppler_scaled = doppler_data*1e-3

    elif doppler_type == 'speed':
        doppler_scaled = doppler_data*1e3

    elif doppler_type == 'power':
        doppler_scaled = doppler_data

    elif doppler_type == 'noflow':
        doppler_scaled = doppler_data*np.nan

    else:
        raise ValueError(
        "The 'doppler_type' parameter should be one of the following: "
        "'color', 'power', 'speed', 'doppler frequency' or 'noflow'. ")
    return doppler_scaled

def prepare_doppler(doppler_array, power_dB, power_limit, doppler_type):
    doppler_array = scale_doppler(doppler_array, doppler_type)
    return  power_mask(doppler_array, power_dB, power_limit)


def show_flow(
    bmode, doppler_array, power_dB,
    xgrid=None, zgrid=None,
    doppler_type='power',
    power_limit=(26, 56), color_limit=None,
    bmode_limit=(-60, 0)):
    """
    The function show blood flow on the b-mode image.

    :param bmode: bmode data array,
    :param doppler_array: doppler data array (i.e. raw color,
                          doppler frequency or blood speed),
    :param power_dB: power data array (in [dB]),
    :param xgrid: (optional) vector of 'x' coordinates in [m],
    :param zgrid: (optional) vector of 'z' coordinates in [m],
    :param doppler_type:(optional) type of flow presentation,
        the following types are possible:
        1. 'color' - raw color estimate [radians],
        2. 'doppler frequency' - color scaled in [kHz],
        3. 'power' - power estimate in [dB],
        4. 'speed' - color scaled in [mm/s],
        5. 'noflow' - bmode only,
    :param power_limit: (optional) flow estimate pixels corresponding to
                            power outside power_limit will not be shown,
    :param color_limit: (optional) two element tuple with color limit,
    :param bmode_limit: (optional) two element tuple with bmode limit.

    """

    if doppler_type == 'power':
        doppler_array = power_dB

    if xgrid is not None and zgrid is not None:
        # convert grid from [m] to [mm]
        xgrid = xgrid*1e3
        zgrid = zgrid*1e3
        extent = (min(xgrid), max(xgrid), max(zgrid), min(zgrid))

        # calculate data aspect for proper image proportions
        dx = xgrid[1]-xgrid[0]
        dz = zgrid[1]-zgrid[0]
        data_aspect = dz/dx
        xlabel = '[mm]'
        ylabel = '[mm]'

    else:
        data_aspect = None
        extent = None
        xlabel = 'lines'
        ylabel = 'samples'


    # set appropriate parameters for plt.imshow()
    if doppler_type == 'color':
        cmap = 'bwr'
        title = 'color doppler'
        cbar_label = '[radians]'
        if color_limit is None:
            color_limit = (-1., 1.)

    elif doppler_type == 'doppler frequency':
        cmap = 'bwr'
        title = 'color doppler'
        cbar_label = '[kHz]'
        if color_limit is None:
            color_limit = (-1, 1)

    elif doppler_type == 'speed':
        cmap = 'bwr'
        title = 'color doppler'
        cbar_label = '[mm/s]'
        if color_limit is None:
            color_limit = (-40, 40)

    elif doppler_type == 'power':
        cmap = 'hot'
        title = 'power doppler'
        cbar_label = '[dB]'
        color_limit = None

    elif doppler_type == 'noflow':
        cmap = 'gray'
        title = 'B-mode'
        cbar_label = '[dB]'
        color_limit = bmode_limit

    else:
        raise ValueError(
            "The 'doppler_type' parameter should be one of the following: "
            "'color', 'power', 'speed', 'doppler frequency'. "
            )

    if color_limit is not None:
        vmin = color_limit[0]
        vmax = color_limit[1]

    else:
        vmin = None
        vmax = None

    # scale doppler data and mask pixels corresponding to too low or too high power (outside power_limit)
    doppler_data = prepare_doppler(doppler_array, power_dB, power_limit, doppler_type)

    # draw
    bmode_img = plt.imshow(
        bmode,
        interpolation='bicubic',
        aspect=data_aspect,
        cmap='gray',
        extent=extent,
        vmin=bmode_limit[0],
        vmax=bmode_limit[1],
    )

    flow_img = plt.imshow(
        doppler_data,
        interpolation='bicubic',
        aspect=data_aspect,
        cmap=cmap,
        extent=extent,
        vmin=vmin, vmax=vmax,
    )

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.colorbar(
        flow_img,
        location='bottom',
        pad=0.2,
        aspect=20,
        fraction=0.05,
        label=cbar_label,
    )
    return bmode_img, flow_img, doppler_data


def show_flow_cineloop(
    bmode, doppler_array, power_dB,
    xgrid=None,
    zgrid=None,
    doppler_type='power',
    power_limit=(26, 70),
    color_limit=None,
    bmode_limit=(-60, 0)):

    """
    The function show animation of blood flow on the b-mode image.

    :param bmode: bmode data array,
    :param doppler_array: doppler data array (i.e. raw color, doppler frequency or blood speed),
    :param power_dB: power data array (in [dB]),
    :param xgrid: (optional) vector of 'x' coordinates in [m],
    :param zgrid: (optional) vector of 'z' coordinates in [m],
    :param doppler_type:(optional) type of flow presentation,
        the following types are possible:
        1. 'color' - raw color estimate [radians],
        2. 'doppler frequency' - color scaled in [kHz],
        3. 'power' - power estimate in [dB],
        4. 'speed' - color scaled in [mm/s],
    :param power_limit: (optional) flow estimate pixels corresponding to
                            power outside power_limit will not be shown,
    :param color_limit: (optional) two element tuple with color limit,
    :param bmode_limit: (optional) two element tuple with bmode limit.

    """

    def init():
        bimg.set_data(bmode[:, :, 0])
        fimg.set_data(doppler_img)
        return (bimg, fimg, )

    def animate(frame):
        doppler_img = prepare_doppler(
            doppler_array[:, :, frame],
            power_dB[:, :, frame],
            power_limit,
            doppler_type)

        bimg.set_data(bmode[:, :, frame])
        fimg.set_data(doppler_img)
        return (bimg, fimg, )

    if doppler_type == 'power':
        doppler_array = power_dB

    fig = plt.figure()
    fig.set_facecolor('white')
    bimg, fimg, doppler_img = show_flow(
        bmode[:, :, 0],
        doppler_array[:, :, 0],
        power_dB[:, :, 0],
        xgrid=xgrid,
        zgrid=zgrid,
        doppler_type=doppler_type,
        power_limit=power_limit,
        color_limit=color_limit,
        bmode_limit=bmode_limit)

    return animation.FuncAnimation(
                    fig, animate,
                    init_func=init,
                    frames=bmode.shape[-1],
                    interval=100, blit=True)


def filter_wall_clutter_cpu(input_signal, Wn=0.2, N=32, axis=0):
    sos = signal.butter(N, Wn, 'high', output='sos')
    output_signal = signal.sosfiltfilt(sos, input_signal, axis=axis)
    return output_signal.astype(np.complex64)


def filter_wall_clutter_gpu(input_signal, Wn=0.2, N=33, axis=0):
    if N % 2 == 0:
        N = N+1
    b = signal.firwin(N, Wn, pass_zero=False)
    b = cp.array(b)
    output_signal = cupyx.scipy.ndimage.convolve1d(
        input_signal,
        b,
        axis=axis)
    return output_signal.astype(np.complex64)