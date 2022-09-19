import cupy as cp
import numpy as np
import pathlib


## LUT

def prepare_beamformer(output_grid, device, medium, tx_rx_sequence):
    # Initial assignments.
    x_grid, y_grid, z_grid = output_grid
    n_elements = device["n_elements_xy"][0] 
    # Note: we are assumming a probe with square (n+n) aperture.
    assert n_elements == device["n_elements_yx"][0]
    pitch = device["pitch"][0]
    angles = tx_rx_sequence["angles"][:]
    c = medium["speed_of_sound"][0]
    
    # Compute position of the each probe element.
    ri = cp.arange(n_elements)-n_elements//2+0.5
    ri = ri*pitch
    
    # Restructure 
    x = cp.asarray(x_grid).reshape(1, -1, 1)      # (1,   nx, 1)
    y = cp.asarray(y_grid).reshape(1, -1, 1)      # (1,   ny, 1)
    z = cp.asarray(z_grid).reshape(1, 1, -1)      # (1,   1,  nz)
    gamma = cp.asarray(angles).reshape(-1, 1, 1)  # (ntx, 1,  1)
    ri = cp.asarray(ri).reshape(-1, 1, 1)         # (nrx, 1,  1)

    # TX delays
    tx_distance = z*np.cos(gamma) + x*cp.sin(gamma)  # [m]
    tx_delays = tx_distance/c  # [s]

    # RX delays
    rx_distance = cp.sqrt(z**2 + (y-ri)**2)
    rx_delays = rx_distance/c
    
    # "TX apodization"
    d1 = -n_elements/2 * pitch
    d2 = n_elements/2 * pitch
    tx_apodization_left = z*cp.sin(gamma) + (d1-x)*cp.cos(gamma) <= 0
    tx_apodization_right = z*cp.sin(gamma) + (d2-x)*cp.cos(gamma) >= 0
    tx_apodization = np.logical_and(tx_apodization_left, tx_apodization_right)
    tx_apodization = tx_apodization.astype(np.uint8)

    # RX apodization
    max_rx_tang = 0.5
    rx_sigma = 1/2
    rx_tang = cp.abs((ri-y)/z)
    rx_apodization = cp.exp(-(rx_tang/max_rx_tang)**2 / (2*rx_sigma))
    rx_apodization[rx_tang > max_rx_tang] = 0.0

    return {
        "tx_delays": tx_delays,
        "rx_delays": rx_delays,
        "tx_apodization": tx_apodization,
        "rx_apodization": rx_apodization,
    }


def delay_and_sum_lut(input_array,
                      output_grid,
                      medium, device, tx_rx_sequence,
                      tx_apodization, rx_apodization,
                      tx_delays, rx_delays):
    # Initialization
    # Compile beamformer source code.
    kernel_source = pathlib.Path("1_delayAndSumLUT.cc").read_text()
    kernel_module = cp.RawModule(code=kernel_source)
    kernel_module.compile()
    kernel = kernel_module.get_function("delayAndSumLUT")
    n_tx, n_rx, n_samples = input_array.shape
    x_grid, y_grid, z_grid = output_grid
    output_array = cp.zeros((n_tx, len(y_grid), len(x_grid), len(z_grid)),
                            dtype=cp.float32)
    
    init_delay = tx_rx_sequence["init_delay"][0].item()
    fc = tx_rx_sequence["transmit_frequency"][0].item()
    fs = device["sampling_frequency"][0].item()
    
    input_array = cp.asarray(input_array).astype(cp.float32)
    n_tx, n_y_pix, n_x_pix, n_z_pix = output_array.shape
    params = (
        output_array,
        input_array,
        tx_apodization.astype(cp.uint8),
        rx_apodization.astype(cp.float32),
        tx_delays.astype(cp.float32),
        rx_delays.astype(cp.float32),
        np.float32(init_delay),
        n_tx, n_samples, n_rx,
        n_y_pix, n_x_pix, n_z_pix,
        np.float32(fs), np.float32(fc)
    )
    x_block_size = min(n_x_pix, 8)
    y_block_size = min(n_y_pix, 8)
    z_block_size = min(n_z_pix, 8)
    block_size = (
        z_block_size,
        x_block_size,
        y_block_size
    )
    grid_size = (
        (n_z_pix-1)//z_block_size+1,
        (n_x_pix-1)//x_block_size+1,
        (n_y_pix-1)//y_block_size+1
    )
    kernel(
        grid_size,
        block_size,
        params
    )
    return output_array.get()


## OTF
def expand_grid(grid):
    x_reg_grid, y_reg_grid, z_reg_grid = grid
    return (
        cp.arange(*x_reg_grid),
        cp.arange(*y_reg_grid),
        cp.arange(*z_reg_grid)
    )


def prepare_beamformer_otf(output_grid, **kwargs):
    kwargs["output_grid"] = expand_grid(output_grid)
    result = prepare_beamformer(**kwargs)
    result.pop("tx_delays", None)
    result.pop("rx_delays", None)
    return result


def delay_and_sum_otf(input_array,
                      output_grid,
                      tx_apodization, rx_apodization,
                      medium, device, tx_rx_sequence):
    # Initialization
    # Compile beamformer source code.
    kernel_source = pathlib.Path("2_delayAndSumOTF.cc").read_text()
    kernel_module = cp.RawModule(code=kernel_source)
    kernel_module.compile()
    kernel = kernel_module.get_function("delayAndSumOTF")
    n_tx, n_rx, n_samples = input_array.shape
    
    x_regular_grid, y_regular_grid, z_regular_grid = output_grid
    x_start, x_end, dx = x_regular_grid
    y_start, y_end, dy = y_regular_grid
    z_start, z_end, dz = z_regular_grid
    n_x_pix = int(round((x_end-x_start)/dx))
    n_y_pix = int(round((y_end-y_start)/dy))
    n_z_pix = int(round((z_end-z_start)/dz))
    
    output_array = cp.zeros((n_tx, n_x_pix, n_y_pix, n_z_pix), dtype=cp.float32)
    input_array = cp.asarray(input_array).astype(cp.float32)
    
    angles = cp.asarray(tx_rx_sequence["angles"][:]).astype(cp.float32)
    init_delay = tx_rx_sequence["init_delay"][0].item()
    n_elements = device["n_elements_yx"][0].item()
    pitch = device["pitch"][0].item()
    speed_of_sound = medium["speed_of_sound"][0].item()
    fs = device["sampling_frequency"][0].item()
    fc = tx_rx_sequence["transmit_frequency"][0].item()
        
    params = (
        output_array,
        input_array,
        tx_apodization.astype(cp.uint8),
        rx_apodization.astype(cp.float32),
        #
        np.float32(speed_of_sound),
        np.float32(pitch),
        n_elements,
        angles, 
        np.float32(x_start), np.float32(dx),
        np.float32(y_start), np.float32(dy),
        np.float32(z_start), np.float32(dz),
        #
        np.float32(init_delay),
        n_tx, n_samples, n_rx,
        n_x_pix, n_y_pix, n_z_pix,
        np.float32(fs), np.float32(fc)
    )
    x_block_size = min(n_x_pix, 8)
    y_block_size = min(n_y_pix, 8)
    z_block_size = min(n_z_pix, 8)
    block_size = (
        z_block_size,
        x_block_size,
        y_block_size
    )
    grid_size = (
        (n_z_pix-1)//z_block_size+1,
        (n_x_pix-1)//x_block_size+1,
        (n_y_pix-1)//y_block_size+1
    )
    kernel(
        grid_size,
        block_size,
        params
    )
    return output_array.get()


