import cupy as cp

from gpu_short_course.tests import (
    run_convolve,
    run_convolve_const,
    run_convolve_2d_input
)

def print_device_info():
    n_gpus = cp.cuda.runtime.getDeviceCount()
    if n_gpus == 0:
        print("NO GPU AVAILABLE.")
    else:
        for i in range(n_gpus):
            device_props = cp.cuda.runtime.getDeviceProperties(i)
            print(f"GPU:{i}: {device_props['name']}")

print_device_info()

