import numpy as np
import time
import argparse

DEFAULT_BENCHMARK_H_SIZE = 256


def _exec_tests(input_func, test_func, benchmark_func, **kwargs):
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", dest="mode",
                        help="Working mode.", required=True)
    args, other = parser.parse_known_args()
    input_args = (o.split("=") for o in other)
    input_args = [(key, int(value)) for key, value in input_args]
    input_args = dict(input_args)
    kwargs = {**kwargs, **input_args}
    if args.mode == "test":
        test_func(input_func, **kwargs)
    elif args.mode == "benchmark":
        benchmark_func(input_func, **kwargs)

        
# ------------------------------------------ ADD_VECTORS
def benchmark_add_vectors(func, n=100, size=2**20, dtype=np.float32):
    times = []
    print("Benchmarking the function, please wait...")
    for i in range(n):
        a = np.random.rand(size).astype(dtype)
        b = np.random.rand(size).astype(dtype)
        start = time.time()
        result = func(a, b)
        end = time.time()
        times.append(end-start)
    print("Benchmark result: ")
    print(f"Average processing time: " 
        + f"{np.mean(times):.4f} "
        + f"seconds (+/- {np.std(times).item():.4f}), "
        + f"median: {np.median(times):.4f}")

    
# ------------------------------------------ CONVOLVE
def run_convolve(input_func, **kwargs):
    _exec_tests(input_func, test_convolve, benchmark_convolve, **kwargs)


def test_convolve(func):
    # Test simple case
    x = np.array([0, 1, 2, 3, 4], dtype=np.float32)
    h = np.array([0, 1, 2], dtype=np.float32)
    result = func(x, h)
    np.testing.assert_equal(result, [0, 1, 4, 7, 10])
    
    # Test a bit longer filter.
    x = np.arange(10).astype(np.float32)
    h = np.arange(5).astype(np.float32)
    result = func(x, h)
    np.testing.assert_equal(result, np.convolve(x, h, mode='same'))
        
    # Test if it gives the "same" results as nump.convolve. 
    rng = np.random.default_rng(29062021)
    for i in range(100):
        intervals = rng.integers(low=1, high=30, size=2)
        h_len, x_len = sorted(intervals)
        x = rng.random(x_len).astype(np.float32)
        h = rng.random(h_len).astype(np.float32)
        result = func(x, h)
        np.testing.assert_almost_equal(result, np.convolve(x, h, mode='same'), decimal=5)
    print("All tests passed.")

    
def benchmark_convolve(func, n=100, x_size=2**20, h_size=DEFAULT_BENCHMARK_H_SIZE, dtype=np.float32,
                       quiet=False):
    times = []
    print("Benchmarking the function, please wait...")
    for i in range(n):
        x = np.random.rand(x_size).astype(dtype)
        h = np.random.rand(h_size).astype(dtype)
        start = time.time()
        result = func(x, h)
        end = time.time()
        times.append(end-start)
    if not quiet:
        print("Benchmark result: ")
        print(f"Average processing time: " 
            + f"{np.mean(times):.4f} "
            + f"seconds (+/- {np.std(times).item():.4f}), "
            + f"median: {np.median(times):.4f}")
    

# ------------------------------------------ CONVOLVE_H_CONST
def run_convolve_const(input_func, **kwargs):
    _exec_tests(input_func, test_convolve_const,
                benchmark_convolve_const, **kwargs)


def test_convolve_const(func, h):  
     # Test if it gives the "same" results as nump.convolve. 
    rng = np.random.default_rng(29062021)
    for i in range(100):
        x_len = rng.integers(low=len(h)+1, high=30, size=1)
        x = rng.random(x_len).astype(np.float32)
        result = func(x)
        np.testing.assert_almost_equal(result, np.convolve(x, h, mode='same'), decimal=5)
    print("All tests passed.")

        
def benchmark_convolve_const(func, h, n=100, x_size=2**20, precision_decimal=4):
    import time
    times = []
    for i in range(n):
        x = np.random.rand(x_size).astype(np.float32)
        start = time.time()
        result = func(x)
        end = time.time()

        cpu_result = np.convolve(x, h, mode='same')
        np.testing.assert_almost_equal(result, cpu_result, decimal=precision_decimal)
        times.append(end-start)
    print("Benchmark result: ")
    print(f"Average processing time: " 
        + f"{np.mean(times):.4f} "
        + f"seconds (+/- {np.std(times).item():.4f}), "
        + f"median: {np.median(times):.4f}")
    
    
# ------------------------------------------ CONVOLVE_2D_INPUT
def run_convolve_2d_input(input_func, **kwargs):
    _exec_tests(input_func, test_convolve_2d_input, benchmark_convolve_2d_input, **kwargs)


def test_convolve_2d_input(func, axis, **kwargs):
    x = np.array([[0,   1,  2,  3,  4],
                  [0,  10, 20, 30, 40],
                  [-1, -2, -3, -4, -5]], dtype=np.float32)
    h = np.array([0, 1, 2], dtype=np.float32)
    result = func(x, h)

    if axis == 0:
        np.testing.assert_equal(result, [[ 0,  1,  2,  3,  4],
                                         [ 0, 12, 24, 36, 48],
                                         [-1, 18, 37, 56, 75]])
    elif axis == 1:
        np.testing.assert_equal(result, [[0,   1,  4,   7,  10],
                                         [0,  10, 40,  70, 100],
                                        [-1, -4, -7, -10, -13]])
    else:
        raise ValueError(f"There are no test for axis {axis}.")
    print("All tests passed.")


def benchmark_convolve_2d_input(func, n=100, n_samples=2**10, n_lines=1024, h_size=256, 
                                dtype=np.float32, quiet=False, **kwargs):
    import time
    times = []
    print("Benchmarking, please wait...")
    for i in range(n):
        x = np.random.rand(n_lines, n_samples).astype(dtype)
        h = np.random.rand(h_size).astype(dtype)
        start = time.time()
        result = func(x, h)
        end = time.time()
        times.append(end-start)
    if not quiet:
        print("Benchmark result: ")
        print(f"Average processing time: " 
            + f"{np.mean(times):.4f} "
            + f"seconds (+/- {np.std(times).item():.4f}), "
            + f"median: {np.median(times):.4f}")
