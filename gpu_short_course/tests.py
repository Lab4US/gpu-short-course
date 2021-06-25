import numpy as np
import time

DEFAULT_BENCHMARK_H_SIZE = 256


def benchmark_add_vectors(func, n=100, size=2**20, dtype=np.float32):
    import time
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


def test_convolve(func, is_h_const=False):
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

    
def benchmark_convolve(func, n=100, x_size=2**20, h_size=DEFAULT_BENCHMARK_H_SIZE, dtype=np.float32):
    import time
    times = []
    print("Benchmarking the function, please wait...")
    for i in range(n):
        x = np.random.rand(x_size).astype(dtype)
        h = np.random.rand(h_size).astype(dtype)
        start = time.time()
        result = func(x, h)
        end = time.time()
        times.append(end-start)
    print("Benchmark result: ")
    print(f"Average processing time: " 
        + f"{np.mean(times):.4f} "
        + f"seconds (+/- {np.std(times).item():.4f}), "
        + f"median: {np.median(times):.4f}")
    

def test_convolve_const(func, h):  
     # Test if it gives the "same" results as nump.convolve. 
    rng = np.random.default_rng(29062021)
    for i in range(100):
        x_len = rng.integers(low=len(h)+1, high=30, size=1)
        x = rng.random(x_len).astype(np.float32)
        result = func(x)
        np.testing.assert_almost_equal(result, np.convolve(x, h, mode='same'), decimal=5)
    print("All tests passed.")

        
def benchmark_convolve_const(func, h, n=100, x_size=2**20):
    import time
    times = []
    for i in range(n):
        x = np.random.rand(x_size).astype(np.float32)
        start = time.time()
        result = func(x)
        end = time.time()
        times.append(end-start)
    print("Benchmark result: ")
    print(f"Average processing time: " 
        + f"{np.mean(times):.4f} "
        + f"seconds (+/- {np.std(times).item():.4f}), "
        + f"median: {np.median(times):.4f}")

    
def benchmark_convolve_2d_input(func, n=100, n_samples=2**10, n_lines=1024, h_size=DEFAULT_BENCHMARK_H_SIZE, dtype=np.float32):
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
    print("Benchmark result: ")
    print(f"Average processing time: " 
        + f"{np.mean(times):.4f} "
        + f"seconds (+/- {np.std(times).item():.4f}), "
        + f"median: {np.median(times):.4f}")
