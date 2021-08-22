import cupy as cp


def register_host_buffer(buffer):
    for element in buffer:
        ptr = element.ctypes.data
        nbytes = element.nbytes
        cp.cuda.runtime.hostRegister(ptr, nbytes, 1)


def unregister_host_buffer(buffer):
    for element in buffer:
        ptr = element.ctypes.data
        cp.cuda.runtime.hostUnregister(ptr)

