import triton
import triton.language as tl

@triton.jit
def hello_triton():
    tl.device_print("Hello Triton!")

hello_triton[(1,)]()