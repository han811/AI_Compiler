import torch
import triton
import triton.language as tl

@triton.jit
def combine_add(a, b):
    return a + b

@triton.jit
def sum_kernel(y_ptr, x_ptr, size, block_size: tl.constexpr):
    pid = tl.program_id(0)
    tl.device_print("pid: ", pid)
    offsets = tl.arange(0, block_size)
    mask = offsets < size

    x = tl.load(x_ptr + offsets, mask)
    y = tl.reduce(x, 0, combine_add)
    tl.store(y_ptr, y)

def sum(x):
    size = x.numel()
    y = torch.empty(1, device="cuda")

    def grid(meta):
        return (1,)
    
    sum_kernel[grid](y, x, size, triton.next_power_of_2(size))

    return y

x = torch.randn(1024, device="cuda")

a = sum(x)
b = torch.sum(x)

assert torch.allclose(a, b)