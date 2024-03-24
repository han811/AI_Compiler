import torch
import triton
import triton.language as tl

@triton.jit
def add_kernel(x_ptr, y_ptr, z_ptr, size, block_size: tl.constexpr):
    pid = tl.program_id(0)
    offsets = tl.arange(0, block_size) + pid * block_size
    mask = offsets < size

    x = tl.load(x_ptr + offsets, mask)
    y = tl.load(y_ptr + offsets, mask)
    z = x + y

    tl.store(z_ptr + offsets, z, mask)

def add(x, y):
    z = torch.empty_like(x, device="cuda")
    size = z.numel()

    def grid(meta):
        return (triton.cdiv(size, meta["block_size"]),)
    
    add_kernel[grid](x, y, z,  size, 1024)

    return z

size = 2**16

x = torch.randn(size, device="cuda")
y = torch.randn(size, device="cuda")

a = add(x, y)
b = x + y
assert torch.allclose(a, b)
