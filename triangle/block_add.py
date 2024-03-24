import torch
import triton
import triton.language as tl

@triton.jit
def add_kernel(x_ptr, y_ptr, z_ptr, size, block_size: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * block_size

    x_block_ptr = tl.make_block_ptr(
        x_ptr,
        shape=(size,),
        strides=(1,),
        offsets=(offsets,),
        block_shape=(block_size,),
        order=(0,),
    )

    y_block_ptr = tl.make_block_ptr(
        y_ptr,
        shape=(size,),
        strides=(1,),
        offsets=(offsets,),
        block_shape=(block_size,),
        order=(0,),
    )

    x = tl.load(x_block_ptr, boundary_check=(0,))
    y = tl.load(y_block_ptr, boundary_check=(0,))
    z = x + y

    z_block_ptr = tl.make_block_ptr(
        z_ptr,
        shape=(size,),
        strides=(1,),
        offsets=(offsets,),
        block_shape=(block_size,),
        order=(0,),
    )

    tl.store(z_block_ptr, z, boundary_check=(0,))

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
