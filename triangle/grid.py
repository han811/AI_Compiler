import triton
import triton.language as tl

@triton.jit
def print_grid():
    x_pid = tl.program_id(0)
    y_pid = tl.program_id(1)
    tl.device_print("x_pid: ", x_pid)
    tl.device_print("y_pid: ", y_pid)

def grid(meta):
    return (4, 2)

print_grid[grid]()