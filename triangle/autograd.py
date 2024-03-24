import torch
import triton
import triton.language as tl

class Mul(torch.autograd.Function):
    @staticmethod
    def forward(ctx: torch.Any, *args: torch.Any, **kwargs: torch.Any) -> torch.Any:
        a, b = args
        ctx.save_for_backward(a, b)
        return a * b
    
    @staticmethod
    def backward(ctx: torch.Any, *grad_outputs: torch.Any) -> torch.Any:
        a, b = ctx.saved_tensors
        return b, a
    
mul = Mul.apply


a = torch.tensor(2.0, requires_grad=True)
b = torch.tensor(4.0)
c = mul(a, b)
print(f"{a} * {b} = {c}")

c.backward()
print((f"a's gradient is {a.grad}"))