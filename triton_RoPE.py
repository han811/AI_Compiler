import torch
import triton
import triton.language as tl


@triton.jit
def rope_fw(
    t_ptr,
    freqs_ptr,
    out_ptr,
    f_size,
    hb_size,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offset = tl.arange(0, BLOCK_SIZE)
    mask = offset < f_size
    t = tl.load(t_ptr + offset + pid * f_size, mask=mask)
    freqs = tl.load(freqs_ptr + offset + (pid // hb_size) * f_size, mask=mask)

    f_half_size = f_size // 2
    new_offset = tl.where(
        offset >= f_half_size, offset - f_half_size, offset + f_half_size
    )
    half_t = tl.load(t_ptr + new_offset + pid * f_size, mask=mask)
    half_t_rotated = tl.where(new_offset >= f_half_size, -half_t, half_t)

    tmp = t * tl.cos(freqs) + half_t_rotated * tl.sin(freqs)
    tmp = tl.where(new_offset < f_size, tmp, 0.0)

    tl.store(out_ptr + offset + pid * f_size, tmp, mask=mask)


@triton.jit
def rope_bw(
    grad_ptr,
    freqs_ptr,
    out_ptr,
    f_size,
    hb_size,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offset = tl.arange(0, BLOCK_SIZE)
    mask = offset < f_size
    grad = tl.load(grad_ptr + offset + pid * f_size, mask=mask)
    freqs = tl.load(freqs_ptr + offset + (pid // hb_size) * f_size, mask=mask)

    f_half_size = f_size // 2
    new_offset = tl.where(
        offset >= f_half_size, offset - f_half_size, offset + f_half_size
    )
    half_grad = tl.load(grad_ptr + new_offset + pid * f_size, mask=mask)
    half_grad_rotated = tl.where(new_offset >= f_half_size, half_grad, -half_grad)

    tmp = grad * tl.cos(freqs) + half_grad_rotated * tl.sin(freqs)
    tmp = tl.where(new_offset < f_size, tmp, 0.0)

    tl.store(out_ptr + offset + pid * f_size, tmp, mask=mask)


class _rotary_pos_emb(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: torch.Any,
        t: torch.Tensor,
        freqs: torch.Tensor,
    ) -> torch.Any:
        # (Seq, batch, head, Feature)
        # (Seq, 1, 1, Feature)
        ctx.save_for_backward(t, freqs)

        s_size, b_size, h_size, f_size = t.shape

        t_copy = t.reshape(s_size * b_size * h_size, f_size).contiguous()
        freqs_copy = freqs.reshape(s_size, f_size).contiguous()
        # (Seq * batch * head, Feature)
        # (Seq, Feature)

        out = torch.zeros_like(t_copy, device="cuda")

        def grid(meta):
            return (s_size * b_size * h_size,)

        BLOCK_SIZE = triton.next_power_of_2(f_size)
        hb_size = h_size * b_size

        rope_fw[grid](
            t_copy,
            freqs_copy,
            out,
            f_size,
            hb_size,
            BLOCK_SIZE,
        )
        out = out.reshape(s_size, b_size, h_size, f_size).contiguous()

        return out

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        freqs = ctx.saved_tensors[1]

        s_size, b_size, h_size, f_size = grad_output.shape

        grad_output_copy = grad_output.reshape(
            s_size * b_size * h_size, f_size
        ).contiguous()
        freqs_copy = freqs.reshape(s_size, f_size).contiguous()
        # (Seq * batch * head, Feature)
        # (Seq, Feature)

        out = torch.zeros_like(grad_output_copy, device="cuda")

        def grid(meta):
            return (s_size * b_size * h_size,)

        BLOCK_SIZE = triton.next_power_of_2(f_size)
        hb_size = h_size * b_size

        rope_bw[grid](
            grad_output_copy,
            freqs_copy,
            out,
            f_size,
            hb_size,
            BLOCK_SIZE,
        )

        out = out.reshape(s_size, b_size, h_size, f_size).contiguous()

        return out, None, None


triton_rotary_pos_emb = _rotary_pos_emb.apply
