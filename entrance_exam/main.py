import numpy as np
import torch
import transformer_engine.pytorch as te
import triton
import triton.language as tl


@triton.jit
def rope_fw(
    t_ptr,
    freqs_ptr,
    out_ptr,
    s_size: tl.constexpr,
    f_size: tl.constexpr,
    b_size: tl.constexpr,
    h_size: tl.constexpr,
    s_block_size: tl.constexpr,
    f_block_size: tl.constexpr,
):
    pid = tl.program_id(0)
    num_f_blocks = tl.cdiv(f_size, f_block_size)
    s_block = pid // num_f_blocks
    f_block = pid % num_f_blocks

    s_stride = b_size * h_size * f_size
    b_stride = h_size * f_size
    h_stride = f_size
    f_stride = 1

    s_offset = s_block * s_block_size
    f_offset = f_block * f_block_size

    t_block_ptr = tl.make_block_ptr(
        t_ptr,
        shape=(s_size, b_size, h_size, f_size),
        strides=(s_stride, b_stride, h_stride, f_stride),
        offsets=(s_offset, 0, 0, f_offset),
        block_shape=(s_block_size, b_size, h_size, f_block_size),
        order=(3, 2, 1, 0),
    )

    freqs_block_ptr = tl.make_block_ptr(
        freqs_ptr,
        shape=(s_size, 1, 1, f_size),
        strides=(s_stride, 0, 0, f_stride),
        offsets=(s_offset, 0, 0, f_offset),
        block_shape=(s_block_size, 1, 1, f_block_size),
        order=(3, 2, 1, 0),
    )

    t = tl.load(t_block_ptr, boundary_check=(0, 3))
    freqs = tl.load(freqs_block_ptr, boundary_check=(0, 3))
    out = tl.zeros_like(t)
    rotate_freqs = tl.reshape(freqs, (s_size, 1, 1, 2, f_size // 2))
    tl.tensor
    tl.swizzle2d
    tl.trans(rotate_freqs, (2, 1))
    # rotate_freqs_1 = rotate_freqs[:1]
    # rotate_freqs_2 = rotate_freqs[:, :, :, 1, :]

    # out = t * freqs

    # tl.store(out_ptr, out)

    pass


class _rotary_pos_emb(torch.autograd.Function):
    @staticmethod
    def forward(ctx: torch.Any, t, freqs, *args: torch.Any, **kwargs: torch.Any) -> torch.Any:
        # (Seq, batch, head, Feature)
        # (Seq, 1, 1, Feature)
        t, freqs = args
        ctx.save_for_backward(t, freqs)
        s_size, b_size, h_size, f_size = t.shape
        out = torch.zeros_like(t, device="cuda")

        def grid(meta):
            return (
                triton.cdiv(s_size, meta["s_block_size"])
                * triton.cdiv(f_size, meta["f_block_size"]),
            )

        S_BLOCK_SIZE = 16
        F_BLOCK_SIZE = 16

        rope_fw[grid](
            t,
            freqs,
            out,
            s_size,
            f_size,
            b_size,
            h_size,
            S_BLOCK_SIZE,
            F_BLOCK_SIZE,
        )

        return out

    @staticmethod
    def backward(ctx: torch.Any, *grad_outputs: torch.Any) -> torch.Any:
        pass


triton_rotary_pos_emb = _rotary_pos_emb.apply

####################
### Test Section ###
####################


def test_value():
    SEQ_LEN = 1024
    BATCH_SIZE = 32
    NUM_HEAD = 8
    FEATURE_DIM = 256

    sample_input = torch.randn(
        (SEQ_LEN, BATCH_SIZE, NUM_HEAD, FEATURE_DIM),
        requires_grad=True,
        device="cuda",
    )  # (S, B, H, D)
    freq_np = np.random.randn(SEQ_LEN, 1, 1, FEATURE_DIM)
    for m in range(SEQ_LEN):
        for d in range(FEATURE_DIM):
            freq_np[m, 0, 0, d] = (m + 1) * (10000 ** (-2 * (d // 2) / FEATURE_DIM))
            # print("m:", m+1, " i:", d//2)
    sample_freq = torch.tensor(freq_np, device="cuda")

    # 1. PyTorch
    sample_output = te.attention.apply_rotary_pos_emb(sample_input, sample_freq)
    result = sample_output.mean()
    result.backward()

    # 2. Triton
    triton_sample_output = triton_rotary_pos_emb(sample_input, sample_freq)

    exit()


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["seq_len"],
        x_vals=[2**i for i in range(6, 11)],
        line_arg="provider",
        line_vals=["transformer_engine", "triton"],
        line_names=["transformer_engine", "triton"],
        styles=[("green", "-"), ("blue", "-")],
        ylabel="runtime(ms)",  # label name for the y-axis
        plot_name="group-gemm-performance",
        args={},
    )
)
def benchmark(seq_len, provider):
    SEQ_LEN = seq_len
    BATCH_SIZE = 32
    NUM_HEAD = 8
    FEATURE_DIM = 256

    sample_input = torch.randn(
        (SEQ_LEN, BATCH_SIZE, NUM_HEAD, FEATURE_DIM),
        requires_grad=True,
        device="cuda",
    )  # (S, B, H, D)
    freq_np = np.random.randn(SEQ_LEN, 1, 1, FEATURE_DIM)
    for m in range(SEQ_LEN):
        for d in range(FEATURE_DIM):
            freq_np[m, 0, 0, d] = (m + 1) * (10000 ** (-2 * (d // 2) / FEATURE_DIM))
            # print("m:", m+1, " i:", d//2)
    sample_freq = torch.tensor(freq_np, device="cuda")

    quantiles = [0.5, 0.2, 0.8]
    if provider == "transformer_engine":
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: te.attention.apply_rotary_pos_emb(sample_input, sample_freq),
            quantiles=quantiles,
        )
    if provider == "triton":
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: te.attention.apply_rotary_pos_emb(sample_input, sample_freq),
            quantiles=quantiles,
        )

    return ms, max_ms, min_ms


test_value()
benchmark.run(print_data=True)
