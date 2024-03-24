from triton_RoPE import triton_rotary_pos_emb

import numpy as np
import torch
import transformer_engine.pytorch as te
import triton

import torch
import random
import torch.backends.cudnn as cudnn

####################
### Test Section ###
####################

torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.cuda.manual_seed_all(0)
np.random.seed(0)
cudnn.benchmark = False
cudnn.deterministic = True
random.seed(1995)


########################################
### 1. Forward & Backward Value Test ###
########################################
def test_value():
    SEQ_LEN = 1024
    BATCH_SIZE = 32
    NUM_HEAD = 8
    FEATURE_DIM = 256

    sample_input_1 = torch.randn(
        (SEQ_LEN, BATCH_SIZE, NUM_HEAD, FEATURE_DIM),
        requires_grad=True,
        device="cuda",
    )  # (S, B, H, D)
    sample_input_2 = sample_input_1.clone().detach().to("cuda")
    sample_input_2.requires_grad = True
    create_pos_emb = te.attention.RotaryPositionEmbedding(FEATURE_DIM)
    sample_freq = create_pos_emb(SEQ_LEN).to("cuda")

    # 1. PyTorch
    sample_output = te.attention.apply_rotary_pos_emb(
        sample_input_1, sample_freq, tensor_format="sbhd"
    )
    result = sample_output.mean()
    result.backward()

    # 2. Triton
    triton_sample_output = triton_rotary_pos_emb(sample_input_2, sample_freq)
    triton_result = triton_sample_output.mean()
    triton_result.backward()

    assert torch.allclose(sample_output, triton_sample_output, atol=1e-5)
    assert torch.allclose(sample_input_1.grad, sample_input_2.grad, atol=1e-5)


#####################################################################
### 2. Forward Perf. Comparison b/w transformer_engine and triton ###
#####################################################################
@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["seq_len"],
        x_vals=[2**i for i in range(6, 20)],
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
    BATCH_SIZE = 1
    NUM_HEAD = 1
    FEATURE_DIM = 1024

    sample_input = torch.randn(
        (SEQ_LEN, BATCH_SIZE, NUM_HEAD, FEATURE_DIM),
        requires_grad=True,
        device="cuda",
    )  # (S, B, H, D)
    create_pos_emb = te.attention.RotaryPositionEmbedding(FEATURE_DIM)
    sample_freq = create_pos_emb(SEQ_LEN).to("cuda")

    quantiles = [0.5, 0.2, 0.8]
    if provider == "transformer_engine":
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: te.attention.apply_rotary_pos_emb(sample_input, sample_freq),
            quantiles=quantiles,
        )
    if provider == "triton":
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: triton_rotary_pos_emb(sample_input, sample_freq),
            quantiles=quantiles,
        )

    return ms, max_ms, min_ms


test_value()
benchmark.run(print_data=True)
