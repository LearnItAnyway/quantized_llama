import math
import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import custom_bwd, custom_fwd

#https://github.com/fpgaminer/GPTQ-triton
"""
Mostly the same as the autotuner in Triton, but with a few changes like using 40 runs instead of 100.
"""


def matmul248_kernel_config_pruner(configs, nargs):
    """
    The main purpose of this function is to shrink BLOCK_SIZE_* when the corresponding dimension is smaller.
    """
    m = max(2**int(math.ceil(math.log2(nargs['M']))), 16)
    n = max(2**int(math.ceil(math.log2(nargs['N']))), 16)
    k = max(2**int(math.ceil(math.log2(nargs['K']))), 16)

    used = set()
    for config in configs:
        block_size_m = min(m, config.kwargs['BLOCK_SIZE_M'])
        block_size_n = min(n, config.kwargs['BLOCK_SIZE_N'])
        block_size_k = min(k, config.kwargs['BLOCK_SIZE_K'])
        group_size_m = config.kwargs['GROUP_SIZE_M']

        if (block_size_m, block_size_n, block_size_k, group_size_m, config.num_stages, config.num_warps) in used:
            continue

        used.add((block_size_m, block_size_n, block_size_k, group_size_m, config.num_stages, config.num_warps))
        yield triton.Config({
            'BLOCK_SIZE_M': block_size_m,
            'BLOCK_SIZE_N': block_size_n,
            'BLOCK_SIZE_K': block_size_k,
            'GROUP_SIZE_M': group_size_m
        },
                            num_stages=config.num_stages,
                            num_warps=config.num_warps)

try:
    import triton
    import triton.language as tl
    import custom_autotune

    # code based https://github.com/fpgaminer/GPTQ-triton
    @custom_autotune.autotune(
        configs=[
            triton.Config({
                'BLOCK_SIZE_M': 64,
                'BLOCK_SIZE_N': 256,
                'BLOCK_SIZE_K': 32,
                'GROUP_SIZE_M': 8
            }, num_stages=4, num_warps=4),
            triton.Config({
                'BLOCK_SIZE_M': 128,
                'BLOCK_SIZE_N': 128,
                'BLOCK_SIZE_K': 32,
                'GROUP_SIZE_M': 8
            }, num_stages=4, num_warps=4),
            triton.Config({
                'BLOCK_SIZE_M': 64,
                'BLOCK_SIZE_N': 128,
                'BLOCK_SIZE_K': 32,
                'GROUP_SIZE_M': 8
            }, num_stages=4, num_warps=4),
            triton.Config({
                'BLOCK_SIZE_M': 128,
                'BLOCK_SIZE_N': 32,
                'BLOCK_SIZE_K': 32,
                'GROUP_SIZE_M': 8
            }, num_stages=4, num_warps=4),
            triton.Config({
                'BLOCK_SIZE_M': 64,
                'BLOCK_SIZE_N': 64,
                'BLOCK_SIZE_K': 32,
                'GROUP_SIZE_M': 8
            }, num_stages=4, num_warps=4),
            triton.Config({
                'BLOCK_SIZE_M': 64,
                'BLOCK_SIZE_N': 128,
                'BLOCK_SIZE_K': 32,
                'GROUP_SIZE_M': 8
            }, num_stages=2, num_warps=8),
            triton.Config({
                'BLOCK_SIZE_M': 64,
                'BLOCK_SIZE_N': 64,
                'BLOCK_SIZE_K': 64,
                'GROUP_SIZE_M': 8
            }, num_stages=3, num_warps=8),
            triton.Config({
                'BLOCK_SIZE_M': 32,
                'BLOCK_SIZE_N': 32,
                'BLOCK_SIZE_K': 128,
                'GROUP_SIZE_M': 8
            }, num_stages=2, num_warps=4),
        ],
        key=['M', 'N', 'K'],
        nearest_power_of_two=True,
        prune_configs_by={
            'early_config_prune': custom_autotune.matmul248_kernel_config_pruner,
            'perf_model': None,
            'top_k': None,
        },
    )
    @triton.jit
    def matmul_248_kernel(a_ptr, b_ptr, c_ptr, scales_ptr, zeros_ptr, g_ptr, M, N, K, bits, maxq, stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn, stride_scales, stride_zeros,
                          BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr, GROUP_SIZE_M: tl.constexpr):
        """
        Compute the matrix multiplication C = A x B.
        A is of shape (M, K) float16
        B is of shape (K//8, N) int32
        C is of shape (M, N) float16
        scales is of shape (G, N) float16
        zeros is of shape (G, N) float16
        g_ptr is of shape (K) int32
        """
        infearure_per_bits = 32 // bits

        pid = tl.program_id(axis=0)
        num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
        num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
        num_pid_k = tl.cdiv(K, BLOCK_SIZE_K)
        num_pid_in_group = GROUP_SIZE_M * num_pid_n
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
        pid_m = first_pid_m + (pid % group_size_m)
        pid_n = (pid % num_pid_in_group) // group_size_m

        offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        offs_k = tl.arange(0, BLOCK_SIZE_K)
        a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)  # (BLOCK_SIZE_M, BLOCK_SIZE_K)
        a_mask = (offs_am[:, None] < M)
        # b_ptrs is set up such that it repeats elements along the K axis 8 times
        b_ptrs = b_ptr + ((offs_k[:, None] // infearure_per_bits) * stride_bk + offs_bn[None, :] * stride_bn)  # (BLOCK_SIZE_K, BLOCK_SIZE_N)
        g_ptrs = g_ptr + offs_k
        # shifter is used to extract the N bits of each element in the 32-bit word from B
        scales_ptrs = scales_ptr + offs_bn[None, :]
        zeros_ptrs = zeros_ptr + (offs_bn[None, :] // infearure_per_bits)

        shifter = (offs_k % infearure_per_bits) * bits
        zeros_shifter = (offs_bn % infearure_per_bits) * bits
        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

        for k in range(0, num_pid_k):
            g_idx = tl.load(g_ptrs)

            # Fetch scales and zeros; these are per-outfeature and thus reused in the inner loop
            scales = tl.load(scales_ptrs + g_idx[:, None] * stride_scales)  # (BLOCK_SIZE_K, BLOCK_SIZE_N,)
            zeros = tl.load(zeros_ptrs + g_idx[:, None] * stride_zeros)  # (BLOCK_SIZE_K, BLOCK_SIZE_N,)

            zeros = (zeros >> zeros_shifter[None, :]) & maxq
            zeros = (zeros + 1)

            a = tl.load(a_ptrs, mask=a_mask, other=0.)  # (BLOCK_SIZE_M, BLOCK_SIZE_K)
            b = tl.load(b_ptrs)  # (BLOCK_SIZE_K, BLOCK_SIZE_N), but repeated

            # Now we need to unpack b (which is N-bit values) into 32-bit values
            b = (b >> shifter[:, None]) & maxq  # Extract the N-bit values
            b = (b - zeros) * scales  # Scale and shift

            accumulator += tl.dot(a, b)
            a_ptrs += BLOCK_SIZE_K
            b_ptrs += (BLOCK_SIZE_K // infearure_per_bits) * stride_bk
            g_ptrs += BLOCK_SIZE_K

        c_ptrs = c_ptr + stride_cm * offs_am[:, None] + stride_cn * offs_bn[None, :]
        c_mask = (offs_am[:, None] < M) & (offs_bn[None, :] < N)
        tl.store(c_ptrs, accumulator, mask=c_mask)

    @custom_autotune.autotune(configs=[
        triton.Config({
            'BLOCK_SIZE_M': 64,
            'BLOCK_SIZE_N': 32,
            'BLOCK_SIZE_K': 256,
            'GROUP_SIZE_M': 8
        }, num_stages=4, num_warps=4),
        triton.Config({
            'BLOCK_SIZE_M': 128,
            'BLOCK_SIZE_N': 32,
            'BLOCK_SIZE_K': 128,
            'GROUP_SIZE_M': 8
        }, num_stages=4, num_warps=4),
        triton.Config({
            'BLOCK_SIZE_M': 64,
            'BLOCK_SIZE_N': 32,
            'BLOCK_SIZE_K': 128,
            'GROUP_SIZE_M': 8
        }, num_stages=4, num_warps=4),
        triton.Config({
            'BLOCK_SIZE_M': 128,
            'BLOCK_SIZE_N': 32,
            'BLOCK_SIZE_K': 32,
            'GROUP_SIZE_M': 8
        }, num_stages=4, num_warps=4),
        triton.Config({
            'BLOCK_SIZE_M': 64,
            'BLOCK_SIZE_N': 32,
            'BLOCK_SIZE_K': 64,
            'GROUP_SIZE_M': 8
        }, num_stages=4, num_warps=4),
        triton.Config({
            'BLOCK_SIZE_M': 64,
            'BLOCK_SIZE_N': 32,
            'BLOCK_SIZE_K': 128,
            'GROUP_SIZE_M': 8
        }, num_stages=2, num_warps=8),
        triton.Config({
            'BLOCK_SIZE_M': 64,
            'BLOCK_SIZE_N': 64,
            'BLOCK_SIZE_K': 64,
            'GROUP_SIZE_M': 8
        }, num_stages=3, num_warps=8),
        triton.Config({
            'BLOCK_SIZE_M': 32,
            'BLOCK_SIZE_N': 128,
            'BLOCK_SIZE_K': 32,
            'GROUP_SIZE_M': 8
        }, num_stages=2, num_warps=4),
    ],
                              key=['M', 'N', 'K'],
                              nearest_power_of_two=True)
    @triton.jit
    def transpose_matmul_248_kernel(a_ptr, b_ptr, c_ptr, scales_ptr, zeros_ptr, g_ptr, M, N, K, bits, maxq, stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn, stride_scales,
                                    stride_zeros, BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr, GROUP_SIZE_M: tl.constexpr):
        """
        Compute the matrix multiplication C = A x B.
        A is of shape (M, N) float16
        B is of shape (K//8, N) int32
        C is of shape (M, K) float16
        scales is of shape (G, N) float16
        zeros is of shape (G, N) float16
        g_ptr is of shape (K) int32
        """
        infearure_per_bits = 32 // bits

        pid = tl.program_id(axis=0)
        num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
        num_pid_k = tl.cdiv(K, BLOCK_SIZE_K)
        num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
        num_pid_in_group = GROUP_SIZE_M * num_pid_k
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
        pid_m = first_pid_m + (pid % group_size_m)
        pid_k = (pid % num_pid_in_group) // group_size_m

        offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        offs_bk = pid_k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
        offs_n = tl.arange(0, BLOCK_SIZE_N)
        a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_n[None, :] * stride_ak)  # (BLOCK_SIZE_M, BLOCK_SIZE_N)
        a_mask = (offs_am[:, None] < M)
        # b_ptrs is set up such that it repeats elements along the K axis 8 times
        b_ptrs = b_ptr + ((offs_bk[:, None] // infearure_per_bits) * stride_bk + offs_n[None, :] * stride_bn)  # (BLOCK_SIZE_K, BLOCK_SIZE_N)
        g_ptrs = g_ptr + offs_bk
        g_idx = tl.load(g_ptrs)

        # shifter is used to extract the N bits of each element in the 32-bit word from B
        scales_ptrs = scales_ptr + offs_n[None, :] + g_idx[:, None] * stride_scales
        zeros_ptrs = zeros_ptr + (offs_n[None, :] // infearure_per_bits) + g_idx[:, None] * stride_zeros

        shifter = (offs_bk % infearure_per_bits) * bits
        zeros_shifter = (offs_n % infearure_per_bits) * bits
        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_K), dtype=tl.float32)

        for n in range(0, num_pid_n):
            # Fetch scales and zeros; these are per-outfeature and thus reused in the inner loop
            scales = tl.load(scales_ptrs)  # (BLOCK_SIZE_K, BLOCK_SIZE_N,)
            zeros = tl.load(zeros_ptrs)  # (BLOCK_SIZE_K, BLOCK_SIZE_N,)

            zeros = (zeros >> zeros_shifter[None, :]) & maxq
            zeros = (zeros + 1)

            a = tl.load(a_ptrs, mask=a_mask, other=0.)  # (BLOCK_SIZE_M, BLOCK_SIZE_N)
            b = tl.load(b_ptrs)  # (BLOCK_SIZE_K, BLOCK_SIZE_N), but repeated

            # Now we need to unpack b (which is N-bit values) into 32-bit values
            b = (b >> shifter[:, None]) & maxq  # Extract the N-bit values
            b = (b - zeros) * scales  # Scale and shift
            b = tl.trans(b)

            accumulator += tl.dot(a, b)
            a_ptrs += BLOCK_SIZE_N
            b_ptrs += BLOCK_SIZE_N
            scales_ptrs += BLOCK_SIZE_N
            zeros_ptrs += (BLOCK_SIZE_N // infearure_per_bits)

        c_ptrs = c_ptr + stride_cm * offs_am[:, None] + stride_cn * offs_bk[None, :]
        c_mask = (offs_am[:, None] < M) & (offs_bk[None, :] < K)
        tl.store(c_ptrs, accumulator, mask=c_mask)
except:
    print('trioton not installed.')


def matmul248(input, qweight, scales, qzeros, g_idx, bits, maxq):
    with torch.cuda.device(input.device):
        output = torch.empty((input.shape[0], qweight.shape[1]), device=input.device, dtype=torch.float16)
        grid = lambda META: (triton.cdiv(input.shape[0], META['BLOCK_SIZE_M']) * triton.cdiv(qweight.shape[1], META['BLOCK_SIZE_N']), )
        matmul_248_kernel[grid](input, qweight, output, scales, qzeros, g_idx, input.shape[0], qweight.shape[1], input.shape[1], bits, maxq, input.stride(0), input.stride(1), qweight.stride(0),
                                qweight.stride(1), output.stride(0), output.stride(1), scales.stride(0), qzeros.stride(0))
        return output


def transpose_matmul248(input, qweight, scales, qzeros, g_idx, bits, maxq):
    with torch.cuda.device(input.device):
        output_dim = (qweight.shape[0] * 32) // bits
        output = torch.empty((input.shape[0], output_dim), device=input.device, dtype=torch.float16)
        grid = lambda META: (triton.cdiv(input.shape[0], META['BLOCK_SIZE_M']) * triton.cdiv(output_dim, META['BLOCK_SIZE_K']), )
        transpose_matmul_248_kernel[grid](input, qweight, output, scales, qzeros, g_idx, input.shape[0], qweight.shape[1], output_dim, bits, maxq, input.stride(0), input.stride(1), qweight.stride(0),
                                          qweight.stride(1), output.stride(0), output.stride(1), scales.stride(0), qzeros.stride(0))
        return output


class QuantLinearFunction(torch.autograd.Function):

    @staticmethod
    @custom_fwd(cast_inputs=torch.float16)
    def forward(ctx, input, qweight, scales, qzeros, g_idx, bits, maxq):
        output = matmul248(input, qweight, scales, qzeros, g_idx, bits, maxq)
        ctx.save_for_backward(qweight, scales, qzeros, g_idx)
        ctx.bits, ctx.maxq = bits, maxq
        return output

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_output):
        qweight, scales, qzeros, g_idx = ctx.saved_tensors
        bits, maxq = ctx.bits, ctx.maxq
        grad_input = None

        if ctx.needs_input_grad[0]:
            grad_input = transpose_matmul248(grad_output, qweight, scales, qzeros, g_idx, bits, maxq)
        return grad_input, None, None, None, None, None, None


class QuantLinear(nn.Module):

    def __init__(self, bits, groupsize, infeatures, outfeatures, bias):
        super().__init__()
        if bits not in [2, 4, 8]:
            raise NotImplementedError("Only 2,4,8 bits are supported.")
        self.infeatures = infeatures
        self.outfeatures = outfeatures
        self.bits = bits
        self.maxq = 2**self.bits - 1
        self.groupsize = groupsize if groupsize != -1 else infeatures

        self.register_buffer('qweight', torch.zeros((infeatures // 32 * self.bits, outfeatures), dtype=torch.int32))
        self.register_buffer('qzeros', torch.zeros((math.ceil(infeatures / self.groupsize), outfeatures // 32 * self.bits), dtype=torch.int32))
        self.register_buffer('scales', torch.zeros((math.ceil(infeatures / self.groupsize), outfeatures), dtype=torch.float16))
        self.register_buffer('g_idx', torch.tensor([i // self.groupsize for i in range(infeatures)], dtype=torch.int32))
        if bias:
            self.register_buffer('bias', torch.zeros((outfeatures), dtype=torch.float16))
        else:
            self.bias = None

    def pack(self, linear, scales, zeros, g_idx=None):
        self.g_idx = g_idx.clone() if g_idx is not None else self.g_idx

        scales = scales.t().contiguous()
        zeros = zeros.t().contiguous()
        scale_zeros = zeros * scales
        self.scales = scales.clone().half()
        if linear.bias is not None:
            self.bias = linear.bias.clone().half()

        intweight = []
        for idx in range(self.infeatures):
            intweight.append(torch.round((linear.weight.data[:, idx] + scale_zeros[self.g_idx[idx]]) / self.scales[self.g_idx[idx]]).to(torch.int)[:, None])
        intweight = torch.cat(intweight, dim=1)
        intweight = intweight.t().contiguous()
        intweight = intweight.numpy().astype(np.uint32)
        qweight = np.zeros((intweight.shape[0] // 32 * self.bits, intweight.shape[1]), dtype=np.uint32)
        i = 0
        row = 0
        while row < qweight.shape[0]:
            if self.bits in [2, 4, 8]:
                for j in range(i, i + (32 // self.bits)):
                    qweight[row] |= intweight[j] << (self.bits * (j - i))
                i += 32 // self.bits
                row += 1
            else:
                raise NotImplementedError("Only 2,4,8 bits are supported.")

        qweight = qweight.astype(np.int32)
        self.qweight = torch.from_numpy(qweight)

        zeros -= 1
        zeros = zeros.numpy().astype(np.uint32)
        qzeros = np.zeros((zeros.shape[0], zeros.shape[1] // 32 * self.bits), dtype=np.uint32)
        i = 0
        col = 0
        while col < qzeros.shape[1]:
            if self.bits in [2, 4, 8]:
                for j in range(i, i + (32 // self.bits)):
                    qzeros[:, col] |= zeros[:, j] << (self.bits * (j - i))
                i += 32 // self.bits
                col += 1
            else:
                raise NotImplementedError("Only 2,4,8 bits are supported.")

        qzeros = qzeros.astype(np.int32)
        self.qzeros = torch.from_numpy(qzeros)

    def forward(self, x):
        out_shape = x.shape[:-1] + (self.outfeatures, )
        out = QuantLinearFunction.apply(x.reshape(-1, x.shape[-1]), self.qweight, self.scales, self.qzeros, self.g_idx, self.bits, self.maxq)
        out = out + self.bias if self.bias is not None else out
        return out.reshape(out_shape)





import math
import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import custom_bwd, custom_fwd



from torch.cuda.amp import custom_bwd, custom_fwd
from transformers.models.llama.modeling_llama import LlamaMLP

class QuantLlamaMLP(nn.Module):

    def __init__(
        self,
        gate_proj,
        down_proj,
        up_proj,
    ):
        super().__init__()
        self.register_buffer('gate_proj_qweight', gate_proj.qweight)
        self.register_buffer('gate_proj_scales', gate_proj.scales)
        self.register_buffer('gate_proj_qzeros', gate_proj.qzeros)
        self.register_buffer('gate_proj_g_idx', gate_proj.g_idx)
        self.register_buffer('up_proj_qweight', up_proj.qweight)
        self.register_buffer('up_proj_scales', up_proj.scales)
        self.register_buffer('up_proj_qzeros', up_proj.qzeros)
        self.register_buffer('up_proj_g_idx', up_proj.g_idx)

        self.infeatures = gate_proj.infeatures
        self.intermediate_size = gate_proj.outfeatures
        self.outfeatures = down_proj.outfeatures
        self.bits = gate_proj.bits
        self.maxq = gate_proj.maxq

        self.down_proj = down_proj

    def forward(self, x):
        return self.down_proj(self.triton_llama_mlp(x))

    def triton_llama_mlp(self, x):
        with torch.cuda.device(x.device):
            out_shape = x.shape[:-1] + (self.intermediate_size, )
            x = x.reshape(-1, x.shape[-1])
            M, K = x.shape
            N = self.intermediate_size
            c = torch.empty((M, N), device=x.device, dtype=torch.float16)
            grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']), )
            fusedmatmul_248_kernel[grid](x, c, self.gate_proj_qweight, self.gate_proj_scales, self.gate_proj_qzeros, self.gate_proj_g_idx, self.up_proj_qweight, self.up_proj_scales,
                                         self.up_proj_qzeros, self.up_proj_g_idx, M, N, K, self.bits, self.maxq, x.stride(0), x.stride(1), self.gate_proj_qweight.stride(0),
                                         self.gate_proj_qweight.stride(1), c.stride(0), c.stride(1), self.gate_proj_scales.stride(0), self.gate_proj_qzeros.stride(0))
            c = c.reshape(out_shape)
            return c

    def fused2cuda(self):
        self.gate_proj_qweight = self.gate_proj_qweight.cuda()
        self.gate_proj_scales = self.gate_proj_scales.cuda()
        self.gate_proj_qzeros = self.gate_proj_qzeros.cuda()
        self.gate_proj_g_idx = self.gate_proj_g_idx.cuda()
        self.up_proj_qweight = self.up_proj_qweight.cuda()
        self.up_proj_scales = self.up_proj_scales.cuda()
        self.up_proj_qzeros = self.up_proj_qzeros.cuda()
        self.up_proj_g_idx = self.up_proj_g_idx.cuda()

    def fused2cpu(self):
        self.gate_proj_qweight = self.gate_proj_qweight.cpu()
        self.gate_proj_scales = self.gate_proj_scales.cpu()
        self.gate_proj_qzeros = self.gate_proj_qzeros.cpu()
        self.gate_proj_g_idx = self.gate_proj_g_idx.cpu()
        self.up_proj_qweight = self.up_proj_qweight.cpu()
        self.up_proj_scales = self.up_proj_scales.cpu()
        self.up_proj_qzeros = self.up_proj_qzeros.cpu()
        self.up_proj_g_idx = self.up_proj_g_idx.cpu()


def make_fused_mlp(m, parent_name=''):
    """
    Replace all LlamaMLP modules with QuantLlamaMLP modules, which fuses many of the operations.
    """
    if isinstance(m, LlamaMLP):
        return QuantLlamaMLP(m.gate_proj, m.down_proj, m.up_proj)

    for name, child in m.named_children():
        child = make_fused_mlp(child, parent_name=f"{parent_name}.{name}")

        if isinstance(child, QuantLlamaMLP):
            setattr(m, name, child)
    return m


def autotune_warmup_fused(model):
    """
    Pre-tunes the quantized kernel
    """
    from tqdm import tqdm

    kn_values = {}

    for _, m in model.named_modules():
        if not isinstance(m, QuantLlamaMLP):
            continue

        k = m.infeatures
        n = m.intermediate_size

        m.fused2cuda()
        if (k, n) not in kn_values:
            kn_values[(k, n)] = m

    print(f'Found {len(kn_values)} unique fused mlp KN values.')

    print('Warming up autotune cache ...')
    with torch.no_grad():
        for m in tqdm(range(0, 12)):
            m = 2**m  # [1, 2048]
            for (k, n), (modules) in kn_values.items():
                a = torch.randn(m, k, dtype=torch.float16, device='cuda')
                modules.triton_llama_mlp(a)

        for (k, n), (modules) in kn_values.items():
            a = torch.randn(m, k, dtype=torch.float16, device='cuda')
            modules.fused2cpu()
    del kn_values

from torch.nn import functional as F
from transformers.models.llama.modeling_llama import LlamaAttention
import triton
import triton.language as tl


@triton.jit
def rotate_half_kernel(
        qk_seq_ptr,
        position_ids_ptr,
        qk_seq_stride,
        position_ids_batch_stride,
        seq_len,
        HEAD_DIM: tl.constexpr,
        BLOCK_HEIGHT: tl.constexpr,
        BLOCK_WIDTH: tl.constexpr,
        INV_BASE: tl.constexpr
):
    # qk_seq_ptr: (bsz, seq_len, 2, num_heads, head_dim) -- OK to be discontinuous in 2nd dimension.
    # position ids: (bsz, seq_len) -- must be contiguous in the last dimension.

    HALF_HEAD: tl.constexpr = HEAD_DIM // 2
    STEPS_PER_ROW: tl.constexpr = HALF_HEAD // BLOCK_WIDTH

    batch_seq = tl.program_id(axis=0)
    row_blk_x_col_blk = tl.program_id(axis=1)

    row_blk = row_blk_x_col_blk // STEPS_PER_ROW
    row = row_blk * BLOCK_HEIGHT
    if BLOCK_WIDTH < HALF_HEAD:
        col_blk = row_blk_x_col_blk % STEPS_PER_ROW
        col = col_blk * BLOCK_WIDTH
    else:
        col: tl.constexpr = 0

    # A block will never cross a sequence boundary, which simplifies things a lot.
    batch = batch_seq // seq_len
    seq = batch_seq % seq_len
    position_id = tl.load(position_ids_ptr + batch * position_ids_batch_stride + seq)
    # As sometimes happens, just calculating this on the fly is faster than loading it from memory.
    # Use `tl.libdevice.exp` rather than `tl.exp` -- the latter is less accurate.
    freq = tl.libdevice.exp((col + tl.arange(0, BLOCK_WIDTH)).to(tl.float32) * INV_BASE) * position_id
    cos = tl.cos(freq).to(tl.float32)
    sin = tl.sin(freq).to(tl.float32)

    col_offsets: tl.constexpr = tl.arange(0, BLOCK_WIDTH)
    embed_offsets = (row * HEAD_DIM + col) + col_offsets
    x_ptrs = (qk_seq_ptr + batch_seq * qk_seq_stride) + embed_offsets

    for k in range(0, BLOCK_HEIGHT):
        x = tl.load(x_ptrs).to(tl.float32)
        y = tl.load(x_ptrs + HALF_HEAD).to(tl.float32)
        out_x = x * cos - y * sin
        tl.store(x_ptrs, out_x)
        out_y = x * sin + y * cos
        tl.store(x_ptrs + HALF_HEAD, out_y)
        x_ptrs += HEAD_DIM


def triton_rotate_half_(qk, position_ids, config=None):
    batch_size, seq_len, qandk, num_heads, head_dim = qk.shape

    # This default is the fastest for most job sizes, at least on my RTX 4090, and when it's not it's within spitting distance of the best option. There are some odd cases where having a block height of 2 or 4 helps but the difference is within 5%. It makes sense that this configuration is fast from a memory bandwidth and caching perspective.
    config = config or {'BLOCK_HEIGHT': 1, 'BLOCK_WIDTH': min(128, head_dim // 2), 'num_warps': 1}
    config['BLOCK_HEIGHT'] = min(config['BLOCK_HEIGHT'], 2 * num_heads)

    if position_ids.shape[0] == 1:
        position_ids = position_ids.repeat(batch_size, 1)
    assert qk.stride(3) == head_dim
    assert qk.stride(4) == 1
    assert position_ids.shape == (batch_size, seq_len)
    assert position_ids.stride(1) == 1, 'position_ids must be contiguous in the last dimension'
    assert (2 * num_heads) % config['BLOCK_HEIGHT'] == 0, f'number of rows not evenly divisible by {config["BLOCK_HEIGHT"]}'
    assert (head_dim // 2) % config['BLOCK_WIDTH'] == 0, f'number of columns ({head_dim // 2}) not evenly divisible by {config["BLOCK_WIDTH"]}'

    qk_by_seq = qk.view(batch_size * seq_len, 2 * num_heads * head_dim)
    grid = (qk_by_seq.shape[0], (2 * num_heads // config['BLOCK_HEIGHT']) * (head_dim // 2 // config['BLOCK_WIDTH']))

    # Must be the same as the theta of the frequencies used to train the model.
    BASE = 10000.0

    rotate_half_kernel[grid](
        qk_by_seq,
        position_ids,
        qk_by_seq.stride(0),
        position_ids.stride(0),
        seq_len,
        HEAD_DIM=head_dim,
        BLOCK_HEIGHT=config['BLOCK_HEIGHT'],
        BLOCK_WIDTH=config['BLOCK_WIDTH'],
        INV_BASE=-2.0 * math.log(BASE) / head_dim,
        num_warps=config['num_warps']
    )


class QuantLlamaAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(
        self,
        hidden_size,
        num_heads,
        qkv_proj,
        o_proj
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        if (self.head_dim * num_heads) != self.hidden_size:
            raise ValueError(f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                             f" and `num_heads`: {num_heads}).")
        self.qkv_proj = qkv_proj
        self.o_proj = o_proj

    def forward(self, hidden_states, past_key_value=None, attention_mask=None, position_ids=None, output_attentions=False, use_cache=False):
        """Input shape: Batch x Time x Channel"""

        bsz, q_len, _ = hidden_states.size()

        qkv_states = self.qkv_proj(hidden_states)
        qkv_states = qkv_states.view(bsz, q_len, 3, self.num_heads, self.head_dim)

        # This updates the query and key states in-place, saving VRAM.
        triton_rotate_half_(qkv_states[:, :, :2], position_ids)

        query_states, key_states, value_states = torch.split(qkv_states, 1, dim=2)
        del qkv_states
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)

        is_causal = past_key_value is None

        kv_seq_len = q_len
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]

        if past_key_value is not None:
            # reuse k, v, self_attention
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        if use_cache:
            # Since qkv_proj is fused, query_states etc will hold a reference to the original qkv_states tensor
            # which can cause excessive memory usage by the cache. `contiguous` is a convenient way to workaround this.
            key_states = key_states.contiguous()
            value_states = value_states.contiguous()
            query_states = query_states.contiguous()

        past_key_value = (key_states, value_states) if use_cache else None

        with torch.backends.cuda.sdp_kernel(enable_math=False):
            attn_output = F.scaled_dot_product_attention(query_states, key_states, value_states, is_causal=is_causal)
        del query_states, key_states, value_states

        attn_output = attn_output.transpose(1, 2).reshape(bsz, q_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)

        return attn_output, None, past_key_value


def make_quant_attn(model):
    """
    Replace all LlamaAttention modules with QuantLlamaAttention modules, fusing the q, k, v projections.
    """

    for name, m in model.named_modules():
        if not isinstance(m, LlamaAttention):
            continue

        q_proj = m.q_proj
        k_proj = m.k_proj
        v_proj = m.v_proj

        qweights = torch.cat([q_proj.qweight, k_proj.qweight, v_proj.qweight], dim=1)
        qzeros = torch.cat([q_proj.qzeros, k_proj.qzeros, v_proj.qzeros], dim=1)
        scales = torch.cat([q_proj.scales, k_proj.scales, v_proj.scales], dim=1)
        g_idx = torch.cat([q_proj.g_idx, k_proj.g_idx, v_proj.g_idx], dim=0)
        bias = torch.cat([q_proj.bias, k_proj.bias, v_proj.bias], dim=0) if q_proj.bias is not None else None

        qkv_layer = QuantLinear(q_proj.bits, q_proj.groupsize, q_proj.infeatures, q_proj.outfeatures + k_proj.outfeatures + v_proj.outfeatures, True if q_proj.bias is not None else False)
        qkv_layer.qweight = qweights
        qkv_layer.qzeros = qzeros
        qkv_layer.scales = scales
        qkv_layer.g_idx = g_idx
        qkv_layer.bias = bias
        # We're dropping the rotary embedding layer m.rotary_emb here. We don't need it in the triton branch.

        attn = QuantLlamaAttention(m.hidden_size, m.num_heads, qkv_layer, m.o_proj)

        if '.' in name:
            parent_name = name.rsplit('.', 1)[0]
            child_name = name[len(parent_name) + 1:]
            parent = model.get_submodule(parent_name)
        else:
            parent_name = ''
            parent = model
            child_name = name

        #print(f"Replacing {name} with quant_attn; parent: {parent_name}, child's name: {child_name}")

        setattr(parent, child_name, attn)

def autotune_warmup_linear(model, transpose=False):
    """
    Pre-tunes the quantized kernel
    """
    from tqdm import tqdm

    kn_values = {}

    for _, m in model.named_modules():
        if not isinstance(m, QuantLinear):
            continue

        k = m.infeatures
        n = m.outfeatures

        if (k, n) not in kn_values:
            kn_values[(k, n)] = (m.qweight.cuda(), m.scales.cuda(), m.qzeros.cuda(), m.g_idx.cuda(), m.bits, m.maxq)

    print(f'Found {len(kn_values)} unique KN Linear values.')

    print('Warming up autotune cache ...')
    with torch.no_grad():
        for m in tqdm(range(0, 12)):
            m = 2**m  # [1, 2048]
            for (k, n), (qweight, scales, qzeros, g_idx, bits, maxq) in kn_values.items():
                a = torch.randn(m, k, dtype=torch.float16, device='cuda')
                matmul248(a, qweight, scales, qzeros, g_idx, bits, maxq)
                if transpose:
                    a = torch.randn(m, n, dtype=torch.float16, device='cuda')
                    transpose_matmul248(a, qweight, scales, qzeros, g_idx, bits, maxq)
    del kn_values

import torch
from torch import nn
import triton
import triton.language as tl
from transformers.models.llama.modeling_llama import LlamaRMSNorm

@triton.jit
def rms_norm_fwd_fused(
    X,  # pointer to the input
    Y,  # pointer to the output
    W,  # pointer to the weights
    stride,  # how much to increase the pointer when moving by 1 row
    N,  # number of columns in X
    eps,  # epsilon to avoid division by zero
    BLOCK_SIZE: tl.constexpr,
):
    # Map the program id to the row of X and Y it should compute.
    row = tl.program_id(0)
    Y += row * stride
    X += row * stride
    # Compute variance
    _var = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        x = tl.load(X + cols, mask=cols < N, other=0.).to(tl.float32)
        x = tl.where(cols < N, x, 0.)
        _var += x * x
    var = tl.sum(_var, axis=0) / N
    rstd = 1 / tl.sqrt(var + eps)
    # Normalize and apply linear transformation
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        w = tl.load(W + cols, mask=mask)
        x = tl.load(X + cols, mask=mask, other=0.).to(tl.float32)
        x_hat = x * rstd
        y = x_hat * w
        # Write output
        tl.store(Y + cols, y, mask=mask)

class TritonLlamaRMSNorm(nn.Module):
    def __init__(self, weight, eps=1e-6):
        """
        LlamaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = weight
        self.variance_epsilon = eps

    def forward(self, x):
        y = torch.empty_like(x)
        # reshape input data into 2D tensor
        x_arg = x.reshape(-1, x.shape[-1])
        M, N = x_arg.shape
        # Less than 64KB per feature: enqueue fused kernel
        MAX_FUSED_SIZE = 65536 // x.element_size()
        BLOCK_SIZE = min(MAX_FUSED_SIZE, triton.next_power_of_2(N))
        if N > BLOCK_SIZE:
            raise RuntimeError("This layer norm doesn't support feature dim >= 64KB.")
        # heuristics for number of warps
        num_warps = min(max(BLOCK_SIZE // 256, 1), 8)
        # enqueue kernel
        rms_norm_fwd_fused[(M,)](x_arg, y, self.weight,
                                 x_arg.stride(0), N, self.variance_epsilon,
                                 BLOCK_SIZE=BLOCK_SIZE, num_warps=num_warps)
        return y


def make_quant_norm(model):
    """
    Replace all LlamaRMSNorm modules with TritonLlamaRMSNorm modules
    """

    for name, m in model.named_modules():
        if not isinstance(m, LlamaRMSNorm):
            continue

        norm = TritonLlamaRMSNorm(m.weight, m.variance_epsilon)

        if '.' in name:
            parent_name = name.rsplit('.', 1)[0]
            child_name = name[len(parent_name) + 1:]
            parent = model.get_submodule(parent_name)
        else:
            parent_name = ''
            parent = model
            child_name = name

        #print(f"Replacing {name} with quant_attn; parent: {parent_name}, child's name: {child_name}")

        setattr(parent, child_name, norm)
