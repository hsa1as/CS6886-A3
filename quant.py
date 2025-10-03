import torch
import torch.nn as nn
import torch.nn.functional as F
import math
# for STE
class QuantizeSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, scale, zp, qmin, qmax):
        q = torch.round(x / scale + zp)
        q = torch.clamp(q, qmin, qmax)
        return (q - zp) * scale

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None, None, None

# template
def qparams_from_minmax(xmin, xmax, n_bits=8, unsigned=False, eps=1e-12):
    if unsigned:
        qmin, qmax = 0, (1 << n_bits) - 1
        xmin = torch.zeros_like(xmin)
        scale = (xmax - xmin).clamp_min(eps) / float(qmax - qmin)
        zp = torch.round(-xmin / scale).clamp(qmin, qmax)
    else:
        qmax = (1 << (n_bits - 1)) - 1
        qmin = -qmax
        max_abs = torch.max(xmin.abs(), xmax.abs()).clamp_min(eps)
        scale = max_abs / float(qmax)
        zp = torch.zeros_like(scale)
    return scale, zp, qmin, qmax

class QATActFakeQuant(nn.Module):
    def __init__(self, n_bits=8, unsigned=True):
        super().__init__()
        self.n_bits = n_bits
        self.unsigned = unsigned
        self.register_buffer("scale", torch.tensor(1.0))
        self.register_buffer("zp", torch.tensor(0.0))
        self.qmin, self.qmax = None, None

    def set_qparams(self, xmin, xmax):
        scale, zp, qmin, qmax = qparams_from_minmax(
            xmin, xmax, n_bits=self.n_bits, unsigned=self.unsigned
        )
        self.scale.copy_(scale)
        self.zp.copy_(zp)
        self.qmin, self.qmax = qmin, qmax

    def forward(self, x):
        self.last_shape = x.shape
        return QuantizeSTE.apply(x, self.scale, self.zp, self.qmin, self.qmax)
    
    def num_values(self):
        if self.last_shape is None:
            return None
        # product of all dimensions except batch
        return int(torch.tensor(self.last_shape[1:]).prod().item())

    # Returns activation sizes, not storage
    def storage_size_bytes(self):
        n = self.num_values()
        if n is None:
            act_storage = 0
        else:
            act_storage = (self.n_bits * n) // 8

        # metadata overhead:
        # scale (4), qmin (4), qmax (4)
        meta_storage = 3 * 4  

        return act_storage , meta_storage 

class QATConv2d(nn.Conv2d):
    def __init__(self, *args, weight_bits=8, per_channel=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.weight_bits = weight_bits
        self.per_channel = per_channel
        if per_channel:
            self.register_buffer("w_scale", torch.ones(self.out_channels))
            self.register_buffer("w_zp", torch.zeros(self.out_channels))
            self.qmin = torch.full((self.out_channels,), -(1 << (weight_bits-1)) + 1, dtype=torch.int)
            self.qmax = torch.full((self.out_channels,), (1 << (weight_bits-1)) - 1, dtype=torch.int)
        else:
            self.register_buffer("w_scale", torch.tensor(1.0))
            self.register_buffer("w_zp", torch.tensor(0.0))
            self.qmin = -(1 << (weight_bits-1)) + 1
            self.qmax = (1 << (weight_bits-1)) - 1

    def set_qparams(self):
        if self.per_channel:
            w = self.weight.detach().view(self.out_channels, -1)
            w_min, _ = w.min(dim=1)
            w_max, _ = w.max(dim=1)
            for i in range(self.out_channels):
                scale, zp, qmin, qmax = qparams_from_minmax(
                    w_min[i], w_max[i], n_bits=self.weight_bits, unsigned=False
                )
                self.w_scale[i] = scale
                self.w_zp[i] = zp
                self.qmin[i] = qmin
                self.qmax[i] = qmax
        else:
            w = self.weight.detach()
            w_min, w_max = w.min(), w.max()
            scale, zp, qmin, qmax = qparams_from_minmax(
                w_min, w_max, n_bits=self.weight_bits, unsigned=False
            )
            self.w_scale.copy_(scale)
            self.w_zp.copy_(zp)
            self.qmin, self.qmax = qmin, qmax

    def forward(self, x):
        if self.per_channel:
            w_flat = self.weight.view(self.out_channels, -1)
            w_dq = []
            for i in range(self.out_channels):
                dq = QuantizeSTE.apply(
                    w_flat[i], self.w_scale[i], self.w_zp[i],
                    int(self.qmin[i]), int(self.qmax[i])
                )
                w_dq.append(dq)
            w_dq = torch.stack(w_dq, dim=0).view_as(self.weight)
        else:
            w_dq = QuantizeSTE.apply(
                self.weight, self.w_scale, self.w_zp, self.qmin, self.qmax
            )
        return F.conv2d(x, w_dq, self.bias, self.stride, self.padding,
                        self.dilation, self.groups)

    def storage_size_bytes(self):
        weight_storage = self.weight.numel() * self.weight_bits // 8
        if self.per_channel:
            # scale, zp, qmin, qmax per channel (4 values per channel, float32 each)
            meta_storage = self.out_channels * 4 * 4
        else:
            # single scale, zp, qmin, qmax
            meta_storage = 4 * 4

        return weight_storage , meta_storage 

class QATLinear(nn.Linear):
    def __init__(self, *args, weight_bits=8, **kwargs):
        super().__init__(*args, **kwargs)
        self.weight_bits = weight_bits
        self.register_buffer("w_scale", torch.tensor(1.0))
        self.register_buffer("w_zp", torch.tensor(0.0))
        self.qmin = -(1 << (weight_bits-1)) + 1
        self.qmax = (1 << (weight_bits-1)) - 1

    def set_qparams(self):
        w = self.weight.detach()
        w_min, w_max = w.min(), w.max()
        scale, zp, qmin, qmax = qparams_from_minmax(
            w_min, w_max, n_bits=self.weight_bits, unsigned=False
        )
        self.w_scale.copy_(scale)
        self.w_zp.copy_(zp)
        self.qmin, self.qmax = qmin, qmax

    def forward(self, x):
        w_dq = QuantizeSTE.apply(
            self.weight, self.w_scale, self.w_zp, self.qmin, self.qmax
        )
        return F.linear(x, w_dq, self.bias)

    def storage_size_bytes(self):
        return self.weight.numel() * self.weight_bits // 8 , 4*3 # for meta

def _ceil_log2(n: int) -> int:
    return 1 if n <= 1 else math.ceil(math.log2(n))

def _kmeans_1d(data, K, iters=10):
    N = data.numel()
    flat = data.reshape(-1)
    perm = torch.randperm(N, device=flat.device)
    codebook = flat[perm[:K]].clone()
    for _ in range(iters):
        dists = (flat[:, None] - codebook[None, :]).abs()
        assign = torch.argmin(dists, dim=1)
        for k in range(K):
            mask = assign == k
            if mask.any():
                codebook[k] = flat[mask].mean()
            else:
                codebook[k] = flat[torch.randint(0, N, (1,), device=flat.device)]
    dists = (flat[:, None] - codebook[None, :]).abs()
    assign = torch.argmin(dists, dim=1)
    return codebook, assign.reshape_as(data)

class KMeansSharedConv2d(nn.Conv2d):
    def __init__(self, *args, num_clusters=16, per_channel=False, kmeans_iters=10, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_clusters = int(num_clusters)
        self.per_channel = bool(per_channel)
        self.kmeans_iters = int(kmeans_iters)
        if self.per_channel:
            cb = torch.zeros(self.out_channels, self.num_clusters)
            self.codebook = nn.Parameter(cb)
            self.register_buffer("assignments", torch.zeros_like(self.weight, dtype=torch.long))
        else:
            cb = torch.zeros(self.num_clusters)
            self.codebook = nn.Parameter(cb)
            self.register_buffer("assignments", torch.zeros_like(self.weight, dtype=torch.long))

    @torch.no_grad()
    def kmeans_init(self):
        w = self.weight.detach()
        if self.per_channel:
            cb = torch.empty_like(self.codebook)
            asg = torch.empty_like(self.assignments)
            for oc in range(self.out_channels):
                sub = w[oc].reshape(-1)
                codebook, assign = _kmeans_1d(sub, self.num_clusters, self.kmeans_iters)
                cb[oc] = codebook
                asg[oc] = assign.reshape_as(w[oc])
            self.codebook.copy_(cb)
            self.assignments.copy_(asg)
        else:
            codebook, assign = _kmeans_1d(w, self.num_clusters, self.kmeans_iters)
            self.codebook.copy_(codebook)
            self.assignments.copy_(assign)

    @torch.no_grad()
    def reassign(self):
        w = self.weight.detach()
        if self.per_channel:
            oc = self.out_channels
            cb = self.codebook
            idx = []
            for i in range(oc):
                wi = w[i].reshape(-1)
                d = (wi[:, None] - cb[i][None, :]).abs()
                idx_i = torch.argmin(d, dim=1).reshape_as(w[i])
                idx.append(idx_i)
            self.assignments.copy_(torch.stack(idx, dim=0))
        else:
            flat = w.reshape(-1)
            d = (flat[:, None] - self.codebook[None, :]).abs()
            idx = torch.argmin(d, dim=1).reshape_as(w)
            self.assignments.copy_(idx)

    def _quantized_weight(self):
        if self.per_channel:
            oc = self.out_channels
            cb = self.codebook  # [oc, K]
            idx = self.assignments.view(oc, -1)  # [oc, M]
            row = torch.arange(oc, device=idx.device).unsqueeze(1).expand_as(idx)  # [oc, M]
            wq_flat = cb[row, idx]  # [oc, M]
            return wq_flat.view_as(self.weight)
        else:
            cb = self.codebook  # [K]
            idx = self.assignments.view(-1)
            wq_flat = cb[idx]
            return wq_flat.view_as(self.weight)

    def forward(self, x):
        wq = self._quantized_weight()
        return F.conv2d(x, wq, self.bias, self.stride, self.padding, self.dilation, self.groups)

    def storage_size_bytes(self, codebook_dtype_bits=32):
        index_bits = _ceil_log2(self.num_clusters)
        indices_bytes = self.weight.numel() * index_bits / 8.0
        if self.per_channel:
            codebook_bytes = self.out_channels * self.num_clusters * codebook_dtype_bits / 8.0
        else:
            codebook_bytes = self.num_clusters * codebook_dtype_bits / 8.0
        bias_bytes = 0 if self.bias is None else self.bias.numel() * 32 / 8.0
        return int(math.ceil(indices_bytes + codebook_bytes + bias_bytes))

class KMeansSharedLinear(nn.Linear):
    def __init__(self, *args, num_clusters=16, kmeans_iters=10, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_clusters = int(num_clusters)
        self.kmeans_iters = int(kmeans_iters)
        cb = torch.zeros(self.num_clusters)
        self.codebook = nn.Parameter(cb)
        self.register_buffer("assignments", torch.zeros_like(self.weight, dtype=torch.long))

    @torch.no_grad()
    def kmeans_init(self):
        codebook, assign = _kmeans_1d(self.weight.detach(), self.num_clusters, self.kmeans_iters)
        self.codebook.copy_(codebook)
        self.assignments.copy_(assign)

    @torch.no_grad()
    def reassign(self):
        flat = self.weight.detach().reshape(-1)
        d = (flat[:, None] - self.codebook[None, :]).abs()
        idx = torch.argmin(d, dim=1).reshape_as(self.weight)
        self.assignments.copy_(idx)

    def _quantized_weight(self):
        cb = self.codebook
        idx = self.assignments.view(-1)
        wq_flat = cb[idx]
        return wq_flat.view_as(self.weight)

    def forward(self, x):
        wq = self._quantized_weight()
        return F.linear(x, wq, self.bias)

    def storage_size_bytes(self, codebook_dtype_bits=32):
        index_bits = _ceil_log2(self.num_clusters)
        indices_bytes = self.weight.numel() * index_bits / 8.0
        codebook_bytes = self.num_clusters * codebook_dtype_bits / 8.0
        bias_bytes = 0 if self.bias is None else self.bias.numel() * 32 / 8.0
        return int(math.ceil(indices_bytes + codebook_bytes + bias_bytes))

def calibrate_and_quantize_activations(model, dataloader, device, num_samples=100, eps=7e-6, n_bits=8):
    from utils import ActivationStats
    model.to(device)
    # 1. collect activations 
    activations = ActivationStats.collect_stats(model, dataloader.testloader, device, num_samples)

    # 2. select layers with low quantization loss
    quantizable = ActivationStats.to_quant_int8(activations, eps=eps, n_bits=n_bits)

    from model import replace_module
    # 3. replace chosen layers with wrapper+ReLU6
    for name, loss, amin, amax, scale in quantizable:
        wrapper = QATActFakeQuant(n_bits=n_bits, unsigned=True)
        wrapper.set_qparams(torch.tensor(amin), torch.tensor(amax))
        new_module = nn.Sequential(wrapper, nn.ReLU6(inplace=True))
        replace_module(model, name, new_module)
        #print(f"Replaced {name} with QATActFakeQuant+ReLU6 (loss={loss:.3e})")

    return model

def conv2d_to_qatconv2d(module: nn.Conv2d, n_bits=8, per_channel=False, device="cpu"):
    """Convert a Conv2d module into a QATConv2d with same parameters and weights."""
    qat_conv = QATConv2d(
        in_channels=module.in_channels,
        out_channels=module.out_channels,
        kernel_size=module.kernel_size,
        stride=module.stride,
        padding=module.padding,
        dilation=module.dilation,
        groups=module.groups,
        bias=(module.bias is not None),
        padding_mode=module.padding_mode,
        weight_bits=n_bits,
        per_channel=per_channel,
    ).to(device)

    # copy weights/bias
    qat_conv.weight.data.copy_(module.weight.data)
    if module.bias is not None:
        qat_conv.bias.data.copy_(module.bias.data)

    # calibrate qparams initially
    qat_conv.set_qparams()
    return qat_conv


def replace_all_conv2d(model, device="cpu", n_bits=8, per_channel=True):
    """Replace all Conv2d modules in the model with QATConv2d wrappers."""
    for name, module in list(model.named_children()):  # only direct children
        if isinstance(module, nn.Conv2d) and not isinstance(module, QATConv2d):
            setattr(model, name, conv2d_to_qatconv2d(module, n_bits, per_channel=per_channel, device=device))
        else:
            # recurse, this is much better than the named_module nightmare
            replace_all_conv2d(module, device=device, n_bits=n_bits, per_channel=per_channel)
    return model

def linear_to_qatlinear(module: nn.Linear, n_bits=8, device="cpu"):
    """Convert a Linear module into a QATLinear with same parameters and weights."""
    qat_linear = QATLinear(
        in_features=module.in_features,
        out_features=module.out_features,
        bias=(module.bias is not None),
        weight_bits=n_bits,
    ).to(device)

    # copy weights/bias
    qat_linear.weight.data.copy_(module.weight.data)
    if module.bias is not None:
        qat_linear.bias.data.copy_(module.bias.data)

    # calibrate qparams initially
    qat_linear.set_qparams()
    return qat_linear


def replace_all_linear(model, device="cpu", n_bits=8):
    """Replace all Linear modules in the model with QATLinear wrappers."""
    for name, module in list(model.named_children()):
        if isinstance(module, nn.Linear) and not isinstance(module, QATLinear):
            setattr(model, name, linear_to_qatlinear(module, n_bits=n_bits, device=device))
        else:
            # recurse
            replace_all_linear(module, device=device, n_bits=n_bits)
    return model