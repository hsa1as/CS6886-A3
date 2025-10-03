import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from model import MobileNetWrapper, get_mobilenet_kwargs
from data import CIFAR10Data
import torch.nn as nn
import random
from torchinfo import summary

from collections import OrderedDict

def strip_prefix(state_dict, prefix="_orig_mod."):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k.startswith(prefix):
            new_state_dict[k[len(prefix):]] = v
        else:
            new_state_dict[k] = v
    return new_state_dict

def qparams_from_minmax(xmin, xmax, n_bits=8, unsigned=False, eps=1e-12):
    """
    Returns (scale, zero_point, qmin, qmax) for uniform quant.
    - unsigned=True  -> [0, 2^b - 1]
    - unsigned=False -> symmetric int range [-2^(b-1)+1, 2^(b-1)-1]
    """
    if unsigned:
        qmin, qmax = 0, (1 << n_bits) - 1
        # (common for post-ReLU) ensure non-negative min for tighter range
        xmin = torch.zeros_like(xmin)
        scale = (xmax - xmin).clamp_min(eps) / float(qmax - qmin)
        zp = torch.round(-xmin / scale).clamp(qmin, qmax)
    else:
        qmax = (1 << (n_bits - 1)) - 1
        qmin = -qmax
        max_abs = torch.max(xmin.abs(), xmax.abs()).clamp_min(eps)
        scale = max_abs / float(qmax)
        zp = torch.zeros_like(scale)
    return scale, zp, int(qmin), int(qmax)

class WeightVisualizer:
    @staticmethod
    def plot(model, bins=100):
        all_weights = []
        for p in model.parameters():
            if p.requires_grad:
                all_weights.extend(p.detach().cpu().numpy().ravel())
        all_weights = np.array(all_weights)
        mask = all_weights != 0
        log_abs = np.log10(np.abs(all_weights[mask]))
        plt.figure(figsize=(8, 5))
        plt.hist(log_abs, bins=bins, color="steelblue", edgecolor="black", alpha=0.7)
        plt.title("Histogram of Weight Magnitudes (log scale)")
        plt.xlabel("Weight magnitude")
        plt.ylabel("Frequency")
        plt.grid(True, linestyle="--", alpha=0.6)
        locs, _ = plt.xticks()
        plt.xticks(locs, [f"{10**x:.0e}" for x in locs])
        plt.show()

class BatchNormVisualizer:
    @staticmethod
    def show_gamma(model, as_table=False, plot_min=True):
        gammas = []
        for name, module in model.named_modules():
            if isinstance(module, nn.BatchNorm2d):
                gamma = module.weight.detach().cpu().numpy()
                gammas.append((name, gamma))
        if as_table:
            for name, gamma in gammas:
                gmin, gmax, gmean = gamma.min(), gamma.max(), gamma.mean()
                print(f"{name:40s} | min={gmin:.4f} max={gmax:.4f} mean={gmean:.4f}")
        elif plot_min:
            names = [name for name, _ in gammas]
            min_vals = [gamma.min() for _, gamma in gammas]

            x = np.arange(len(names))
            plt.figure(figsize=(max(12, len(names) * 0.4), 6))
            plt.bar(x, min_vals, color="coral")
            plt.xticks(x, names, rotation=90)
            plt.ylabel("Min Gamma")
            plt.title("Min BatchNorm Gamma per Layer")
            plt.tight_layout()
            plt.show()
        else:
            for name, gamma in gammas:
                plt.figure(figsize=(8, 4))
                plt.bar(np.arange(len(gamma)), gamma)
                plt.title(f"BN Gamma values - {name}")
                plt.xlabel("Channel")
                plt.ylabel("Gamma")
                plt.show()

class LayerStats:
    @staticmethod
    def calculate_stats(model, n_bits=8, plot=False, unsigned=False, module_types=(nn.Conv2d,), plot_total=True):
        stats = {}
        total_losses = {}
        for name, module in model.named_modules():
            total_loss = 0
            if isinstance(module, module_types) and hasattr(module, "weight"):
                W = module.weight.detach().cpu().numpy()  # shape [out_c, in_c/groups, kH, kW]
                out_c = W.shape[0]

                stats[name] = []
                for oc in range(out_c):
                    w = W[oc].ravel()
                    wmin, wmax = w.min(), w.max()

                    scale, zp, qmin, qmax = qparams_from_minmax(
                        torch.tensor(wmin), torch.tensor(wmax),
                        n_bits=n_bits, unsigned=unsigned
                    )
                    scale, zp = float(scale), float(zp)

                    if scale > 0:
                        q = np.round(w / scale + zp).clip(qmin, qmax)
                        dq = (q - zp) * scale
                        loss = np.mean((w - dq) ** 2)
                    else:
                        loss = 0.0
                    total_loss += loss
                    stats[name].append({
                        "out_channel": oc,
                        "loss": loss,
                        "min": float(wmin),
                        "max": float(wmax),
                        "scale": scale,
                        "qmin": qmin,
                        "qmax": qmax,
                    })
                total_losses[name] = total_loss
                if plot:
                    losses = [s["loss"] for s in stats[name]]
                    plt.figure(figsize=(max(8, out_c * 0.2), 4))
                    plt.bar(np.arange(out_c), losses, color="coral")
                    plt.xlabel("Output Channel")
                    plt.ylabel("Quantization MSE Loss")
                    plt.title(f"Quantization Loss per Output Channel - {name}")
                    plt.tight_layout()
                    plt.show()
        if plot_total and total_losses:
            names = list(total_losses.keys())
            losses = list(total_losses.values())
            plt.figure(figsize=(max(12, len(names) * 0.4), 6))
            plt.bar(np.arange(len(names)), losses, color="steelblue")
            plt.xticks(np.arange(len(names)), names, rotation=90)
            plt.ylabel("Total Quantization MSE Loss")
            plt.title("Total Conv2d Quantization Loss per Layer")
            plt.tight_layout()
            plt.show()
        return stats

    @staticmethod
    def show_layer(layer_name, module, plot=False, bins=100):
        if not hasattr(module, "weight"):
            return
        w = module.weight.detach().cpu().numpy().ravel()
        wmin, wmax = w.min(), w.max()
        scale = (wmax - wmin) / 255.0 if wmax > wmin else 0.0
        print(f"{layer_name:40s} | min={wmin:.6f} max={wmax:.6f} int8_scale={scale:.6f}")
        if(plot):
            mask = w != 0
            log_abs = np.log10(np.abs(w[mask])) if mask.any() else np.array([0])
            plt.figure(figsize=(8, 5))
            plt.hist(log_abs, bins=bins, color="steelblue", edgecolor="black", alpha=0.7)
            plt.title(f"Histogram of Weights (log scale) - {layer_name}")
            plt.xlabel("Weight magnitude")
            plt.ylabel("Frequency")
            plt.grid(True, linestyle="--", alpha=0.6)
            locs, _ = plt.xticks()
            plt.xticks(locs, [f"{10**x:.0e}" for x in locs])
            plt.show()

    @staticmethod
    def show_all(model, plot=False, bins=100):
        for name, module in model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                LayerStats.show_layer(name, module,plot=plot,bins=bins)

class ActivationStats:
    @staticmethod
    def collect_stats(model, dataloader, device, num_samples):
        model.eval()
        hooks = {}
        activations = {}

        def hook_fn(name):
            def fn(_, __, output):
                out = output.detach().cpu().numpy().ravel()
                if name not in activations:
                    activations[name] = []
                activations[name].append(out)
            return fn

        for name, module in model.named_modules():
            if isinstance(module, nn.ReLU6):
                hooks[name] = module.register_forward_hook(hook_fn(name))

        # collect class balanced num_samples
        per_class = num_samples // 10
        idxs_per_class = {c: [] for c in range(10)}
        for i in range(len(dataloader.dataset)):
            _, label = dataloader.dataset[i]
            if len(idxs_per_class[label]) < per_class:
                idxs_per_class[label].append(i)
            if all(len(v) == per_class for v in idxs_per_class.values()):
                break
        chosen_idxs = [i for v in idxs_per_class.values() for i in v]
        random.shuffle(chosen_idxs)
        subset = torch.utils.data.Subset(dataloader.dataset, chosen_idxs)
        loader = torch.utils.data.DataLoader(subset, batch_size=32, shuffle=False)

        with torch.no_grad():
            for x, _ in loader:
                x = x.to(device)
                _ = model(x)

        for h in hooks.values():
            h.remove()
        return activations

    @staticmethod
    def to_quant_int8(activations, n_bits=8, unsigned=True, eps=1e-4):
        stats = []
        for name, acts in activations.items():
            arr = np.concatenate(acts, axis=0)
            amin, amax = arr.min(), arr.max()

            # derive quantization params
            scale, zp, qmin, qmax = qparams_from_minmax(
                torch.tensor(amin), torch.tensor(amax),
                n_bits=n_bits, unsigned=unsigned
            )
            scale, zp = float(scale), float(zp)

            if scale > 0:
                q = np.round(arr / scale + zp).clip(qmin, qmax)
                dq = (q - zp) * scale
                loss = np.mean((arr - dq) ** 2)
            else:
                loss = 0.0

            if loss <= eps:
                stats.append((name, loss, amin, amax, scale))
        return stats

    @staticmethod
    def show(model, dataloader, device, num_samples=100, plot=True):
        activations = ActivationStats.collect_stats(model, dataloader, device, num_samples)
        stats = []
        qloss = []
        for name, acts in activations.items():
            arr = np.concatenate(acts, axis=0)
            amin, amax = arr.min(), arr.max()
            scale = (amax - amin) / 255.0 if amax > amin else 0.0

            if scale > 0:
                q = np.round((arr - amin) / scale).clip(0, 255)
                dq = q * scale + amin
                loss = np.mean((arr - dq) ** 2)
            else:
                loss = 0.0

            stats.append((name, amin, amax, scale))
            qloss.append((name, loss))
            print(f"{name:40s} | min={amin:.6f} max={amax:.6f} int8_scale={scale:.6f} qloss={loss:.6e}")

        if plot and stats:
            names = [s[0] for s in stats]
            amin_vals = [s[1] for s in stats]
            amax_vals = [s[2] for s in stats]
            scale_vals = [s[3] for s in stats]
            losses = [l[1] for l in qloss]

            x = np.arange(len(names))
            width = 0.35

            fig, ax1 = plt.subplots(figsize=(max(12, len(names)*0.4), 6))
            ax1.bar(x - width/2, amin_vals, width/2, label="min", color="steelblue")
            ax1.bar(x, amax_vals, width/2, label="max", color="orange")
            ax1.set_ylabel("Activation Min/Max")
            ax1.set_xticks(x)
            ax1.set_xticklabels(names, rotation=90)

            ax2 = ax1.twinx()
            ax2.bar(x + width/2, scale_vals, width/2, label="int8 scale", color="green")
            ax2.set_ylabel("Int8 Scale")

            h1, l1 = ax1.get_legend_handles_labels()
            h2, l2 = ax2.get_legend_handles_labels()
            ax1.legend(h1 + h2, l1 + l2, loc="upper right")
            plt.title("Post-ReLU6 Activation Stats per Layer")
            plt.tight_layout()
            plt.show()

            plt.figure(figsize=(max(12, len(names)*0.4), 6))
            plt.bar(x, losses, color="red")
            plt.xticks(x, names, rotation=90)
            plt.ylabel("MSE Loss (Quantization)")
            plt.title("Post-ReLU6 Activation Quantization Loss (int8)")
            plt.tight_layout()
            plt.show()

def evaluate(model, dataloader, device, loss_fn=None):
    model.eval()
    if loss_fn is None:
        loss_fn = nn.CrossEntropyLoss()

    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)

            outputs = model(x)
            loss = loss_fn(outputs, y)

            total_loss += loss.item() * x.size(0)

            preds = outputs.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)

    avg_loss = total_loss / total
    acc = correct / total
    return acc, avg_loss

# since i am laxy, do this online
def largest_relu6_layer(model, dataloader, device="cpu"):

    model.eval()
    max_layer, max_size = None, 0

    x, _ = next(iter(dataloader))
    x = x.to(device)

    original_forwards = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.ReLU6):
            original_forwards[name] = module.forward

            def make_forward(n):
                def forward_fn(self, inp):
                    out = nn.functional.relu6(inp)
                    size = out[0].numel()  # per sample size
                    nonlocal max_layer, max_size
                    if size > max_size:
                        max_size =size
                    return out
                return forward_fn

            module.forward = make_forward(name).__get__(module, nn.ReLU6)

    with torch.no_grad():
        _ = model(x)

    for name, module in model.named_modules():
        if isinstance(module, nn.ReLU6):
            module.forward = original_forwards[name]

    return max_size

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_classes", type=int, default=10)
    parser.add_argument("--width_mult", type=float, default=0.5)
    parser.add_argument("--round_nearest", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--weights", type=str, required=True)
    parser.add_argument("--device", type=str, default="cpu")

    parser.add_argument("--histogram", action="store_true")
    parser.add_argument("--bn-gamma", action="store_true")
    parser.add_argument("--bn-table", action="store_true")
    parser.add_argument("--layer-stats", action="store_true")
    parser.add_argument("--layer-stats-plot-all", action="store_true")
    parser.add_argument("--layer-stats-plot-all-bins", type=int, default=100)
    parser.add_argument("--activation-stats", action="store_true")
    parser.add_argument("--torchinfo", action="store_true")
    parser.add_argument("--perlayer-conv-int8-loss", action="store_true")
    parser.add_argument("--get-max-act-size",action="store_true")
    args = parser.parse_args()

    mobilenet_params = get_mobilenet_kwargs(args)
    model = MobileNetWrapper(**mobilenet_params)
    state_dict = strip_prefix(torch.load(args.weights, map_location=args.device))
    model.load_state_dict(state_dict)
    model.to(args.device)
    if args.torchinfo:
        input_size = (1,3,224,224)
        print(summary(model, input_size=input_size, device=args.device))
    if args.histogram:
        WeightVisualizer.plot(model)
    if args.bn_gamma:
        BatchNormVisualizer.show_gamma(model, as_table=args.bn_table)
    if args.layer_stats:
        LayerStats.calculate_stats(model, n_bits=8)
        LayerStats.show_all(model, plot=args.layer_stats_plot_all, bins=args.layer_stats_plot_all_bins)
    if args.activation_stats:
        data = CIFAR10Data(batch_size=256)
        ActivationStats.show(model, data.testloader, args.device)
    if args.perlayer_conv_int8_loss:
        LayerStats.calculate_stats(model, n_bits=8, plot=True)
    if args.get_max_act_size:
        data = CIFAR10Data(batch_size=1)
        print("Largest Relu is ", largest_relu6_layer(model,data.testloader,device=args.device))

if __name__ == "__main__":
    main()
