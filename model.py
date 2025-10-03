import torchvision
import torch
import torch.nn as nn
import inspect
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights, MobileNetV2
import wandb

# to insert fake quant for evals
def replace_module(model, layer_name, new_module):
    parts = layer_name.split(".")
    parent = model
    for p in parts[:-1]:
        if p.isdigit():
            parent = parent[int(p)]
        else:
            parent = getattr(parent, p)
    last = parts[-1]
    if last.isdigit():
        parent[int(last)] = new_module
    else:
        setattr(parent, last, new_module)

# Class wrapper
class MobileNetWrapper(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        model = torchvision.models.mobilenet_v2(*args, **kwargs)
        self.model = model

    def forward(self, x):
        return self.model(x)

# i need this to get relevant cli params to construct mobilenet
def get_mobilenet_kwargs(args):
    args_dict = vars(args)

    sig = inspect.signature(MobileNetV2.__init__)
    mobilenet_keys = sig.parameters.keys()

    # drop 'self'
    mobilenet_keys = [k for k in mobilenet_keys if k != "self"]

    mobilenet_kwargs = {k: v for k, v in args_dict.items() if k in mobilenet_keys and v is not None}
    return mobilenet_kwargs

# get model size from parameters(), this assumes float32
def get_model_size(model: torch.nn.Module):
    # Total number of trainable params
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # Approx memory (bytes)
    mem_size = num_params * 4 
    return num_params, mem_size

def get_quantized_model_size(model):
    custom_classes = {"QATLinear", "QATConv2d", "KMeansSharedConv2d", "KMeansSharedLinear"}
    fp32_classes = {"Conv2d", "BatchNorm2d", "Linear"}
    weight_bytes = 0
    meta_bytes = 0
    for name, module in model.named_modules():
        cls_name = module.__class__.__name__

        # custom modules (use their storage_size_bytes)
        if cls_name in custom_classes:
            weight_bytes_this,  meta_bytes_this = module.storage_size_bytes()
            weight_bytes += weight_bytes_this
            meta_bytes += meta_bytes_this

        # vanilla FP32 layers
        elif cls_name in fp32_classes:
            n_params = sum(p.numel() for p in module.parameters())
            weight_bytes += n_params * 4  # fp32 = 4 bytes
        elif cls_name == "QATActFakeQuant":
            meta_bytes += 12
    total_bytes = weight_bytes + meta_bytes
    wandb.log({"model/weight_bytes": weight_bytes, "model/meta_bytes": meta_bytes})
    return total_bytes