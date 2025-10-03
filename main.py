import argparse
import wandb
from data import CIFAR10Data
from model import MobileNetWrapper, get_mobilenet_kwargs, get_model_size, get_quantized_model_size
from trainer import Trainer
from utils import evaluate 
import torch
    

def main():
    parser = argparse.ArgumentParser()

    # MobileNetV2 parameters
    parser.add_argument("--num_classes", type=int, default=10)
    parser.add_argument("--width_mult", type=float, default=0.5)
    parser.add_argument("--round_nearest", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--weights", type=str, default=None)

    # training
    parser.add_argument("--no-train", action="store_true")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument('--batch-size', type=int, default=256)

    # device
    parser.add_argument("--device", type=str, default="cuda:0")

    # pruning

    # quantisattion
    parser.add_argument("--quant", action="store_true")
    parser.add_argument("--quant-bits-act", type=int, default=8)
    parser.add_argument("--quant-all-conv", action="store_true")
    parser.add_argument("--quant-conv-bits", type=int, default=8)
    parser.add_argument("--quant-conv-perchannel", action="store_true")
    parser.add_argument("--quant-linear", action="store_true")
    parser.add_argument("--quant-linear-bits", type=int, default=8)



    # wandb params
    parser.add_argument("--wandb-project", type=str, default="Assignment3")
    parser.add_argument("--run-name", type=str, default=None)

    # get args
    args = parser.parse_args()

    mobilenet_params = get_mobilenet_kwargs(args)

    wandb.init(name=args.run_name, project=args.wandb_project, config={**vars(args), **{"mobilenet": mobilenet_params}})

    torch.backends.cudnn.benchmark = True

    data = CIFAR10Data(batch_size=args.batch_size)
    model = MobileNetWrapper(**mobilenet_params)

    # if weights is not None, we will load model weights
    if(args.weights is not None):
        from utils import strip_prefix
        state_dict = strip_prefix(torch.load(args.weights, map_location="cpu"))
        model.load_state_dict(state_dict)
        model.to(args.device)
        acc, loss = evaluate(model, data.testloader, args.device)
        wandb.log({"rand_test_acc": acc})
        print("Raw model #Params", sum(p.numel() for p in model.parameters()))
        print("Raw model size: ", get_model_size(model)[1])
        print("Raw model accuracy: ", acc)

    wandb.log({"model/num_params": sum(p.numel() for p in model.parameters())})
    wandb.log({"model/raw_size": get_model_size(model)[1]})

    if not args.no_train:
        trainer = Trainer(model, (data.trainloader, data.testloader), device=args.device)
        best_acc = trainer.fit(epochs=args.epochs)   
    else:
        if args.weights is None:
            print("Not sure what i should do now.....")
            exit(1)

    if(args.quant):
        from quant import calibrate_and_quantize_activations
        calibrate_and_quantize_activations(model, data, args.device, num_samples=100, eps=7, n_bits=args.quant_bits_act)
    if(args.quant_linear):
        from quant import replace_all_linear
        replace_all_linear(model, device=args.device, n_bits=args.quant_linear_bits)
    if(args.quant_all_conv):
        from quant import replace_all_conv2d
        replace_all_conv2d(model, device=args.device, n_bits=args.quant_conv_bits, per_channel=args.quant_conv_perchannel)
    acc, loss = evaluate(model, data.testloader, args.device)
    wandb.log({"model/post_comp_acc": acc})
    quant_model_size = get_quantized_model_size(model)
    wandb.log({"model/post_comp_size": quant_model_size})

    print("Quantized model #Params = ", sum(p.numel() for p in model.parameters()))
    print("Quantized model size = ", quant_model_size)
    print("Quantized model Accuracy = ", acc)



if __name__=="__main__":
    main()
