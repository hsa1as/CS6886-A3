#!/bin/bash

WEIGHTS="mobilenetv2_cifar10.pth"

for act in 8 4 2; do
  for conv in 8 4 2; do
    for lin in 8 4 2; do
      RUN_NAME="pll_activation_${act}b_conv_${conv}b_linear_${lin}b"
      echo "Running: $RUN_NAME"
      python3 main.py \
        --weights "$WEIGHTS" \
        --no-train \
        --run-name "$RUN_NAME" \
        --quant \
        --quant-all-conv \
        --quant-linear \
        --quant-linear-bits "$lin" \
        --quant-conv-bits "$conv" \
        --quant-bits "$act"
    done
  done
done
