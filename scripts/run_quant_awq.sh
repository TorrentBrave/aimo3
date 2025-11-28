#!/bin/bash

# for examples
CUDA_VISIBLE_DEVICES=0 python scripts/quant_awq.py ./models/quantized_models/repkv_models/dpsk-qwen-14b-finetune-v1-epoch4-repkv --out-model-path ./models/quantized_models/dpsk-qwen-14b-finetune-v1-epoch4-repkv-awq
CUDA_VISIBLE_DEVICES=0 python scripts/quant_awq.py ./models/quantized_models/repkv_models/dpsk-qwen-14b-finetune-v1-epoch2-repkv --out-model-path ./models/quantized_models/dpsk-qwen-14b-finetune-v1-epoch2-repkv-awq
CUDA_VISIBLE_DEVICES=0 python scripts/quant_awq.py ./models/quantized_models/repkv_models/dpsk-qwen-14b-repkv --out-model-path ./models/quantized_models/dpsk-qwen-14b-repkv-awq
