#!/bin/bash

# for examples
CUDA_VISIBLE_DEVICES=0 python scripts/quant_rep_kv.py models/dpsk-qwen-14b-finetune-v1-epoch4/ /mnt/public/public_datasets/RedPajama-Data-1T-Sample --max-key-path ./models/quantized_models/repkv_models/dpsk-qwen-14b-finetune-v1-epoch4-repkv-maxkey.pt --out-model-path ./models/quantized_models/repkv_models/dpsk-qwen-14b-finetune-v1-epoch4-repkv
CUDA_VISIBLE_DEVICES=0 python scripts/quant_rep_kv.py models/dpsk-qwen-14b-finetune-v1-epoch2/ /mnt/public/public_datasets/RedPajama-Data-1T-Sample --max-key-path ./models/quantized_models/repkv_models/dpsk-qwen-14b-finetune-v1-epoch2-repkv-maxkey.pt --out-model-path ./models/quantized_models/repkv_models/dpsk-qwen-14b-finetune-v1-epoch2-repkv
CUDA_VISIBLE_DEVICES=0 python scripts/quant_rep_kv.py models/dpsk-qwen-14b/ /mnt/public/public_datasets/RedPajama-Data-1T-Sample --max-key-path ./models/quantized_models/repkv_models/dpsk-qwen-14b-repkv-maxkey.pt --out-model-path ./models/quantized_models/repkv_models/dpsk-qwen-14b-repkv
