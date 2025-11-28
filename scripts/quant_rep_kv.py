import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from kvquant.datasets.calib_dataset import get_calib_redpajama
from kvquant.quantization.methods.smoothattention.apply_smoothattention import (
    get_max_keys,
)
from kvquant.quantization.methods.smoothattention.apply_smoothattention import (
    apply_smoothattention_rep,
)

parser = argparse.ArgumentParser()
parser.add_argument("model_path", type=str, help="Input model path")
parser.add_argument("dataset_path", type=str, help="Calibration dataset")
parser.add_argument("--max-key-path", type=str)
parser.add_argument("--out-model-path", type=str)
parser.add_argument("--n-samples", type=int, default=128)
parser.add_argument("--calib-seq-len", type=int, default=16000)
args = parser.parse_args()

model = AutoModelForCausalLM.from_pretrained(
    args.model_path, device_map="auto", torch_dtype=torch.bfloat16
)
tokenizer = AutoTokenizer.from_pretrained(args.model_path)

calib_dataset = get_calib_redpajama(
    args.dataset_path, args.n_samples, args.calib_seq_len, tokenizer
)
max_keys = get_max_keys(model, calib_dataset, args.max_key_path)

apply_smoothattention_rep(model, args.max_key_path)
model.save_pretrained(args.out_model_path)
tokenizer.save_pretrained(args.out_model_path)
