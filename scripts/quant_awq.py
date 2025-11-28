import os
import argparse

from transformers import AutoTokenizer
from awq import AutoAWQForCausalLM  # from the autoawq package

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", type=str, help="Input model path")
    parser.add_argument("--out-model-path", type=str)
    parser.add_argument("--calib-seq-len", default=128)
    args = parser.parse_args()

    quant_config = {
        "zero_point": True,
        "q_group_size": 128,
        "w_bit": 4,
        "version": "GEMM",
    }

    # Load model
    model = AutoAWQForCausalLM.from_pretrained(args.model_path, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    # Quantize
    model.quantize(
        tokenizer,
        quant_config=quant_config,
        max_calib_seq_len=args.calib_seq_len,
        max_calib_samples=1024,
    )

    # Save quantized model
    model.save_quantized(args.out_model_path)
    tokenizer.save_pretrained(args.out_model_path)
