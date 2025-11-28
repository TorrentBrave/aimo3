# This script is modefied from the script 'train-dpo.sh' provided by Light-R1 project.
hostfile="hostfile.2nodes"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
deepspeed --hostfile $hostfile src/train.py \
    --stage dpo \
    --do_train \
    --max_steps -1 \
    --model_name_or_path /path/to/model/dpsk-14b-sft \
    --template qwen \
    --dataset dpo-1 \
    --preprocessing_num_workers 16 \
    --finetuning_type full \
    --sequence_parallel_size 8 \
    --gradient_checkpointing True \
    --flash_attn fa2  \
    --pref_beta 0.3 \
    --pref_loss nca_pair \
    --cache_dir .cache \
    --overwrite_cache \
    --cutoff_len 20000 \
    --output_dir /path/to/model/dpsk-14b-sft-dpo \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --lr_scheduler_type constant \
    --save_strategy epoch \
    --logging_steps 1 \
    --save_total_limit 8 \
    --learning_rate 5e-7 \
    --save_only_model True \
    --num_train_epochs 4.0 \
    --bf16 true \
    --plot_loss \
    --seed 42 \
    --do_eval false \
    --deepspeed ./examples/deepspeed/ds_z3_offload_config.json \
    --report_to tensorboard \
    --overwrite_output_dir \
    --ddp_timeout 180000000 \