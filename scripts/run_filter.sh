# This script is used for filtering the dataset for dpo
# dpo-1: use length ratio and min length, 1.0 means do not considering similarity
python filter_dpo_data.py \
    --similarity_threshold 1.0 \
    --length_ratio_threshold 0.6 \
    --min_length_threshold 8000 \
    --num_files 3 \
    --output_file /path/to/output/json \
    --input_dir /path/to/input/folder \

# dpo-2: use length ratio, min length and similarity
python filter_dpo_data.py \
    --similarity_threshold 0.83 \
    --length_ratio_threshold 0.7 \
    --min_length_threshold 8000 \
    --num_files 10 \
    --output_file /path/to/output/json \
    --input_dir /path/to/input/folder \