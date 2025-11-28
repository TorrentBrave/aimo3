# This script is used to run the deduplication process on limo and light-r1
python deduplicate_sft_data.py \
    --input_file1 /path/to/limo.json \
    --input_file2 /path/to/light-r1/stage2-3k.json \
    --output_file /path/to/output/deduplicated.json \