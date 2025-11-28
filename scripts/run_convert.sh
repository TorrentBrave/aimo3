# This script is used for converting the model to turbomind format

export CUDA_VISIBLE_DEVICES="0,1,2,3"
lmdeploy convert your_model_name /path/to/your/model/ \
    --dst-path /path/to/your/turbomind/model/ \
    --tp 4