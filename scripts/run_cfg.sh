#!/bin/bash

SEED=${SEED:-123}
cfg_file=$1
base_exp_name=$(basename $cfg_file)
base_exp_name=${base_exp_name%".yaml"}
data_file=$2
base_data_name=$(basename $data_file)
base_data_name=${base_data_name%".csv"}

cmd="python imagination_aimo2/local_eval.py ${cfg_file} --seed ${SEED} --exam-dataset-files ${data_file} --output-path results/0325-${base_exp_name}/seed${SEED}/${base_data_name}"

echo $cmd
eval $cmd
