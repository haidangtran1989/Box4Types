#!/bin/bash

# Select data from {ufet, onto, figer, bbn}
export DATA_NAME=ufet

CUDA_VISIBLE_DEVICES=0 python3 -u main.py \
    --model_id box_${DATA_NAME}_dev \
    --reload_model_name box_${DATA_NAME} \
    --load \
    --model_type bert-large-uncased-whole-word-masking \
    --mode test \
    --goal $DATA_NAME \
    --emb_type box \
    --threshold 0.5 \
    --mc_box_type CenterSigmoidBoxTensor \
    --type_box_type CenterSigmoidBoxTensor \
    --gumbel_beta=0.00036026463511690845 \
    --inv_softplus_temp=1.2471085395024732 \
    --softplus_scale 1.0 \
    --box_dim=109 \
    --proj_layer highway \
    --per_gpu_eval_batch_size 8 \
    --eval_data ${DATA_NAME}/${DATA_NAME}_dev.json

#1998 1998
#Eval: 1998 1998 3.694 P:0.526 R:0.360 F1:0.427 Ma_P:0.529 Ma_R:0.391 Ma_F1:0.450	 Dev EM: 1.9%
