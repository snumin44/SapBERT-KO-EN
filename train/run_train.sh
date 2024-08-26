#!/bin/sh

GPU_ID=0
CUDA_VISIBLE_DEVICES=$GPU_ID python3 train.py \
    	--model 'klue/bert-base' \
        --tokenizer 'klue/bert-base' \
    	--train_data '../data/kostom_pair.csv' \
    	--output_path '../output/pretrained_model' \
    	--epochs 1 \
        --batch_size 64 \
        --max_length 64 \
        --dropout 0.1 \
        --threshold 0.8 \
        --scale_pos 1 \
        --scale_neg 60 \
        --pooler 'cls' \
        --eval_step 100 \
        --amp