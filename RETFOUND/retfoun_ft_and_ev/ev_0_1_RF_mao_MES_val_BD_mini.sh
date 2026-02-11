#!/bin/bash

# ==== Model settings ====
# adaptation {finetune,lp}
ADAPTATION="finetune"
MODEL="RETFound_mae"
MODEL_ARCH="retfound_mae"
#FINETUNE="RETFound_mae_natureCFP"

for FOLD in {0..9}
do
	# ==== Data settings ====
	# # change the dataset name and corresponding class number
	DATASET_TRAIN="MES_FOLD_01_${FOLD}"
    DATASET="MES_val_BD_mini_FOLD_${FOLD}"
	NUM_CLASS=2
	# =======================
    DATA_PATH="mini_BD_tvt_01/fold${FOLD}"
    TASK_TRAIN="${MODEL_ARCH}_${DATASET_TRAIN}_${ADAPTATION}"
    TASK="${MODEL_ARCH}_${DATASET}_${ADAPTATION}"
    
    # Path to the trained checkpoint (adjust if you saved elsewhere)
    CKPT="./output_dir/${TASK_TRAIN}/checkpoint-best.pth"
    # ==== Evaluation only ====
    torchrun --nproc_per_node=1 --master_port=48766 extract_f_p_RETFound.py \
    --model "${MODEL}" \
    --batch_size 128 \
    --nb_classes "${NUM_CLASS}" \
    --data_path "${DATA_PATH}" \
    --input_size 224 \
    --task "${TASK}" \
    --num_workers 0 \
    --resume "${CKPT}"        
done 
