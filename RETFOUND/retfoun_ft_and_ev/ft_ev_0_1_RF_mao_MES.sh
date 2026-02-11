#!/bin/bash

# ==== Model settings ====
# adaptation {finetune,lp}
ADAPTATION="finetune"
MODEL="RETFound_mae"
MODEL_ARCH="retfound_mae"
FINETUNE="RETFound_mae_natureCFP"

for FOLD in {0..9}
do
	# ==== Data settings ====
	# # change the dataset name and corresponding class number
	DATASET="MES_FOLD_01_${FOLD}"
	NUM_CLASS=2
	# =======================
	DATA_PATH="messidor_tvt_01/fold${FOLD}"
	TASK="${MODEL_ARCH}_${DATASET}_${ADAPTATION}"
	torchrun --nproc_per_node=1 --master_port=48766 main_finetune.py \
		--model "${MODEL}" \
		--model_arch "${MODEL_ARCH}" \
		--finetune "${FINETUNE}" \
		--savemodel \
		--global_pool \
		--batch_size 24 \
		--world_size 1 \
		--epochs 50 \
		--nb_classes "${NUM_CLASS}" \
		--data_path "${DATA_PATH}" \
		--input_size 224 \
		--task "${TASK}" \
		--adaptation "${ADAPTATION}" \
        --num_workers 2
          
done 
