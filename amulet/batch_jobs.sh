#!/bin/bash -x

# --model.init_args.load_ckpt_file "$AMLT_DATA_DIR/model_ckpts/step=22818-exec_acc=0.5156-exec_rate=0.7236.ckpt" \
# --model.init_args.load_ckpt_file "$AMLT_DATA_DIR/model_ckpts/mle_best_pass_80.ckpt" \

export EXP_NAME="mathqa_full-mle"
export CUDA_VISIBLE_DEVICES=0,1,2,3
export MASTER_PORT=52000
python trainer.py fit \
    --config training_configs/mathqa_gpt_finetuning.yaml \
    --seed_everything 1 \
    --trainer.default_root_dir "$AMLT_OUTPUT_DIR/$EXP_NAME" \
    --data.init_args.train_file_path "$AMLT_DATA_DIR/train-python.jsonl" \
    --data.init_args.val_file_path "$AMLT_DATA_DIR/val-python.jsonl" \
    --trainer.gpus 4 &

sleep 1m

export EXP_NAME="mathqa_full-mml"
export CUDA_VISIBLE_DEVICES=4,5,6,7
export MASTER_PORT=52100
python trainer.py fit \
    --config training_configs/mathqa_gpt_mml_finetuning.yaml \
    --seed_everything 2 \
    --trainer.default_root_dir "$AMLT_OUTPUT_DIR/$EXP_NAME" \
    --data.init_args.train_file_path "$AMLT_DATA_DIR/train-python.jsonl" \
    --data.init_args.val_file_path "$AMLT_DATA_DIR/val-python.jsonl" \
    --model.init_args.mle_lambda 0.0 \
    --model.init_args.mml_lambda 1.0 \
    --trainer.gpus 4 &

sleep 1m

export EXP_NAME="mathqa_full-mml-beta_mml"
export CUDA_VISIBLE_DEVICES=8,9,10,11
export MASTER_PORT=52200
python trainer.py fit \
    --config training_configs/mathqa_gpt_mml_finetuning.yaml \
    --seed_everything 3 \
    --trainer.default_root_dir "$AMLT_OUTPUT_DIR/$EXP_NAME" \
    --data.init_args.train_file_path "$AMLT_DATA_DIR/train-python.jsonl" \
    --data.init_args.val_file_path "$AMLT_DATA_DIR/val-python.jsonl" \
    --model.init_args.mle_lambda 0.0 \
    --model.init_args.mml_lambda 1.0 \
    --model.init_args.beta_smoothing 0.25 \
    --trainer.gpus 4 &

sleep 1m

export EXP_NAME="mathqa_full-partial_mml-mml"
export CUDA_VISIBLE_DEVICES=12,13,14,15
export MASTER_PORT=52300
python trainer.py fit \
    --config training_configs/mathqa_gpt_partial_mml_finetuning.yaml \
    --seed_everything 4 \
    --trainer.default_root_dir "$AMLT_OUTPUT_DIR/$EXP_NAME" \
    --data.init_args.train_file_path "$AMLT_DATA_DIR/train-python.jsonl" \
    --data.init_args.val_file_path "$AMLT_DATA_DIR/val-python.jsonl" \
    --model.init_args.mle_lambda 0.0 \
    --model.init_args.mml_lambda 1.0 \
    --model.init_args.beta_smoothing 1.0 \
    --trainer.gpus 4 &

wait

sleep 1d
