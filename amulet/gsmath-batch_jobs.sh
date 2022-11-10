#!/bin/bash -x

# --model.init_args.load_ckpt_file "$AMLT_DATA_DIR/model_ckpts/step=22818-exec_acc=0.5156-exec_rate=0.7236.ckpt" \
# --model.init_args.load_ckpt_file "$AMLT_DATA_DIR/model_ckpts/mle_best_pass_80.ckpt" \

# export EXP_NAME="gsmath-2.7B-partial_mml-mle_aug-eval_val"
# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# export MASTER_PORT=56220
# python trainer.py validate \
#     --config training_configs/mathqa_gpt_partial_mml_finetuning.yaml \
#     --seed_everything 1 \
#     --trainer.default_root_dir "$AMLT_OUTPUT_DIR/$EXP_NAME" \
#     --data.init_args.train_file_path "$AMLT_DATA_DIR/gsmath_train.jsonl" \
#     --data.init_args.val_file_path "$AMLT_DATA_DIR/gsmath_val.jsonl" \
#     --model.init_args.load_ckpt_file "$AMLT_DATA_DIR/model_ckpts/gsmath-2.7B-partial_mml-mle_aug-step=16007-exec_acc=0.2086-exec_rate=0.9993-fp32.ckpt" \
#     --model.init_args.pass_at_k 100 \
#     --trainer.gpus 8

# export EXP_NAME="gsmath-2.7B-partial_mml-mle_aug-eval_val-full"
# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# export MASTER_PORT=56230
# python trainer.py validate \
#     --config training_configs/mathqa_gpt_partial_mml_finetuning.yaml \
#     --seed_everything 1 \
#     --trainer.default_root_dir "$AMLT_OUTPUT_DIR/$EXP_NAME" \
#     --data.init_args.train_file_path "$AMLT_DATA_DIR/gsmath_train.jsonl" \
#     --data.init_args.val_file_path "$AMLT_DATA_DIR/gsmath_val.jsonl" \
#     --model.init_args.load_ckpt_file "$AMLT_DATA_DIR/model_ckpts/gsmath-2.7B-partial_mml-mle_aug-step=22967-exec_acc=0.2099-exec_rate=0.9986-fp32.ckpt" \
#     --model.init_args.pass_at_k 100 \
#     --trainer.gpus 8

# export EXP_NAME="gsmath-2.7B-partial_mml-mle_aug-eval_test"
# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# export MASTER_PORT=56240
# python trainer.py validate \
#     --config training_configs/mathqa_gpt_partial_mml_finetuning.yaml \
#     --seed_everything 1 \
#     --trainer.default_root_dir "$AMLT_OUTPUT_DIR/$EXP_NAME" \
#     --data.init_args.train_file_path "$AMLT_DATA_DIR/gsmath_train.jsonl" \
#     --data.init_args.val_file_path "$AMLT_DATA_DIR/gsmath_test.jsonl" \
#     --model.init_args.load_ckpt_file "$AMLT_DATA_DIR/model_ckpts/gsmath-2.7B-partial_mml-mle_aug-step=16007-exec_acc=0.2086-exec_rate=0.9993-fp32.ckpt" \
#     --model.init_args.pass_at_k 100 \
#     --trainer.gpus 8

# export EXP_NAME="gsmath-2.7B-partial_mml-mle_aug-eval_test-full"
# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# export MASTER_PORT=56250
# python trainer.py validate \
#     --config training_configs/mathqa_gpt_partial_mml_finetuning.yaml \
#     --seed_everything 1 \
#     --trainer.default_root_dir "$AMLT_OUTPUT_DIR/$EXP_NAME" \
#     --data.init_args.train_file_path "$AMLT_DATA_DIR/gsmath_train.jsonl" \
#     --data.init_args.val_file_path "$AMLT_DATA_DIR/gsmath_test.jsonl" \
#     --model.init_args.load_ckpt_file "$AMLT_DATA_DIR/model_ckpts/gsmath-2.7B-partial_mml-mle_aug-step=22967-exec_acc=0.2099-exec_rate=0.9986-fp32.ckpt" \
#     --model.init_args.pass_at_k 100 \
#     --trainer.gpus 8

export EXP_NAME="gsmath-2.7B-mml-mle_aug-eval_val"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export MASTER_PORT=56120
python trainer.py validate \
    --config training_configs/mathqa_gpt_mml_finetuning.yaml \
    --seed_everything 1 \
    --trainer.default_root_dir "$AMLT_OUTPUT_DIR/$EXP_NAME" \
    --data.init_args.train_file_path "$AMLT_DATA_DIR/gsmath_train.jsonl" \
    --data.init_args.val_file_path "$AMLT_DATA_DIR/gsmath_val.jsonl" \
    --model.init_args.load_ckpt_file "$AMLT_DATA_DIR/model_ckpts/gsmath-2.7B-mml-mle_aug-step=13223-exec_acc=0.2092-exec_rate=0.9980-fp32.ckpt" \
    --model.init_args.pass_at_k 100 \
    --trainer.gpus 8

export EXP_NAME="gsmath-2.7B-mml-mle_aug-eval_val-full"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export MASTER_PORT=56130
python trainer.py validate \
    --config training_configs/mathqa_gpt_mml_finetuning.yaml \
    --seed_everything 1 \
    --trainer.default_root_dir "$AMLT_OUTPUT_DIR/$EXP_NAME" \
    --data.init_args.train_file_path "$AMLT_DATA_DIR/gsmath_train.jsonl" \
    --data.init_args.val_file_path "$AMLT_DATA_DIR/gsmath_val.jsonl" \
    --model.init_args.load_ckpt_file "$AMLT_DATA_DIR/model_ckpts/gsmath-2.7B-mml-mle_aug-step=22271-exec_acc=0.2065-exec_rate=0.9993-fp32.ckpt" \
    --model.init_args.pass_at_k 100 \
    --trainer.gpus 8

export EXP_NAME="gsmath-2.7B-mml-mle_aug-eval_test"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export MASTER_PORT=56140
python trainer.py validate \
    --config training_configs/mathqa_gpt_mml_finetuning.yaml \
    --seed_everything 1 \
    --trainer.default_root_dir "$AMLT_OUTPUT_DIR/$EXP_NAME" \
    --data.init_args.train_file_path "$AMLT_DATA_DIR/gsmath_train.jsonl" \
    --data.init_args.val_file_path "$AMLT_DATA_DIR/gsmath_test.jsonl" \
    --model.init_args.load_ckpt_file "$AMLT_DATA_DIR/model_ckpts/gsmath-2.7B-mml-mle_aug-step=13223-exec_acc=0.2092-exec_rate=0.9980-fp32.ckpt" \
    --model.init_args.pass_at_k 100 \
    --trainer.gpus 8

export EXP_NAME="gsmath-2.7B-mml-mle_aug-eval_test-full"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export MASTER_PORT=56150
python trainer.py validate \
    --config training_configs/mathqa_gpt_mml_finetuning.yaml \
    --seed_everything 1 \
    --trainer.default_root_dir "$AMLT_OUTPUT_DIR/$EXP_NAME" \
    --data.init_args.train_file_path "$AMLT_DATA_DIR/gsmath_train.jsonl" \
    --data.init_args.val_file_path "$AMLT_DATA_DIR/gsmath_test.jsonl" \
    --model.init_args.load_ckpt_file "$AMLT_DATA_DIR/model_ckpts/gsmath-2.7B-mml-mle_aug-step=22271-exec_acc=0.2065-exec_rate=0.9993-fp32.ckpt" \
    --model.init_args.pass_at_k 100 \
    --trainer.gpus 8

sleep 1d
