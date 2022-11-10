#!/bin/bash -x

# --model.init_args.load_ckpt_file "$AMLT_DATA_DIR/model_ckpts/step=6904-exec_acc=0.4480-exec_rate=0.6445.ckpt" \

python trainer.py \
    --config training_configs/mathqa_gpt_partial_mml_finetuning.yaml \
    --trainer.default_root_dir "$AMLT_OUTPUT_DIR" \
    --data.init_args.train_file_path "$AMLT_DATA_DIR/gsmath_train.jsonl" \
    --data.init_args.val_file_path "$AMLT_DATA_DIR/gsmath_val.jsonl" \
    "$@"
sleep 1d
