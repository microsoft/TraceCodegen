#!/bin/bash -x

# --model.init_args.load_ckpt_file "$AMLT_DATA_DIR/model_ckpts/step=22818-exec_acc=0.5156-exec_rate=0.7236.ckpt" \
# --model.init_args.load_ckpt_file "$AMLT_DATA_DIR/model_ckpts/mle_best_pass_80.ckpt" \

python trainer.py \
    --config training_configs/mathqa_gpt_finetuning.yaml \
    --trainer.default_root_dir "$AMLT_OUTPUT_DIR" \
    --data.init_args.train_file_path "$AMLT_DATA_DIR/train_dedup.jsonl" \
    --data.init_args.val_file_path "$AMLT_DATA_DIR/val_dedup.jsonl" \
    "$@"
sleep 1d
