#!/bin/bash -x

python trainer.py validate \
    --config training_configs/mathqa_gpt_prompting.yaml \
    --trainer.default_root_dir "$AMLT_OUTPUT_DIR" \
    --data.init_args.val_file_path "$AMLT_DATA_DIR/val_python_with_states.jsonl" \
    "$@"
sleep 1d
