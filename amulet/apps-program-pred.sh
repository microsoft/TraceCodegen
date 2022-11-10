#!/bin/bash -x

source amulet/launch-util.sh

eval "$CODEGEN_PYTHON_CMD" trainer_apps.py fit \
    --config training_configs/apps_program_codegen.yaml \
    --config amulet/apps-program-pred-overrides.yaml \
    --trainer.default_root_dir "$AMLT_OUTPUT_DIR" \
    --data.train_file_path "$AMLT_DATA_DIR/train.jsonl" \
    --data.val_file_path "$AMLT_DATA_DIR/val.jsonl" \
    "$@"
sleep 1d
