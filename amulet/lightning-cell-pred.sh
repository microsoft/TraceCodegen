#!/bin/bash -x

source amulet/launch-util.sh

eval "$CODEGEN_PYTHON_CMD" trainer_nb.py \
    --config training_configs/nb_cell_codegen.yaml \
    --config amulet/lightning-cell-pred-overrides.yaml \
    --trainer.default_root_dir "$AMLT_OUTPUT_DIR" \
    --data.train_file_path "$AMLT_DATA_DIR/train_shards" \
    --data.val_file_path "$AMLT_DATA_DIR/val_shards" \
    "$@"
sleep 1d
