#!/bin/bash -x

USE_AMLT=true

if [ "$USE_AMLT" != true ]; then
  echo "exporting artificial AMLT env vars"

  AMLT_OUTPUT_DIR=pl-logger-debug
  AMLT_DATA_DIR=data/apps
fi

# export the neptune key and specify the output dir for lightning loggers
source amulet/export_keys.sh
export PL_LOG_DIR=${AMLT_OUTPUT_DIR}/logs

base_port="${TRACE_CODEGEN_DEBUG_BASE_PORT:-5678}"
rank="${LOCAL_RANK:-0}"
port=$((base_port + rank))

case "$TRACE_CODEGEN_DEBUG" in
  true|TRUE|1) CODEGEN_PYTHON_CMD="debugpy-run --no-wait -g -p :$port";;
  *)           CODEGEN_PYTHON_CMD="python";;
esac
export CODEGEN_PYTHON_CMD
