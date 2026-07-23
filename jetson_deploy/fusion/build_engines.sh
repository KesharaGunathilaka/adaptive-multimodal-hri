#!/usr/bin/env bash
# Build native TensorRT engines from the ONNX files — RUN THIS ON THE JETSON.
#
# Engines are hardware+TensorRT-version specific: they CANNOT be built on a PC
# and copied. Run on the Orin Nano itself. Output: onnx/*.engine (FP16).
#
# This is Route B (native trtexec) — max performance + reusable .engine files.
# Route A (ONNX Runtime TensorRT EP, zero build step) is simpler: just run
#   python fusion/pipeline.py --backend tensorrt
# See ../TENSORRT_GUIDE.md for when to use which.
#
# Usage:  bash fusion/build_engines.sh
set -euo pipefail

# trtexec ships with TensorRT; on JetPack it's here (adjust if your PATH differs)
TRTEXEC="${TRTEXEC:-/usr/src/tensorrt/bin/trtexec}"
if ! command -v "$TRTEXEC" >/dev/null 2>&1 && [ ! -x "$TRTEXEC" ]; then
  TRTEXEC="trtexec"   # fall back to PATH
fi

ONNX_DIR="$(cd "$(dirname "$0")/.." && pwd)/onnx"
echo "trtexec : $TRTEXEC"
echo "onnx dir: $ONNX_DIR"
"$TRTEXEC" --version || { echo "trtexec not found — is TensorRT installed?"; exit 1; }

# All 4 nets have a dynamic batch axis (exported with dynamic_axes). For the
# streaming pipeline batch is always 1, so we build fixed batch=1 engines
# (fastest, no optimization-profile juggling). --fp16 is the Jetson default.
build () {
  local name="$1" ; shift
  echo "── building ${name} ─────────────────────────────────────────"
  "$TRTEXEC" \
    --onnx="${ONNX_DIR}/${name}.onnx" \
    --saveEngine="${ONNX_DIR}/${name}.engine" \
    --fp16 \
    "$@" \
    --skipInference=false --noDataTransfers=false 2>&1 | \
    grep -E "Throughput|GPU Compute Time|mean|median|Engine built|error|Error" || true
}

build emotion_mobilenetv2 --shapes=image:1x3x224x224
build gesture_tcn         --shapes=window:1x32x185
build motion_lstm         --shapes=window:1x30x84
build fusion_attn         --shapes=x:1x24,obs:1x4

echo
echo "done. engines:"
ls -la "${ONNX_DIR}"/*.engine
echo
echo "verify each vs ONNX with:  python fusion/trt_check.py"
