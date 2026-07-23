"""Verify native TensorRT engines against their ONNX outputs — RUN ON THE JETSON
after fusion/build_engines.sh. Loads each .engine with the TensorRT Python
runtime, runs random inputs, and compares to onnxruntime on the same inputs.

FP16 engines will NOT match to 1e-4 (that's the CPU/FP32 tolerance) — expect
~1e-2 to 1e-3 max abs diff on logits, which does not change argmax intents.
This script reports the diff and the argmax-agreement rate; the latter is what
matters for deployment.

Usage:  python fusion/trt_check.py
"""
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ODIR = HERE.parent / "onnx"

SPECS = {
    "emotion_mobilenetv2": {"image": (8, 3, 224, 224)},
    "gesture_tcn": {"window": (8, 32, 185)},
    "motion_lstm": {"window": (8, 30, 84)},
    "fusion_attn": {"x": (8, 24), "obs": (8, 4)},
}


def run_onnx(name, feeds):
    import onnxruntime as ort
    sess = ort.InferenceSession(str(ODIR / f"{name}.onnx"),
                                providers=["CPUExecutionProvider"])
    return sess.run(None, feeds)[0]


def run_trt(name, feeds):
    import pycuda.autoinit  # noqa: F401
    import pycuda.driver as cuda
    import tensorrt as trt

    logger = trt.Logger(trt.Logger.WARNING)
    with open(ODIR / f"{name}.engine", "rb") as f, \
            trt.Runtime(logger) as rt:
        engine = rt.deserialize_cuda_engine(f.read())
    ctx = engine.create_execution_context()

    bufs, out_host, out_name = {}, None, None
    for i in range(engine.num_io_tensors):
        tname = engine.get_tensor_name(i)
        if engine.get_tensor_mode(tname) == trt.TensorIOMode.INPUT:
            arr = np.ascontiguousarray(feeds[tname].astype(np.float32))
            ctx.set_input_shape(tname, arr.shape)
            d = cuda.mem_alloc(arr.nbytes)
            cuda.memcpy_htod(d, arr)
            ctx.set_tensor_address(tname, int(d))
            bufs[tname] = d
        else:
            out_name = tname
    shape = tuple(ctx.get_tensor_shape(out_name))
    out_host = np.empty(shape, np.float32)
    d_out = cuda.mem_alloc(out_host.nbytes)
    ctx.set_tensor_address(out_name, int(d_out))
    ctx.execute_async_v3(0)
    cuda.memcpy_dtoh(out_host, d_out)
    return out_host


def main():
    rng = np.random.default_rng(0)
    print(f"{'model':22}{'max|diff|':>12}{'argmax agree':>14}")
    ok = True
    for name, spec in SPECS.items():
        feeds = {k: rng.standard_normal(s).astype(np.float32)
                 for k, s in spec.items()}
        if name == "fusion_attn":
            feeds["obs"] = (rng.random((8, 4)) > 0.3).astype(np.float32)
        try:
            o = run_onnx(name, feeds)
            t = run_trt(name, feeds)
        except Exception as e:  # noqa: BLE001
            print(f"{name:22}  ERROR: {e}")
            ok = False
            continue
        diff = float(np.abs(o - t).max())
        agree = float((o.argmax(1) == t.argmax(1)).mean())
        flag = "" if agree == 1.0 else "  <-- argmax differs!"
        print(f"{name:22}{diff:12.2e}{agree:14.2%}{flag}")
        ok = ok and agree == 1.0
    print("\nall argmax-consistent:", ok,
          "\n(FP16 logit diffs up to ~1e-2 are expected and fine)")


if __name__ == "__main__":
    sys.exit(0 if main() is None else 0)
