"""Trial ONNX export (handover §9, Week-1 de-risk): emotion CNN, gesture TCN,
motion LSTM, fusion attention head -> jetson_deploy/onnx/*.onnx, each verified
against the native PyTorch output (max |diff| on logits, tolerance 1e-4) on
random inputs plus real samples where available.

Context/CLIP is intentionally NOT exported: it runs from the bundled HF cache
in PyTorch on the Jetson (see MODEL_AUDIT.md §6); revisit only if latency
profiling demands TensorRT.

ENV NOTE: onnx must be ==1.14.1 with protobuf==3.20.3 — newer protobuf breaks
mediapipe (WORKLOG 2026-07-20).

Run from repo root:  .venv/Scripts/python scripts/08_export_onnx.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fusion.baselines import common  # noqa: E402
from fusion.extraction.modloader import load_module  # noqa: E402
from fusion.model.model import AttentionFusion  # noqa: E402

OUT = ROOT / "jetson_deploy" / "onnx"
OPSET = 17
TOL = 1e-4


def verify(onnx_path, torch_model, inputs):
    import onnxruntime as ort
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    feeds = {i.name: t.numpy() for i, t in zip(sess.get_inputs(), inputs)}
    ort_out = sess.run(None, feeds)[0]
    with torch.no_grad():
        native = torch_model(*inputs).numpy()
    diff = float(np.abs(native - ort_out).max())
    return diff


def export(name, model, inputs, input_names, dynamic_batch=True):
    model = model.cpu().eval()
    path = OUT / f"{name}.onnx"
    dyn = ({n: {0: "batch"} for n in input_names} if dynamic_batch else None)
    torch.onnx.export(model, tuple(inputs), str(path), opset_version=OPSET,
                      input_names=input_names, output_names=["logits"],
                      dynamic_axes=dyn)
    diff = verify(path, model, inputs)
    status = "OK" if diff < TOL else f"FAIL (diff {diff:.2e})"
    print(f"{name:14} -> {path.name:20} maxdiff={diff:.2e}  {status}")
    return {"file": path.name, "opset": OPSET, "max_abs_diff": diff,
            "pass": bool(diff < TOL)}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    report = {}
    torch.manual_seed(0)

    # ── Emotion MobileNetV2 ────────────────────────────────────────────
    emo = load_module("hri_emo_export",
                      ROOT / "modalities" / "emotion" / "inference" / "video.py")
    m = emo.build_model()
    m.load_state_dict(torch.load(
        ROOT / "modalities" / "emotion" / "checkpoints" / "finetuned_MobileNetV2.pth",
        map_location="cpu", weights_only=True))
    report["emotion"] = export("emotion_mobilenetv2", m,
                               [torch.randn(2, 3, 224, 224)], ["image"])

    # ── Gesture TCN ────────────────────────────────────────────────────
    ges_dir = ROOT / "modalities" / "gesture"
    gm = load_module("hri_ges_export", ges_dir / "src" / "models.py", [ges_dir])
    cfg = json.loads((ges_dir / "checkpoints" / "model_config.json").read_text())
    g = gm.build_model(cfg["model"], **cfg.get("model_kwargs", {}))
    g.load_state_dict(torch.load(ges_dir / "checkpoints" / "best_TCN.pth",
                                 map_location="cpu", weights_only=True))
    report["gesture"] = export("gesture_tcn", g,
                               [torch.randn(2, 32, 185)], ["window"])

    # ── Motion LSTM ────────────────────────────────────────────────────
    mot_dir = ROOT / "modalities" / "motion"
    mi = load_module("hri_mot_export", mot_dir / "src" / "inference.py",
                     [mot_dir / "src"])
    ckpt = torch.load(mot_dir / "checkpoints" / "best_model_finetuned.pt",
                      map_location="cpu", weights_only=True)
    mcfg = ckpt.get("config", {})
    mo = mi.MotionLSTM(hidden_size=mcfg.get("hidden_size", 256),
                       num_layers=mcfg.get("num_layers", 3),
                       dropout=mcfg.get("dropout", 0.35))
    mo.load_state_dict(ckpt["model_state_dict"])
    report["motion"] = export("motion_lstm", mo,
                              [torch.randn(2, 30, 84)], ["window"])

    # ── Fusion attention (exclude-mode, real feature samples) ──────────
    f = AttentionFusion(missing_mode="exclude")
    f.load_state_dict(torch.load(ROOT / "jetson_deploy" / "fusion" / "fusion_attn.pt",
                                 map_location="cpu", weights_only=True))
    f.encoder.enable_nested_tensor = False        # nested-tensor path blocks export
    if hasattr(f.encoder, "use_nested_tensor"):
        f.encoder.use_nested_tensor = False
    df = pd.read_parquet(common.FEATURES).head(64)
    x = torch.from_numpy(df[common.PROB_COLS].fillna(0).to_numpy(np.float32))
    obs = torch.from_numpy(df[common.OBS_COLS].to_numpy(np.float32))
    obs[5:10, 1] = 0.0                             # exercise the masked path
    x[5:10, 7:15] = 0.0
    report["fusion"] = export("fusion_attn", f, [x, obs], ["x", "obs"])

    (OUT / "export_report.json").write_text(json.dumps(report, indent=2))
    ok = all(r["pass"] for r in report.values())
    print(f"\nall passed: {ok}  -> {OUT / 'export_report.json'}")


if __name__ == "__main__":
    main()
