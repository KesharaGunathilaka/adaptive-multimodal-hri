# Jetson Deployment Benchmark Report

## 1. Deployment Summary
- Backend: gbt
- Runtime window: 30.05 s
- Frames processed: 282
- End-to-end throughput: 9.38 FPS
- Mean per-frame latency: 98.51 ms
- RSS memory: min=623.87 MB, mean=727.30 MB, max=740.26 MB
- Live demo flags: `--holistic-complexity 0 --context-every 5 --holistic-every 3`

## 2. Time-to-Intent (whole-flow latency)

Time for one full pass -- camera frame -> landmarks -> emotion -> gesture -> motion -> (context, if scheduled) -> fusion -- measured only on the iterations that actually produced/refreshed an INTENT value (most frames only update the rolling feature window and don't count here).

| Metric | Value |
|---|---:|
| Mean | 69.7 ms |
| Median | 61.0 ms |
| P95 | 79.0 ms |
| Min | 37.5 ms |
| Max | 297.5 ms |
| Under 100ms | 46/48 (96%) |

- Wall-clock gap between successive INTENT updates: mean 632.0 ms, median 642.4 ms. This is gated by `--fusion-hz` (recompute-fusion rate), independent of how fast the compute path above is -- a fast compute pass still waits for the next fusion tick before a new INTENT is shown.

## 3. Stage Latency Breakdown

| Stage | Mean latency | P95 latency | Max latency |
|---|---:|---:|---:|
| emotion | 8.82 ms | 11.30 ms | 20.32 ms |
| gesture | 3.34 ms | 5.28 ms | 14.11 ms |
| motion | 10.31 ms | 19.29 ms | 104.48 ms |
| context | 15.35 ms | 28.06 ms | 36.34 ms |
| fusion | 10.27 ms | 12.05 ms | 12.73 ms |

## 4. Raw Per-Intent Export

- Raw per-frame / per-intent CSV: /home/udayanga/Desktop/HRI/hri-jeston-V2/jetson_deploy/reports/jetson_deployment_report.raw.csv
  (includes `intent_latency_ms` / `since_last_intent_ms` -- see §2)
- Intent summary CSV: /home/udayanga/Desktop/HRI/hri-jeston-V2/jetson_deploy/reports/jetson_deployment_report.intent_summary.csv
- Raw tegrastats samples (one line per ~500ms during the run): /home/udayanga/Desktop/HRI/hri-jeston-V2/jetson_deploy/reports/jetson_deployment_report.tegrastats.raw.txt
- Open the CSV files in Excel for the per-intent deployment sheet.

## 5. Model Size Summary

| Artifact | Size (MB) |
|---|---:|
| emotion_mobilenetv2.onnx | 8.49 |
| gesture_tcn.onnx | 2.60 |
| motion_lstm.onnx | 5.43 |
| context_efficientnet_b0.onnx | 15.29 |
| emotion_mobilenetv2.engine | 4.73 |
| gesture_tcn.engine | 1.44 |
| motion_lstm.engine | 4.87 |
| context_efficientnet_b0.engine | 9.33 |

## 6. Power / Thermal / GPU Telemetry

- Samples: 71 (`tegrastats --interval 500`, captured *during* the live_demo.py run above, not before/after it)

| Metric | Mean | Min | Max |
|---|---:|---:|---:|
| Board power (VDD_IN, instantaneous) | 5124.1 mW | 4618.0 mW | 5694.0 mW |
| Board power (VDD_IN, running avg) | 5283.4 mW | 5124.0 mW | 5522.0 mW |
| GPU utilization (GR3D_FREQ) | 29.3% | 6.0% | 87.0% |
| GPU temp | 49.4°C | 49.0°C | 49.7°C |
| CPU temp | 49.3°C | 49.2°C | 49.6°C |
| RAM used | 4814.6 MB | 4354.0 MB | 4884.0 MB |
- `nvidia-smi` sample: Orin (nvgpu), [N/A], [N/A], [N/A], [N/A] (expected `[N/A]` on Jetson's integrated GPU -- it isn't a discrete `nvidia-smi`-managed card)

## 7. Accuracy vs Latency Notes

- This report captures runtime performance on the actual Jetson deployment run.
- Accuracy values should be filled from your validation/test set or the same dataset used in the training repo; the live demo itself only reports top-label confidence, not benchmark accuracy.
- Latency reduction knobs available: `--holistic-complexity 0`, `--context-every N`, `--holistic-every N` (frame-skip the pose model, reuses last landmarks in between -- see §2/§3 for the latency this run got with it), `--fusion-hz` (INTENT refresh rate), and TensorRT engines.

## 8. Optimization / Quantization Notes

- TensorRT engine artifacts are present, so the deployment can run with the optimized TensorRT execution path.
- The live pipeline is already validated to use TensorRT when a matching `.engine` file exists, otherwise it falls back transparently to ONNX Runtime.

## 9. Run Command
```bash
cd jetson_deploy
source .venv-runtime/bin/activate
python benchmark_report.py --run-seconds 30 --backend gbt --output reports/jetson_deployment_report.md --holistic-complexity 0 --context-every 5 --holistic-every 3
```
