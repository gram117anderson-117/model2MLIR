# LSTMNetVIT (vitfly)

Vision-based quadrotor obstacle avoidance: a depth image in, a 3-DoF velocity command out.
Repo: https://github.com/anish-bhattacharya/vitfly (`models/model.py`), 3.56 M params.

Structure: a two-stage SegFormer-style Mix-Transformer encoder over a 60×90 depth image
(overlapping patch-embed convs, efficient self-attention with spatial reduction, a
depthwise-conv MixFFN), a `PixelShuffle` + bilinear-upsample fusion of the two feature scales,
a spectral-norm Linear decoder, then a **3-layer LSTM** head.

Capture unit: ONE control step, output tensor only.

## Scope limits, stated up front

- **The LSTM hidden state is dropped.** The published forward returns `(command, hidden)` and the
  real controller feeds `hidden` back each step. The capture contract is one output tensor, so
  this is one step from a ZERO initial state: it exercises the recurrence arithmetic
  (`torch.export` unrolls the LSTM into per-timestep gate matmuls) without proving multi-step
  recurrent behaviour.
- **Random init.** vitfly publishes its trained models as a password-protected
  `pretrained_models.tar` on Box, not in the repo, so the golden checks lowering exactness rather
  than flight accuracy. Point `VITFLY_CKPT` at an unpacked LSTMNetVIT state_dict to change that.

## Input shapes (the published forward is easy to get wrong)

`forward([depth, desired_velocity, quaternion])` with `depth (N,1,60,90)`,
`desired_velocity (N,1)`, `quaternion (N,4)`. The desired velocity is **one** element: the LSTM's
`input_size=517` is `512 + 1 + 4`, so passing a 5-vector makes the concatenation 521 wide and the
model raises.

## Venv
The main `model2MLIR/.venv`. No extra deps.

## Run
```bash
cd /scratch/agustin/projects/model2MLIR
python workloads/capture.py lstmnetvit --formats fp32,int8
.venv/bin/python workloads/capture_consistent.py lstmnetvit int8 \
    <merlin>/out/artifacts/recaptures/lstmnetvit_int8_full
```
Env: `VITFLY_DIR`, `VITFLY_CKPT`.

## Status
- fp32: 653 `linalg`, **0 opaque**. Verified end to end against the torch golden through
  merlin's host dispatch runtime: **cos = 1.000000000, rel = 2.8e-07**.
- int8: 653 `linalg`, **0 opaque**.

## Op-set notes
- The two `spectral_norm` Linear layers are folded back into plain `weight` before export. In
  eval mode that value is fixed, so folding is exact — and it is required, because torchAO's
  `quantize_` swaps `weight` in place and raises when a reparametrization owns the attribute.
- Needed the padded / depthwise / rank-3 conv work, and bilinear upsample, which lowers as two
  resize contractions against constant weight matrices (every size is static, so the
  interpolation weights are compile-time data).
- `PixelShuffle` and the unrolled LSTM already lowered.
