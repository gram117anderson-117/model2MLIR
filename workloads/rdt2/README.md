# RDT2 (thu-ml/RDT2, PyTorch)

RDT2's `RDT` (`models.rdt.model.RDT`) is a flow-matching ACTION EXPERT: a DiT-style
stack (adaLN-Zero modulation + GQA self-attn + GQA cross-attn + SwiGLU FFN) that
cross-attends to the KV cache of a frozen **Qwen2.5-VL-7B-Instruct** VLM. Default
config = hidden 1024, depth 14, heads 8, kv_heads 4 (`configs/rdt/post_train.yaml`).
Capture unit: ONE flow-matching step (`RDT.forward`, called in the Euler ODE loop of
`RDTRunner.conditional_sample`), random init, no checkpoint download.

The Qwen2.5-VL forward (image+text -> per-layer KV cache) and the act/state adaptor
MLPs run host-side, OUTSIDE the captured graph. Their outputs enter as: adapted
noisy-action tokens `x` (B, horizon, D), adapted state token `state_c` (B, 1, D),
and the per-block language KV cache `lang_c_kv` = list of (k, v), one pair per RDT
block, each `(B, seq_len, kv_heads, head_size)`. The 5-step loop is NOT captured.

## Venv (dedicated)
RDT2 pins torch + transformers 4.51.3 + diffusers 0.35.1 + timm 1.0.15 (+ vllm/peft/
bitsandbytes for the full VLA stack, NOT needed for the RDT core). To instantiate the
action expert you only need torch + timm (`Mlp`) + numpy. Capturing the full VLA
(Qwen2.5-VL-7B) would need the heavy stack + a ~15GB checkpoint — out of scope here.

```bash
git clone https://github.com/thu-ml/RDT2 /scratch/agustin/projects/RDT2
cd /scratch/agustin/projects/RDT2
uv venv --python 3.10
uv pip install torch timm==1.0.15 numpy==1.26.4
# frontend (FXImporter mode; a matching cp310 torch-mlir wheel can be added too)
uv pip install xdsl structlog ml_dtypes
uv pip install -e /scratch/agustin/projects/model2MLIR --no-deps
```

## Run
```bash
cd /scratch/agustin/projects/RDT2
M2M_RDT2_DEPTH=2 uv run --no-sync python -c "
import m2m, sys; sys.path.insert(0,'/scratch/agustin/projects/model2MLIR/workloads/rdt2')
from loader import get_model_and_inputs
m,i=get_model_and_inputs(); r=m2m.convert(m,i)
open('/scratch/agustin/projects/model2MLIR/workloads/rdt2/rdt2.mlir','w').write(r.mlir_text)
print(r.path_taken, r.mlir_text.count('linalg.'))"
```
Env: `M2M_RDT2_DEPTH=N` (number of RDT blocks; default 2 smoke, real default = 14).

## Feasibility
- PyTorch, torch.export-able: MAYBE (very likely yes for the action expert).
  `RDT.forward` is pure tensor ops (Linear, RMSNorm, SiLU/SwiGLU, SDPA, adaLN chunk),
  BUT it has data-dependent Python control flow that must resolve at trace time:
    * `conds = [lang_c_kv or lang_c]`, then per block `if isinstance(c, List): ...`
      `elif c.dim() == 4: ...` — selecting the cache vs. dense-condition path.
    * `ck, cv = c[i % len(c)]` indexes a Python list per block.
  These are constant-foldable under torch.export FOR A FIXED input choice; this loader
  pins the `lang_c_kv` (KV-cache) path used at deployment, which unrolls cleanly.
  Set `use_flash_attn=False` (done here) so attention is plain matmul/SDPA, not a
  flash-attn custom op. The full VLA (with Qwen2.5-VL-7B + vLLM) is NOT export-friendly
  and is intentionally excluded — we capture only the diffusion action expert.
- Note: the manual (`use_flash_attn=False`) CrossAttention masked path has an upstream
  bug (`attn` referenced before assignment) when `mask is not None`; we pass mask=None,
  so it is not hit.

## Status
- fp32: scaffolded, not yet captured (depth=2 smoke recommended first).
- int8/fp8: pending.
