# AllenAI MolmoAct (PyTorch)

Molmo-style VLA: SigLIP2 ViT vision backbone + adapter + Qwen2-style LLM decoder that
autoregressively emits reasoning / depth / action tokens, then decodes action bins.

## Which repo
The task pointed at https://github.com/allenai/molmoact2, but that repo is **only a
FastAPI inference server** (`molmoact2-policy-server`): it loads the checkpoint from the
HF Hub via `AutoModelForImageTextToText.from_pretrained("allenai/MolmoAct2-DROID",
trust_remote_code=True)` and a hub-side `modeling_molmoact2.py`. No model class lives in
that repo (only example clients/servers for DROID and bimanual YAM robots, plus lerobot/
YAM/EVA_DROID git submodules, none fetched). MolmoAct2 uses a flow-matching action head.

The full PyTorch model definition lives in https://github.com/allenai/molmoact (the `olmo`
training package), cloned at /scratch/agustin/projects/molmoact. This loader builds against
that — the autoregressive MolmoAct (v1) lineage, same ViT+LLM backbone.

Model classes (in `olmo/hf_model/molmoact/modeling_molmoact.py`):
- `MolmoActForActionReasoning(MolmoActConfig)` (PreTrainedModel, GenerationMixin) — full VLA;
  action output goes through `.generate(...)` (a loop, NOT capturable as one graph).
- `MolmoActForCausalLM(MolmoActLlmConfig)` — the LLM decoder, plain `input_ids -> logits`
  causal LM. **This is the capture unit** (tiny_llama style).
- `MolmoActVisionBackbone` — SigLIP2 ViT (separate capturable unit; not wrapped here).
- Configs (`configuration_molmoact.py`): `MolmoActConfig` (vit+adapter+llm), `MolmoActLlmConfig`
  (hidden 3584, 48 layers, 28 heads / 4 kv heads, head_dim 128, vocab 152064, rope_theta 1e6).

Capture unit: `MolmoActForCausalLM.forward(input_ids) -> logits` — the representative heavy
transformer, avoiding the generation loop, ViT preprocessing, and image-token splicing
(all data-dependent / host-side).

## PyTorch-exportable?
Yes (maybe, in practice) for the LLM causal-LM forward: it is a standard transformers-style
decoder (`wte -> rotary -> decoder layers -> RMSNorm -> lm_head`), tensor->tensor, built from
a config with random init. Caveats for torch.export: uses HF `DynamicCache`/`_update_causal_mask`
and `transformers` SDPA — set `use_cache=False` and `_attn_implementation="eager"` (done in the
loader), same handling as tiny_llama / pi0.5. The full `MolmoActForActionReasoning` is NOT
directly exportable (autoregressive `.generate` loop + image splicing control flow).

## Venv (dedicated)
The `olmo` package targets recent transformers; the model file imports `GradientCheckpointingLayer`,
`Cache`/`DynamicCache`, `FlashAttentionKwargs`, `ROPE_INIT_FUNCTIONS` from transformers, so a
modern transformers (>=4.51) is needed. The repo's own pyproject is heavy (full training stack);
for capture, only `olmo/hf_model/molmoact/` + transformers + torch are required. The loader puts
the repo on `sys.path` (no `pip install -e` of olmo needed) via `MOLMOACT_REPO`.

```bash
# molmoact already cloned (shallow, LFS skipped):
#   GIT_LFS_SKIP_SMUDGE=1 git clone --depth 1 https://github.com/allenai/molmoact /scratch/agustin/projects/molmoact
uv venv --python 3.11 --directory /scratch/agustin/projects/model2MLIR/workloads/molmoact
# minimal capture deps:
uv pip install "torch==2.7.*" "transformers>=4.51" einops numpy
# frontend (FXImporter mode):
uv pip install xdsl structlog ml_dtypes
uv pip install -e /scratch/agustin/projects/model2MLIR --no-deps
```

## Run
```bash
MOLMOACT_REPO=/scratch/agustin/projects/molmoact \
M2M_MOLMOACT_LAYERS=4 M2M_MOLMOACT_VOCAB=4096 \
uv run --no-sync python -c "
import m2m, sys; sys.path.insert(0,'/scratch/agustin/projects/model2MLIR/workloads/molmoact')
from loader import get_model_and_inputs
m,i=get_model_and_inputs(); r=m2m.convert(m,i)
open('/scratch/agustin/projects/model2MLIR/workloads/molmoact/molmoact.mlir','w').write(r.mlir_text)
print(r.path_taken, r.mlir_text.count('linalg.'))"
```

## Status
- Scaffold only — not yet captured. Random-init LLM decoder, fp32, vocab shrunk by default
  (full 152064 vocab is a ~2 GB fp32 embedding + lm_head).
- Open items: confirm transformers version vs the model file's imports
  (`GradientCheckpointingLayer`, `ROPE_INIT_FUNCTIONS`); the custom `qk_norm`/`rope_scaling`
  paths may need to stay on the eager attention path for a clean trace.
- The ViT backbone (`MolmoActVisionBackbone`) and the MolmoAct2 flow-matching head are
  separate future capture units.
- int8/fp8: pending.
