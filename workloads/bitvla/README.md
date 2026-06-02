# BitVLA (1-bit VLA, OpenVLA-OFT variant)

BitVLA = W1.58-A8 (BitNet) language model + SigLIP-L vision tower, fine-tuned
with OpenVLA-OFT for LIBERO. ~3B params. Repo: https://github.com/ustcwhy/BitVLA
(uses a VENDORED `transformers` fork that hardcodes `BitNetForCausalLM` +
`SiglipVisionModel` and adds W1.58-A8 `BitLinear` + a `use_bi_attn` kwarg).

Capture unit: the inner bi-directional VLM forward on already-built
`inputs_embeds` -> logits (the call `predict_action` makes after host-side
embedding assembly). Carries the full BitNet LM + BitLinear quant math. Vision
encoding, `masked_scatter` of image/proprio tokens, and numpy action
unnormalization stay host-side (out of the graph).

Source model class: `BitVLAForActionPrediction(LlavaForConditionalGeneration)`
in `openvla-oft/bitvla/model/bitvla_for_action_prediction.py`.

The loader builds a SMALL random Llava+BitNet config — no weights downloaded.

## PyTorch-exportable?
**Maybe.** The captured inner forward is torch.export-friendly in principle
(BitLinear is pure `round/clamp/abs`-mean quant via `autograd.Function`, traced
through each `.forward`; bi-directional attention is a static mask). Risks:
- depends on the vendored `transformers` fork being installed (eager attn path).
- The full `predict_action` is NOT exportable (`.item()`, masked_scatter,
  numpy, image processor) — kept host-side by design.

## Venv (dedicated)
```bash
cd /scratch/agustin/projects/bitvla_capture
uv venv --python 3.10
# Install the VENDORED transformers fork (editable) + OFT deps:
uv pip install -e /scratch/agustin/projects/BitVLA/transformers
uv pip install numpy timm tokenizers accelerate
# m2m + torch-mlir:
uv pip install xdsl structlog ml_dtypes
uv pip install --pre torch-mlir \
  --extra-index-url https://download.pytorch.org/whl/nightly/cpu \
  -f https://github.com/llvm/torch-mlir-release/releases/expanded_assets/dev-wheels
uv pip install -e /scratch/agustin/projects/model2MLIR --no-deps
```
Note: `bitvla_for_action_prediction` imports `prismatic.vla.constants` and
`prismatic.training.train_utils` at MODULE LOAD time (the loader adds
`openvla-oft/` to `sys.path`). Importing the `prismatic` package runs
`prismatic/__init__.py` -> `from .models import ... load`, which can pull heavy
deps (timm, draccus, etc.). If install is too heavy, the lean fix is to stub the
two symbols the module actually needs before import:
`prismatic.vla.constants.{ACTION_DIM,NUM_ACTIONS_CHUNK,ACTION_PROPRIO_NORMALIZATION_TYPE,NormalizationType}`
and `prismatic.training.train_utils.{get_current_action_mask,get_next_actions_mask}`
— none are exercised by the captured inner-VLM forward.

## Run
```bash
cd /scratch/agustin/projects/bitvla_capture
BITVLA_LLM_LAYERS=2 uv run --no-sync python -c "
import m2m, sys; sys.path.insert(0,'/scratch/agustin/projects/model2MLIR/workloads/bitvla')
from loader import get_model_and_inputs
m,i=get_model_and_inputs(); r=m2m.convert(m,i)
open('/scratch/agustin/projects/model2MLIR/workloads/bitvla/bitvla.mlir','w').write(r.mlir_text)
print(r.path_taken, r.mlir_text.count('linalg.'))"
```

## Status
- Scaffold only; not yet captured. Validate the vendored-fork install + that
  `LlavaForConditionalGeneration.forward(..., use_bi_attn=True)` traces cleanly.
