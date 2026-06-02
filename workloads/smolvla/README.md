# smolVLA (lerobot)

smolVLA `VLAFlowMatching`: SmolVLM2-500M backbone + action expert (~0.45B params).
Capture unit: one flow-matching `denoise_step` (prefix pass + one expert pass), with one camera.

## Venv (dedicated)
```bash
cd /scratch/agustin/projects/smolvla_capture
uv venv --python 3.12
uv pip install 'lerobot[smolvla]==0.5.1'
uv pip install xdsl structlog ml_dtypes
uv pip install --pre torch-mlir \
  --extra-index-url https://download.pytorch.org/whl/nightly/cpu \
  -f https://github.com/llvm/torch-mlir-release/releases/expanded_assets/dev-wheels
uv pip install -e /scratch/agustin/projects/model2MLIR --no-deps
```

## Run
```bash
cd /scratch/agustin/projects/smolvla_capture
uv run --no-sync python -c "
import model2mlir, sys; sys.path.insert(0,'/scratch/agustin/projects/model2MLIR/workloads/smolvla')
from loader import get_model_and_inputs
m,i=get_model_and_inputs(); r=model2mlir.convert(m,i)
open('/scratch/agustin/projects/model2MLIR/workloads/smolvla/smolvla.mlir','w').write(r.mlir_text)
print(r.path_taken, r.mlir_text.count('linalg.'))"
```

## Status
- fp32: captured → 605 `linalg` (FXImporter mode; 1.26 MB). torch-mlir is installed but aborts
  on `aten.empty_permuted` (introduced by decomposition); a torch-mlir-friendly decomposition
  (`empty_permuted → empty`) is the proper fix and is the next step.
- int8/fp8: pending.
