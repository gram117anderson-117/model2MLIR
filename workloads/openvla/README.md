# OpenVLA-7b (vision-language-action)

OpenVLA: Llama-2 7B language model + DINOv2/SigLIP vision backbone (`openvla/openvla-7b`,
custom HF code). Capture unit: `forward(input_ids, pixel_values) -> logits` (the single
forward; `predict_action` wraps a generation loop that isn't export-friendly).

## Venv (dedicated)
```bash
mkdir -p /scratch/agustin/projects/openvla_capture && cd /scratch/agustin/projects/openvla_capture
uv venv --python 3.12
uv pip install 'transformers>=4.40' timm tokenizers accelerate pillow numpy
uv pip install xdsl structlog ml_dtypes
uv pip install -e /scratch/agustin/projects/model2MLIR --no-deps
# optional torch-mlir wheel (cp312/torch 2.x) for the accelerator path
```

## Run
```bash
cd /scratch/agustin/projects/openvla_capture
uv run --no-sync python -c "
import m2m, sys; sys.path.insert(0,'/scratch/agustin/projects/model2MLIR/workloads/openvla')
from loader import get_model_and_inputs
m,i=get_model_and_inputs(); r=m2m.convert(m,i,backend='fx_importer')
open('/scratch/agustin/projects/model2MLIR/workloads/openvla/openvla.mlir','w').write(r.mlir_text)
print(r.ok, r.mlir_text.count('linalg.'))"
```

## Status
Scaffold ready. Needs the dedicated venv + a ~14 GB weight download. Expect the FXImporter
backend (7B + custom vision stack will OOM torch-mlir, like the other VLAs). bf16 weights mean
some ops stay opaque until bf16 emission lands; everything else lowers to standard dialects.
