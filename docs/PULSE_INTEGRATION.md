# PULSE description augmentation for MedTsLLM

This integration implements the first, lowest-risk PULSE experiment:

```text
PTB-XL signal ────────────────> MedTsLLM signal branch ──> classifier
       └─> 12-lead ECG image ─> frozen PULSE-7B ─────────> feature-only text prompt
```

PULSE is run **offline in its own environment**. MedTsLLM only loads the resulting JSONL descriptions, so the incompatible PULSE/LLaVA and MedTsLLM package versions do not need to coexist in one Python environment. The official PULSE repository likewise exposes inference through its LLaVA code and `PULSE-ECG/PULSE-7B` checkpoint. PULSE itself is based on LLaVA-v1.6/Vicuna and was trained for ECG-image interpretation.

## Files added

- `tools/pulse/render_ptbxl_images.py`: renders a deterministic 4x3 12-lead image plus lead-II rhythm strip.
- `tools/pulse/generate_pulse_descriptions.py`: loads PULSE once, generates structured feature-only JSON, and writes resumable JSONL.
- `tools/pulse/validate_pulse_descriptions.py`: verifies coverage, generation failures, duplicates, and obvious class-label leakage.
- `configs/datasets/ptbxl_pulse.toml`: PULSE descriptions with BiomedCoOp disabled, for a clean PULSE-only ablation.
- `configs/datasets/ptbxl_pulse_biomedcoop.toml`: PULSE descriptions plus the existing BiomedCoOp head.
- `requirements-pulse-preprocess.txt`: extra packages for rendering PTB-XL images.

`datasets/ptbxl.py` now optionally reads PULSE descriptions. No change to `models/medtsllm.py` is required because the existing prompt builder already inserts each sample's `descriptions` field when `[models.medtsllm.prompting].clip = true`.

## 1. Prepare the MedTsLLM environment

From the `medtsllm5` repository:

```bash
source .venv/bin/activate
pip install -r requirements.txt
pip install -r requirements-pulse-preprocess.txt
```

PTB-XL must be located at:

```text
data/ptbxl/
├── ptbxl_database.csv
├── scp_statements.csv
└── records100/...
```

The data root can be changed through `root` under `[datasets.PTB-XL]`.

## 2. Render the ECG images

```bash
python tools/pulse/render_ptbxl_images.py \
  --ptbxl-root data/ptbxl \
  --output-dir data/ptbxl/pulse_images \
  --manifest data/ptbxl/pulse_images/manifest.jsonl \
  --split all
```

The script applies the same single-superclass filtering and PTB-XL folds as `datasets/ptbxl.py`:

- train: folds 1–8
- validation: fold 9
- test: fold 10

The manifest contains labels only for auditing. Labels are never printed into images or sent in the PULSE prompt.

For a smoke test:

```bash
python tools/pulse/render_ptbxl_images.py \
  --ptbxl-root data/ptbxl \
  --output-dir data/ptbxl/pulse_images_smoke \
  --manifest data/ptbxl/pulse_images_smoke/manifest.jsonl \
  --split test \
  --limit 8 \
  --overwrite
```

## 3. Install PULSE in a separate Conda environment

```bash
git clone https://github.com/AIMedLab/PULSE.git
cd PULSE/LLaVA
conda create -n pulse-llava python=3.10 -y
conda activate pulse-llava
pip install -e ".[train]"
pip install flash-attn --no-build-isolation
```

The PULSE repository's default branch currently used by the project is `dev`. Record the exact commit used in your experiment.

## 4. Generate feature-only descriptions

Run the following from `medtsllm5`, while the `pulse-llava` environment is active:

```bash
python tools/pulse/generate_pulse_descriptions.py \
  --pulse-root /absolute/path/to/PULSE/LLaVA \
  --manifest data/ptbxl/pulse_images/manifest.jsonl \
  --output data/ptbxl/pulse_descriptions/pulse_feature_only.jsonl \
  --model-path PULSE-ECG/PULSE-7B
```

The output is appended after every record and completed ECG IDs are skipped on restart.

For lower-memory inference, try one of the following, subject to the PULSE/LLaVA environment supporting bitsandbytes correctly:

```bash
# 4-bit
python tools/pulse/generate_pulse_descriptions.py ... --load-4bit

# 8-bit
python tools/pulse/generate_pulse_descriptions.py ... --load-8bit
```

### Multi-GPU sharding

Launch one process per GPU with a different rank. Each process should write a separate file:

```bash
CUDA_VISIBLE_DEVICES=0 python tools/pulse/generate_pulse_descriptions.py \
  --pulse-root /path/to/PULSE/LLaVA \
  --manifest data/ptbxl/pulse_images/manifest.jsonl \
  --output data/ptbxl/pulse_descriptions/part-0.jsonl \
  --rank 0 --world-size 4
```

Repeat for ranks 1–3, then merge:

```bash
cat data/ptbxl/pulse_descriptions/part-*.jsonl \
  > data/ptbxl/pulse_descriptions/pulse_feature_only.jsonl
```

## 5. Validate before training

```bash
python tools/pulse/validate_pulse_descriptions.py \
  --manifest data/ptbxl/pulse_images/manifest.jsonl \
  --descriptions data/ptbxl/pulse_descriptions/pulse_feature_only.jsonl \
  --summary-json data/ptbxl/pulse_descriptions/validation_summary.json \
  --strict
```

Do not train with `strict = true` until missing records and generation errors are resolved.

## 6. Train the ablations

Existing repository baseline:

```bash
python train.py configs/datasets/ptbxl.toml
```

PULSE text augmentation without BiomedCoOp:

```bash
python train.py configs/datasets/ptbxl_pulse.toml
```

PULSE text augmentation plus BiomedCoOp:

```bash
python train.py configs/datasets/ptbxl_pulse_biomedcoop.toml
```

Use the same seeds and data folds for all runs. At minimum, report:

1. MedTsLLM baseline with BiomedCoOp disabled.
2. MedTsLLM + PULSE feature-only descriptions.
3. MedTsLLM + BiomedCoOp.
4. MedTsLLM + PULSE + BiomedCoOp.

The existing `ptbxl.toml` has BiomedCoOp enabled, so it is not a pure MedTsLLM baseline. Create a copy with `[models.medtsllm.biomedcoop].enabled = false` for the first row.

## Important evaluation cautions

### PTB-XL exposure

PULSE was trained with PTB-XL-derived data. Treat results on PTB-XL as in-domain and explicitly document possible pretraining exposure. A stronger generalization experiment should also use a dataset not included in PULSE training.

### Description leakage

The provided prompt asks PULSE for visual features and blocks the five superclass labels. The generated text can still contain clinically diagnostic findings, such as bundle-branch-block morphology or ST elevation. This is expected semantic augmentation, but it must not be described as a label-free independent modality.

### Signal/image mismatch

The current MedTsLLM config uses a centered 512-sample signal crop, while the image renderer uses the complete 10-second ECG to preserve a standard clinical layout. For a strict same-information experiment, add a second baseline using `history_len = 1000` and compare it with a PULSE configuration that also uses `history_len = 1000`.

## Next implementation phase

After this text-augmentation baseline works, the stronger method is feature fusion:

```text
MedTsLLM signal representation + PULSE hidden visual representation
                         └──── learned gated/cross-attention fusion ────> class logits
```

That phase requires extracting PULSE multimodal hidden states and modifying the MedTsLLM classification head. It should be implemented only after the offline-description baseline is reproducible.
