
## Scope

This repository contains source code for the MRI shared-potential experiment. 


## Contents

- `trainer_sumnot_MRI_sharedpot.py`
- `tester_sumnot.py`
- `model_fvmot.py`
- `util.py`
- `utils/`
- `requirements.txt`
- `.gitignore`

## External resources

Place the MRI dataset outside the repository or at a private local path. Place `pre_trained_medmae.pth` and the selected checkpoint outside the repository unless a private secure distribution method is used.

## Training

```bash
# MRI SUMNOT Shared-Potentials Handoff

This package is a local, code-level handoff for the MRI shared-potential variant.
Run all commands from this folder.

## Contents

- `trainer_sumnot_MRI_sharedpot.py`: training entry point
- `tester_sumnot.py`: MRI testing and pure-image inference entry point
- `model_sumnot.py`: complete local copy of the Restormer/model implementation
- `utils/`: MRI dataset and image utility modules
- `util.py`: minimal freeze/unfreeze helpers used by training
- `checkpoint/model_sumnot_mri_post_128_bs3.pth`: default evaluation checkpoint
- `weights/pre_trained_medmae.pth`: MedMAE initialization weight

## Data Layout

Place the MRI data beside this package or set `--data_dir` to its location:

```text
data/MFMRI/
  HF/train/
  HF/test/
  LF_0.055T/test/
  LF_0.075T/train/
  LF_0.075T/test/
  ...
```



## Test

```bash
python tester_sumnot.py \
  --weights checkpoint/model_sumnot_mri_post_128_bs3.pth \
  --data_dir /path/to/data/MFMRI \
  --save_dir results/MFMRI \
  --gpus 0
```

## Train

Example single-GPU launch:

```bash
  CUDA_VISIBLE_DEVICES=2 torchrun \
  --nproc_per_node=1 \
  --master_port=9812 \
  trainer_sumnot_MRI_sharedpot.py \
  --tau=0.5 \
  --step=40 \
  --batchSize=3 \
  --nEpochs=80 \
  --backbone=Restormer \
  --grad_accum_steps=1 \
  --lr=2e-4 \
  --num_sources=7 \
  --de_type  LF_0.075T LF_0.1T LF_0.3T LF_0.5T LF_0.8T  \
  --patch_size=128 \
  --type=sumnot_mri \
  --resume_training=false \
```

For DDP, use `torchrun` and set `--gpus` as appropriate for the process launch.



```

For multi-GPU use `torchrun --standalone --nproc_per_node=2` with the same script and arguments.

## Testing

```bash
python tester_sumnot.py \
  --weights /private/path/to/checkpoint.pth \
  --data_dir /private/path/to/MFMRI \
  --save_dir results/MFMRI \
  --gpus 0
```
