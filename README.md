# Adaptive Patch-ioner

Training-free region captioning built on [Patch-ioner](https://github.com/Ruggero1912/Patch-ioner). This project was developed for Peking University's Multimodal Learning course.

The model receives an image and a target box. Balanced decides whether a second, zoomed view is worthwhile using features from the original image. When it uses a crop, it combines the original and zoomed features with a size-dependent weight. All pretrained weights, the full text memory bank, and the decoder stay frozen.

## Results

Held-out Visual Genome evaluation on 5,000 regions in 512 images. Latency is the mean of a 96-region online audit on an RTX 4060 Laptop GPU with 8 GB of memory. It includes image encoding and generation, excluding model loading and image-file reads.

| Method | CIDEr ↑ | Change vs. baseline | Latency |
| --- | ---: | ---: | ---: |
| Baseline | 0.35070 | — | 355.8 ms |
| Fixed detail fusion | 0.37248 | +6.21% | 447.5 ms |
| Previous best | 0.38863 | +10.82% | 445.6 ms |
| Quality | 0.38869 | +10.83% | 430.4 ms |
| Balanced | 0.37823 | +7.85% | 402.3 ms |
| Efficient | 0.35662 | +1.69% | 372.4 ms |
| Large-target fallback | 0.38366 | +9.40% | 403.7 ms |

Balanced used a crop for 60.8% of test regions. Its paired image-level 95% interval for the gain over baseline is [0.01476, 0.04033] CIDEr. Its estimated latency increase is 13.06%, with a 95% interval of [−5.55%, 32.88%]. Quality and the previous best scored almost identically. The large-target fallback reproduced baseline captions for all 1,666 large targets.

![Quality and online inference time on the test set](figures/quality_latency.svg)

- [Full test results](results/main_results.csv)
- [Validation ablations](results/ablation.csv)
- [Method and evaluation protocol](docs/method.md)

## Setup

Use Python 3.10, CUDA-enabled PyTorch 2.5.1, and torchvision 0.20.1. An example Conda environment is shown below.

```sh
conda create -n adaptive-patch-ioner python=3.10
conda activate adaptive-patch-ioner
conda install pytorch=2.5.1 torchvision=0.20.1 pytorch-cuda=12.4 -c pytorch -c nvidia
```

The setup scripts fetch the pinned upstream code, official weights, and example image. Allow about 3 GB for these assets and additional space for the Python environment. Put Java on PATH to use the benchmark's text tokenizer and METEOR.

Linux

```bash
bash scripts/setup.sh
```

Windows PowerShell

```powershell
pwsh -File scripts/setup.ps1
```

## Run

Give the box as `x y width height` in pixels of the input image.

```bash
python scripts/evaluate.py --config configs/baseline.yaml --image assets/examples/dog-frisbee.jpg --bbox 50 50 200 150
python scripts/evaluate.py --config configs/adaptive.yaml --method balanced --image assets/examples/dog-frisbee.jpg --bbox 50 50 200 150
```

To evaluate region captions against local references, supply a dataset manifest and the directory relative to which its image paths resolve.

```bash
python scripts/evaluate.py --config configs/adaptive.yaml --method balanced --manifest data/manifest.json --image-root data --output-dir runs/balanced
```

A manifest contains a `regions` array. Each entry identifies a target box and one or more reference captions.

```json
{
  "regions": [
    {
      "region_id": "example-1",
      "image_id": "example",
      "image_path": "example.jpg",
      "bbox_xywh": [50, 50, 200, 150],
      "references": ["a dog jumping to catch a frisbee"]
    }
  ]
}
```

Use `--method all` to evaluate all nine presets. To reuse an existing copy of the assets, set `PATCH_IONER_ASSET_ROOT` to the directory containing both `assets/` and `upstream/`.

The repository contains the course-project implementation and aggregate results. The data, pretrained weights, and upstream checkout are fetched or supplied locally.

## Upstream

This project uses the official Talk2DINO and DeCap checkpoint and builds on [One Patch to Caption Them All: A Unified Zero-Shot Captioning Framework](https://arxiv.org/abs/2510.02898), Bianchi et al. The setup script pins Patch-ioner to [commit 34d8130](https://github.com/Ruggero1912/Patch-ioner/tree/34d8130dafba0e4bb148bd0912c4a9186e398a05).
