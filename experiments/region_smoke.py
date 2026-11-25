import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from time import perf_counter


from _paths import ROOT


ASSET_ROOT = Path(os.environ.get("PATCH_IONER_ASSET_ROOT", ROOT)).resolve()
UPSTREAM = ASSET_ROOT / "upstream"
ASSETS = ASSET_ROOT / "assets"
MODEL_DIR = ASSETS / "talk2dino-decap"
DINO_CACHE = ASSETS / "torch" / "hub"
MODEL_REVISION = "b2ff74dc8f81b5182576dc72db408e113d6c6bce"
MIB = 1024**2


def main():
    parser = argparse.ArgumentParser(description="Run the official Patch-ioner region captioner on one image.")
    parser.add_argument("--image", type=Path, default=ASSETS / "examples" / "dog-frisbee.jpg")
    parser.add_argument("--bbox", type=float, nargs=4, metavar=("X", "Y", "WIDTH", "HEIGHT"),
                        help="Box in original-image pixels. Default: the central half of the image.")
    parser.add_argument("--output", type=Path, help="Save the measured result as a new JSON file.")
    args = parser.parse_args()
    if args.output and args.output.exists():
        parser.error(f"Output file already exists: {args.output}")

    config_path = MODEL_DIR / "config.yaml"
    decoder_path = MODEL_DIR / "coco_karpathy-009.pt"
    memory_path = MODEL_DIR / "coco_captions_text_embeddings-B16-ViT-B.16-591753.h5"
    dino_path = DINO_CACHE / "checkpoints" / "dinov2_vitb14_reg4_pretrain.pth"
    prerequisites = {
        UPSTREAM / "Patch-ioner" / "src" / "model.py": None,
        DINO_CACHE / "facebookresearch_dinov2_main" / "hubconf.py": None,
        config_path: 766,
        decoder_path: 273331640,
        memory_path: 1869966480,
        dino_path: 346393545,
    }
    missing = [str(path) for path, size in prerequisites.items()
               if not path.is_file() or (size is not None and path.stat().st_size != size)]
    if missing:
        parser.exit(2, "Missing or incomplete files. Run scripts/setup.sh or scripts/setup.ps1.\n" + "\n".join(missing) + "\n")

    os.environ["IM2TXT_MEMORY_PATH"] = str(MODEL_DIR)
    os.environ["TORCH_HOME"] = str(DINO_CACHE.parent)
    os.environ["HF_HOME"] = str(ASSETS / "huggingface")
    os.environ["HF_HUB_CACHE"] = str(ASSETS / "huggingface" / "hub")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ.pop("TRANSFORMERS_CACHE", None)
    sys.path.insert(0, str(UPSTREAM / "Patch-ioner"))

    import psutil
    import torch
    import yaml
    from PIL import Image
    from src import Patchioner
    from src.bbox_utils import adjust_bbox_for_transform_no_scale

    if not torch.cuda.is_available():
        parser.exit(2, f"CUDA unavailable in {sys.executable} (PyTorch {torch.__version__}).\n")
    device = torch.device("cuda:0")
    process = psutil.Process()
    ram_before = psutil.virtual_memory().used
    available_vram, total_vram = torch.cuda.mem_get_info(device)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["decap_weights"] = str(decoder_path)

    with Image.open(args.image) as source:
        image = source.convert("RGB")
    width, height = image.size
    bbox = args.bbox or [width / 4, height / 4, width / 2, height / 2]
    x, y, w, h = bbox
    if (not all(map(math.isfinite, bbox)) or min(x, y) < 0 or min(w, h) <= 0
            or x + w > width or y + h > height):
        parser.error(f"--bbox must be within the {width}x{height} image with positive width and height.")

    with torch.inference_mode():
        torch.cuda.reset_peak_memory_stats(device)
        started = perf_counter()
        model = Patchioner.from_config(config, device=str(device)).eval()
        torch.cuda.synchronize(device)
        load_seconds = perf_counter() - started
        load_peak = torch.cuda.max_memory_allocated(device)

        image_tensor = model.image_transforms_no_crop(image).unsqueeze(0).to(device)
        image_height, image_width = image_tensor.shape[-2:]
        scaled_bbox = adjust_bbox_for_transform_no_scale(image, bbox, image_width, image_height)
        # extract_bboxes_feats mutates the box tensor, so each pass needs its own tensor.
        model_boxes = torch.tensor([[scaled_bbox]], dtype=torch.float32)
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        started = perf_counter()
        outputs = model(image_tensor, bboxes=model_boxes, get_cls_capt=False, gaussian_avg=False)
        torch.cuda.synchronize(device)
        inference_ms = (perf_counter() - started) * 1000

    captions = outputs.get("bbox_capts")
    if not captions or not captions[0] or not isinstance(captions[0][0], str) or not captions[0][0].strip():
        raise RuntimeError(f"The model did not return a region caption: {captions!r}")

    application_ram = process.memory_info()
    result = {
        "git_commit": subprocess.check_output(["git", "-C", str(UPSTREAM), "rev-parse", "HEAD"], text=True).strip(),
        "dinov2_commit": subprocess.check_output(
            ["git", "-C", str(DINO_CACHE / "facebookresearch_dinov2_main"), "rev-parse", "HEAD"], text=True
        ).strip(),
        "model_revision": MODEL_REVISION,
        "image": str(args.image.resolve()),
        "image_size_wh": [width, height],
        "bbox_xywh": bbox,
        "transformed_bbox_xywh": scaled_bbox,
        "caption": captions[0][0],
        "pytorch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(device),
        "load_seconds": round(load_seconds, 4),
        "inference_ms": round(inference_ms, 2),
        "vram_total_mib": round(total_vram / MIB, 2),
        "vram_available_before_load_mib": round(available_vram / MIB, 2),
        "vram_load_peak_allocated_mib": round(load_peak / MIB, 2),
        "vram_inference_peak_allocated_mib": round(torch.cuda.max_memory_allocated(device) / MIB, 2),
        "vram_inference_peak_reserved_mib": round(torch.cuda.max_memory_reserved(device) / MIB, 2),
        "process_ram_peak_mib": round(getattr(application_ram, "peak_wset", application_ram.rss) / MIB, 2),
        "system_ram_before_load_mib": round(ram_before / MIB, 2),
        "system_ram_after_inference_mib": round(psutil.virtual_memory().used / MIB, 2),
    }
    payload = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        with args.output.open("x", encoding="utf-8") as output_file:
            output_file.write(payload + "\n")
    print(payload)


if __name__ == "__main__":
    main()
