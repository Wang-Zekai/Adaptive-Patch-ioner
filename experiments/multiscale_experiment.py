"""Benchmark the five region-captioning modes on a fixed region manifest."""

import _paths  # Make the shared source modules importable.

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import random
from statistics import mean
import subprocess
from time import perf_counter


AREA_GROUPS = ("small", "medium", "large")
FUSION_MODES = ("detail_fusion", "context_fusion", "multiscale_fusion")


def read_manifest(path, image_root):
    raw = path.read_bytes()
    document = json.loads(raw)
    regions = document["regions"]
    if not isinstance(regions, list) or not regions:
        raise ValueError("The manifest must contain a non-empty regions list.")
    seen = set()
    image_paths = {}
    for region in regions:
        region_id = str(region["region_id"])
        if region_id in seen:
            raise ValueError(f"Duplicate manifest region {region_id}")
        seen.add(region_id)
        path_on_disk = (image_root / region["image_path"]).resolve()
        if not path_on_disk.is_file():
            raise FileNotFoundError(f"Image for region {region_id} is missing: {path_on_disk}")
        image_paths[region_id] = path_on_disk
        refs = region["references"]
        if not isinstance(refs, list) or not refs or any(not isinstance(s, str) or not s.strip() for s in refs):
            raise ValueError(f"Region {region_id} needs at least one non-empty reference description.")
        box = region["bbox_xywh"]
        if len(box) != 4 or not all(math.isfinite(v) for v in box) or min(box[2:]) <= 0:
            raise ValueError(f"Invalid xywh coordinates for region {region_id}: {box!r}")
        if region["area_group"] not in AREA_GROUPS:
            raise ValueError(f"Invalid area group for region {region_id}: {region['area_group']!r}")
    return document, image_paths, hashlib.sha256(raw).hexdigest()


def summarize(rows, modes):
    groups = []
    for mode in modes:
        for group in ("all", *AREA_GROUPS):
            matches = [r for r in rows if r["mode"] == mode and (group == "all" or r["area_group"] == group)]
            groups.append({
                "mode": mode,
                "area_group": group,
                "regions": len(matches),
                "mean_inference_ms": round(mean(r["inference_ms"] for r in matches), 3) if matches else None,
                "peak_allocated_mib": max((r["peak_allocated_mib"] for r in matches), default=None),
                "peak_reserved_mib": max((r["peak_reserved_mib"] for r in matches), default=None),
            })
    return groups


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_csv(path, rows):
    fields = dict.fromkeys(key for row in rows for key in row)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, ensure_ascii=False) if isinstance(value, (list, tuple, dict)) else value
                             for key, value in row.items()})


def reset_rng(seed, np, torch):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def measured_prediction(captioner, image, bbox, mode, options, torch, mib):
    torch.cuda.synchronize(captioner.device)
    torch.cuda.reset_peak_memory_stats(captioner.device)
    started = perf_counter()
    prediction = captioner.predict(image, bbox, mode, **options)
    torch.cuda.synchronize(captioner.device)
    elapsed_ms = (perf_counter() - started) * 1000
    caption = prediction["caption"]
    if not isinstance(caption, str) or not caption.strip():
        raise RuntimeError(f"{mode} returned an empty region caption.")
    return {
        "caption": caption.strip(),
        "context_available": prediction.get("context_available"),
        "effective_weights": prediction.get("effective_weights"),
        "inference_ms": round(elapsed_ms, 3),
        "peak_allocated_mib": round(torch.cuda.max_memory_allocated(captioner.device) / mib, 3),
        "peak_reserved_mib": round(torch.cuda.max_memory_reserved(captioner.device) / mib, 3),
    }


def parse_arguments(modes, detail_scales, context_scales):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path, help="JSON with fixed image paths, boxes and references.")
    parser.add_argument("--image-root", type=Path, help="Root for relative image paths. Defaults to manifest directory.")
    parser.add_argument("--output-dir", required=True, type=Path, help="New directory for results and region checkpoints.")
    parser.add_argument("--modes", nargs="+", choices=modes, default=list(modes))
    parser.add_argument("--detail-scale", type=float, choices=detail_scales, default=1.2)
    parser.add_argument("--context-scale", type=float, choices=context_scales, default=2.0)
    parser.add_argument("--weights", action="append", nargs=4, default=[],
                        metavar=("MODE", "ORIGINAL", "DETAIL", "CONTEXT"), help="Override the weights for one fusion mode.")
    parser.add_argument("--seed", type=int, help="Shared inference seed. Defaults to the manifest seed.")
    parser.add_argument("--warmup", type=int, default=1, help="Untimed predictions per mode on the first region.")
    parser.add_argument("--resume", action="store_true", help="Continue a benchmark that used the same data and options.")
    args = parser.parse_args()
    if args.warmup < 1:
        parser.error("--warmup must be at least 1.")
    if len(set(args.modes)) != len(args.modes):
        parser.error("--modes must not contain duplicates.")
    overrides = {}
    for mode, *raw in args.weights:
        if mode not in FUSION_MODES or mode not in args.modes or mode in overrides:
            parser.error(f"Invalid or repeated weight override for {mode!r}.")
        try:
            weights = list(map(float, raw))
        except ValueError:
            parser.error(f"Weights for {mode} must be numeric.")
        if any(not math.isfinite(x) or x < 0 for x in weights) or not any(weights):
            parser.error(f"Weights for {mode} must be non-negative, finite, and not all zero.")
        overrides[mode] = weights
    return args, overrides


def main():
    import multiscale
    from region_smoke import DINO_CACHE, MIB, MODEL_REVISION, UPSTREAM

    args, overrides = parse_arguments(multiscale.MODES, multiscale.DETAIL_SCALES, multiscale.CONTEXT_SCALES)
    manifest = args.manifest.resolve()
    image_root = args.image_root.resolve() if args.image_root else manifest.parent
    document, image_paths, manifest_hash = read_manifest(manifest, image_root)
    regions = document["regions"]
    seed = args.seed if args.seed is not None else document.get("seed", 20261124)
    if not isinstance(seed, int) or not 0 <= seed < 2**32:
        raise ValueError("Inference seed must be an integer from 0 through 4294967295.")

    import numpy as np
    from PIL import Image
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Run with the Patch-ioner Conda environment and RTX 4060.")
    modes = args.modes
    mode_options = {mode: {"detail_scale": args.detail_scale, "context_scale": args.context_scale,
                           "weights": overrides.get(mode)} for mode in modes}
    settings = {
        "manifest": str(manifest),
        "manifest_sha256": manifest_hash,
        "image_root": str(image_root),
        "dataset": document.get("dataset"),
        "split": document.get("split"),
        "seed": seed,
        "mode_options": mode_options,
        "configured_weights": {mode: list(overrides.get(mode, multiscale.DEFAULT_WEIGHTS[mode])) for mode in modes},
        "modes": modes,
        "warmup_runs_per_mode": args.warmup,
        "device": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "model_revision": MODEL_REVISION,
        "multiscale_code_sha256": hashlib.sha256(Path(multiscale.__file__).read_bytes()).hexdigest(),
        "git_commit": subprocess.check_output(["git", "-C", str(UPSTREAM), "rev-parse", "HEAD"], text=True).strip(),
        "dinov2_commit": subprocess.check_output(
            ["git", "-C", str(DINO_CACHE / "facebookresearch_dinov2_main"), "rev-parse", "HEAD"], text=True
        ).strip(),
        "timing": "CUDA-synchronized single-region prediction after warm-up, including input transforms, excluding disk I/O and model load.",
        "memory": "PyTorch peak MiB including loaded model. Reserved memory can reflect allocations kept by earlier modes.",
    }
    output = args.output_dir.resolve()
    metadata_file = output / "metadata.json"
    progress = output / "progress"
    if args.resume:
        if json.loads(metadata_file.read_text(encoding="utf-8")) != settings:
            raise ValueError("Previous benchmark settings differ. Run into a new output directory.")
    else:
        output.mkdir(parents=True)
        progress.mkdir()
        write_json(metadata_file, settings)

    pending = [index for index in range(len(regions)) if not (progress / f"{index:06d}.json").is_file()]
    if pending:
        reset_rng(seed, np, torch)
        captioner = multiscale.load_captioner(device="cuda:0")
        first = regions[0]
        with Image.open(image_paths[str(first["region_id"])]) as source:
            warmup_image = source.convert("RGB")
        with torch.inference_mode():
            for mode in modes:
                for _ in range(args.warmup):
                    reset_rng(seed, np, torch)
                    captioner.predict(warmup_image, first["bbox_xywh"], mode, **mode_options[mode])
            torch.cuda.synchronize(captioner.device)

            for completed, index in enumerate(pending, start=1):
                region = regions[index]
                with Image.open(image_paths[str(region["region_id"])]) as source:
                    image = source.convert("RGB")
                predictions = {}
                offset = index % len(modes)
                for mode in modes[offset:] + modes[:offset]:
                    reset_rng((seed + index) % 2**32, np, torch)
                    predictions[mode] = measured_prediction(captioner, image, region["bbox_xywh"], mode,
                                                             mode_options[mode], torch, MIB)
                baseline = predictions.get("baseline", {}).get("caption")
                rows = [{**region, "mode": mode, "baseline_caption": baseline, **predictions[mode]} for mode in modes]
                write_json(progress / f"{index:06d}.json", rows)
                if completed % 10 == 0 or completed == len(pending):
                    print(f"Measured {completed}/{len(pending)} pending regions", flush=True)

    rows = [row for index in range(len(regions))
            for row in json.loads((progress / f"{index:06d}.json").read_text(encoding="utf-8"))]
    summary = summarize(rows, modes)
    write_json(output / "results.json", {"metadata": settings, "rows": rows, "summary": summary})
    write_csv(output / "results.csv", rows)
    write_csv(output / "summary.csv", summary)
    print(output / "results.json", flush=True)


if __name__ == "__main__":
    main()
