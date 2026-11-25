"""Exercise the real pretrained model and save the sample comparison."""

import _paths  # Make the shared source modules importable.

import argparse
import csv
import json
import math
from pathlib import Path
import random
from statistics import mean
import subprocess
from time import perf_counter
from unittest.mock import patch

from PIL import Image, ImageDraw, ImageFont
import torch

from multiscale import (
    BRANCHES, CONTEXT_SCALES, DEFAULT_WEIGHTS, DETAIL_SCALES, MODES,
    Box, clip_box, context_patch_weights, detail_geometry, fuse_features,
    load_captioner, patch_overlap,
)
from region_smoke import ASSETS, DINO_CACHE, MIB, MODEL_REVISION, ROOT, UPSTREAM


IMAGE_PATH = ASSETS / "examples" / "dog-frisbee.jpg"
SAMPLES = {
    "dog": (44, 108, 273, 244),
    "person": (453, 15, 160, 424),
}
EDGE_CASES = {
    "small": (250.25, 157.5, 0.75, 0.5),
    "boundary": (-3, -2, 11, 14),
    "far_edge": (639.999999, 438.999999, 0.000001, 0.000001),
    "large": (4, 3, 626, 428),
    "wide": (7, 200, 626, 1),
    "tall": (315, 7, 1, 425),
    "full_image": (0, 0, 640, 439),
}


def check_geometry(image):
    checked = []
    for name, values in {**SAMPLES, **EDGE_CASES}.items():
        box = clip_box(values, image.size)
        for scale in DETAIL_SCALES:
            crop, target = detail_geometry(box, image.size, scale)
            left, top, right, bottom = crop
            assert 0 <= left < right <= image.width and 0 <= top < bottom <= image.height
            assert target.x >= 0 and target.y >= 0
            assert target.x + target.w <= right - left + 1e-10
            assert target.y + target.h <= bottom - top + 1e-10
            resized = target.scaled(518 / (right - left), 518 / (bottom - top))
            recovered = (
                resized.x * (right - left) / 518 + left,
                resized.y * (bottom - top) / 518 + top,
                resized.w * (right - left) / 518,
                resized.h * (bottom - top) / 518,
            )
            assert all(math.isclose(a, b, abs_tol=1e-10) for a, b in zip(box.xywh, recovered))
            area = patch_overlap(resized, (37, 37), 14)
            assert (area > 0).any()
            assert math.isclose(area.sum().item(), resized.w * resized.h, rel_tol=1e-4, abs_tol=1e-12)
            checked.append({"case": name, "detail_scale": scale, "crop_ltrb": crop, "target_resized_xywh": resized.xywh})
    dog = clip_box(SAMPLES["dog"], image.size)
    crop, target = detail_geometry(dog, image.size, 1.5)
    resized = target.scaled(518 / (crop[2] - crop[0]), 518 / (crop[3] - crop[1]))
    assert (patch_overlap(resized, (37, 37), 14) == 0).any(), "Full crop background entered the detail pool"

    grid_target = Box(14, 14, 14, 14)
    grid_outer = Box(0, 0, 42, 42)
    weights = context_patch_weights(grid_target, grid_outer, (3, 3), 14)
    assert (weights > 0).sum().item() == 8 and weights[1, 1] == 0
    for bbox in ((0, 0, 0, 1), (0, 0, -1, 1), (math.nan, 0, 1, 1), (0, 0, math.inf, 1), (900, 0, 5, 5)):
        try:
            clip_box(bbox, image.size)
        except ValueError:
            continue
        raise AssertionError(f"Invalid box accepted: {bbox}")
    return checked


@torch.inference_mode()
def check_live_features(captioner, image, baseline_reference):
    model = captioner.model
    assert not model.training and not any(p.requires_grad for p in model.parameters())
    assert model.im_proj.text_dataset.shape[0] == model.im_proj.SUPPORT_MEMORY_SIZE == 591753
    region = clip_box(SAMPLES["dog"], image.size)
    tensor = captioner.prepare_image(image)
    boxes = captioner.upstream_boxes(image, region, tensor)
    upstream = {}
    caption_tokens = model.caption_tokens

    def capture(feature, **options):
        upstream["feature"] = feature.clone()
        return caption_tokens(feature, **options)

    with patch.object(model, "caption_tokens", side_effect=capture):
        direct = model(tensor, bboxes=boxes.clone(), get_cls_capt=False, gaussian_avg=False)["bbox_capts"][0][0]
    baseline = captioner.predict(image, region.xywh, "baseline")["caption"]
    assert baseline == direct
    if baseline_reference is not None:
        saved_smoke = json.loads(baseline_reference.read_text(encoding="utf-8"))
        if saved_smoke.get("bbox_xywh") != list(region.xywh) or saved_smoke.get("image_size_wh") != list(image.size):
            raise ValueError("The baseline reference must use the example image and --bbox 44 108 273 244.")
        assert baseline == saved_smoke["caption"]

    patches, _ = captioner._patches(tensor)
    unmodified_boxes = boxes.clone()
    original = captioner._original_feature(patches, boxes)
    assert torch.equal(boxes, unmodified_boxes), "Upstream pooling mutated the caller's coordinates"
    assert torch.equal(original, upstream["feature"]), "Original branch differs from upstream forward"
    fused, effective = fuse_features({"original": original}, (1, 0, 0))
    assert torch.equal(fused, upstream["feature"]) and effective == (1, 0, 0)
    identity = captioner.predict(image, region.xywh, "multiscale_fusion", weights=(1, 0, 0))
    assert identity["caption"] == baseline

    feature_checks = []
    for name, values in {**SAMPLES, **EDGE_CASES}.items():
        box = clip_box(values, image.size)
        for detail_scale, context_scale in zip(DETAIL_SCALES, CONTEXT_SCALES):
            vectors = captioner.extract_features(image, box, detail_scale=detail_scale, context_scale=context_scale)
            stats = {}
            for branch in BRANCHES:
                vector = vectors[branch]
                if vector is None:
                    assert branch == "context"
                    stats[branch] = None
                    continue
                assert vector.shape == (1, 768) and vector.dtype == torch.float32
                assert torch.isfinite(vector).all().item()
                length = torch.linalg.vector_norm(vector).item()
                assert length > 0
                stats[branch] = {"l2": length, "min": vector.min().item(), "max": vector.max().item()}
            combined, weights = fuse_features(vectors, DEFAULT_WEIGHTS["multiscale_fusion"])
            assert torch.isfinite(combined).all().item() and sum(weights) > 0
            if name == "full_image":
                assert vectors["context"] is None and weights[2] == 0
            feature_checks.append({"case": name, "detail_scale": detail_scale, "context_scale": context_scale,
                                   "branch_stats": stats, "effective_weights": weights})

    full = EDGE_CASES["full_image"]
    full_baseline = captioner.predict(image, full, "baseline")
    full_context = captioner.predict(image, full, "context_fusion")
    full_multiscale = captioner.predict(image, full, "multiscale_fusion")
    assert full_context["context_available"] is False and full_multiscale["context_available"] is False
    assert full_context["caption"] == full_baseline["caption"] and full_context["effective_weights"] == (1, 0, 0)
    assert all(math.isclose(x, y) for x, y in zip(full_multiscale["effective_weights"], (2 / 3, 1 / 3, 0)))
    return feature_checks


def measure(captioner, image, repeats):
    device = captioner.device
    for mode in MODES:
        for _ in range(2):
            captioner.predict(image, SAMPLES["dog"], mode)
    torch.cuda.synchronize(device)
    rows = []
    for j, (region, box) in enumerate(SAMPLES.items()):
        index_shift = j % len(MODES)
        for mode in MODES[index_shift:] + MODES[:index_shift]:
            elapsed, allocated, reserved = [], [], []
            captions = []
            for _ in range(repeats):
                torch.cuda.reset_peak_memory_stats(device)
                start = perf_counter()
                result = captioner.predict(image, box, mode)
                torch.cuda.synchronize(device)
                elapsed.append((perf_counter() - start) * 1000)
                allocated.append(torch.cuda.max_memory_allocated(device) / MIB)
                reserved.append(torch.cuda.max_memory_reserved(device) / MIB)
                captions.append(result["caption"])
            assert len(set(captions)) == 1 and captions[0].strip(), (region, mode, captions)
            rows.append({"region": region, "mode": mode, **result, "avg_ms": mean(elapsed),
                         "peak_allocated_mib": max(allocated), "peak_reserved_mib": max(reserved),
                         "elapsed_ms": elapsed})
    return sorted(rows, key=lambda row: (list(SAMPLES).index(row["region"]), MODES.index(row["mode"])))


def check_memory_plateau(captioner, image):
    device = captioner.device

    def snapshot(label):
        torch.cuda.synchronize(device)
        return {"label": label, "allocated_mib": torch.cuda.memory_allocated(device) / MIB,
                "reserved_mib": torch.cuda.memory_reserved(device) / MIB}

    regions = (*SAMPLES.values(), *EDGE_CASES.values())
    snapshots = []
    for cycle in range(3):
        for idx, box in enumerate(regions):
            captioner.predict(image, box, MODES[(cycle + idx) % len(MODES)])
        snapshots.append(snapshot(cycle + 1))
    assert max(x["allocated_mib"] for x in snapshots) - min(x["allocated_mib"] for x in snapshots) <= 1
    assert snapshots[-1]["reserved_mib"] == snapshots[-2]["reserved_mib"], snapshots
    return {"region_calls": 3 * len(regions), "per_cycle": snapshots}


def draw_sample(image, rows, output):
    try:
        title = ImageFont.truetype("arial.ttf", 30)
        regular = ImageFont.truetype("arial.ttf", 19)
        bold = ImageFont.truetype("arialbd.ttf", 19)
    except OSError:
        title = ImageFont.load_default(size=30)
        regular = bold = ImageFont.load_default(size=19)
    canvas = Image.new("RGB", (1330, 960), "#f6f7f9")
    draw = ImageDraw.Draw(canvas)
    draw.text((28, 19), "Patch-ioner | Official image, two selected regions", font=title, fill="#111827")
    for index, (name, xywh) in enumerate(SAMPLES.items()):
        pane = image.copy()
        pane_draw = ImageDraw.Draw(pane)
        box = clip_box(xywh, pane.size)
        crop, _ = detail_geometry(box, pane.size, 1.2)
        outer = clip_box(box.expanded(2.0).xywh, pane.size)
        pane_draw.rectangle((outer.x, outer.y, outer.x + outer.w - 1, outer.y + outer.h - 1), outline="#ffca56", width=4)
        pane_draw.rectangle((crop[0], crop[1], crop[2] - 1, crop[3] - 1), outline="#55c7ff", width=4)
        pane_draw.rectangle((box.x, box.y, box.x + box.w - 1, box.y + box.h - 1), outline="#4de39a", width=4)
        offset = 25 + index * 645
        canvas.paste(pane, (offset, 67))
        draw.text((offset, 519), name.capitalize() + " region", font=title, fill="#111827")
        for j, row in enumerate(r for r in rows if r["region"] == name):
            y = 567 + j * 69
            draw.text((offset, y), row["mode"], font=bold, fill="#343d4b")
            draw.text((offset, y + 24), row["caption"].strip(), font=regular, fill="#111827")
    draw.text((25, 923), "Green: target    Blue: zoom crop    Gold: outer context box", font=regular, fill="#343d4b")
    canvas.save(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "multiscale_sample")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--baseline-reference", type=Path, help="Optional JSON from region_smoke.py")
    args = parser.parse_args()
    if args.repeats < 1 or args.output_dir.exists():
        parser.error("Choose a new output directory and at least one measured repeat")
    random.seed(2026)
    torch.manual_seed(2026)
    with Image.open(IMAGE_PATH) as source:
        image = source.convert("RGB")
    geometry = check_geometry(image)
    print("Geometry, masks, and invalid input checks passed.", flush=True)
    captioner = load_captioner()
    checks = check_live_features(captioner, image, args.baseline_reference)
    print("Upstream equivalence, identity weights, real features, and fallback checks passed.", flush=True)
    rows = measure(captioner, image, args.repeats)
    memory = check_memory_plateau(captioner, image)
    print("All five modes and repeated-region memory checks passed.", flush=True)

    payload = {
        "image": IMAGE_PATH.name,
        "gpu": torch.cuda.get_device_name(captioner.device),
        "upstream_commit": subprocess.check_output(["git", "-C", str(UPSTREAM), "rev-parse", "HEAD"], text=True).strip(),
        "dino_commit": subprocess.check_output(["git", "-C", str(DINO_CACHE / "facebookresearch_dinov2_main"), "rev-parse", "HEAD"], text=True).strip(),
        "model_revision": MODEL_REVISION,
        "pytorch": torch.__version__, "cuda": torch.version.cuda, "seed": 2026, "warmup_per_mode": 2,
        "repeats_per_region_and_mode": args.repeats,
        "checks": {"upstream_caption": True, "original_feature_bitwise_equal": True,
                   "identity_weights_bitwise_equal": True, "boxes_unmodified": True,
                   "empty_context_fallback": True, "all_parameters_frozen": True,
                   "invalid_boxes_rejected": True, "memory_plateau": True},
        "geometry": geometry, "features": checks, "continuous_memory": memory, "sample_results": rows,
    }
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / "sample_results.json").write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    with (args.output_dir / "sample_results.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    draw_sample(image, rows, args.output_dir / "sample_regions.png")
    print(f"Results written to {args.output_dir}")


if __name__ == "__main__":
    main()
