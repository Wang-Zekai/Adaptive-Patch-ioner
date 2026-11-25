"""Caption an image region, or score the same methods on a labeled manifest."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil
from statistics import mean
import sys

import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vg_metrics import AREA_GROUPS, score_predictions, write_csv


METRICS = ("CIDEr", "BLEU", "ROUGE_L", "METEOR")


def read_regions(path, image_root):
    raw = path.read_bytes()
    document = json.loads(raw)
    regions = document["regions"]
    if not isinstance(regions, list) or not regions:
        raise ValueError("The manifest must contain a nonempty regions list")
    seen, resolved = set(), []
    for row in regions:
        key = str(row["region_id"])
        if key in seen:
            raise ValueError(f"Duplicate region_id: {key}")
        seen.add(key)
        refs = row["references"]
        if not isinstance(refs, list) or not refs or any(not isinstance(s, str) or not s.strip() for s in refs):
            raise ValueError(f"Region {key} needs at least one reference description")
        box = row["bbox_xywh"]
        if len(box) != 4 or not all(math.isfinite(v) for v in box) or min(box[2:]) <= 0:
            raise ValueError(f"Invalid bbox_xywh for region {key}")
        group = row.get("area_group")
        if group is not None and group not in AREA_GROUPS:
            raise ValueError(f"Unknown area group for region {key}: {group}")
        image_path = image_root / row["image_path"]
        if not image_path.is_file():
            raise ValueError(f"Image for region {key} does not exist: {image_path}")
        resolved.append((row, image_path))
    return resolved, hashlib.sha256(raw).hexdigest()


def manifest_predictions(model, regions, methods):
    # Load each requested image once per consecutive group. Timed intervals exclude image I/O.
    from visual_features import image_rgb

    rows = []
    configs = list(methods.values())
    previous_path, image = None, None
    for idx, (region, path) in enumerate(regions):
        if path != previous_path:
            image = image_rgb(path)
            previous_path = path
        box = region["bbox_xywh"]
        if idx == 0:
            for config in configs:
                for _ in range(2):
                    model.predict(image, box, config)
        offset = idx % len(configs)
        for config in configs[offset:] + configs[:offset]:
            result = model.predict(image, box, config)
            _, _, width, height = result["bbox_xywh"]
            fraction = width * height / (image.width * image.height)
            group = region.get("area_group") or ("small" if fraction < 0.02 else "medium" if fraction < 0.10 else "large")
            rows.append({**region, **result, "mode": config["id"], "area_group": group})
    return rows


def summarize(rows, methods):
    summary = []
    for config in methods.values():
        mode = config["id"]
        for group in ("all", *AREA_GROUPS):
            matches = [r for r in rows if r["mode"] == mode and (group == "all" or r["area_group"] == group)]
            peaks = [r["peak_allocated_mib"] for r in matches if r["peak_allocated_mib"] is not None]
            summary.append({
                "mode": mode,
                "area_group": group,
                "mean_inference_ms": mean(r["inference_ms"] for r in matches) if matches else None,
                "peak_allocated_mib": max(peaks, default=None),
                "detail_activation_rate": mean(r["detail_active"] for r in matches) if matches else None,
            })
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/adaptive.yaml")
    parser.add_argument("--method", help="A name from the config, or 'all' for a manifest")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--image", type=Path)
    source.add_argument("--manifest", type=Path)
    parser.add_argument("--bbox", type=float, nargs=4, metavar=("X", "Y", "WIDTH", "HEIGHT"))
    parser.add_argument("--image-root", type=Path, help="Base directory for image_path in the manifest")
    parser.add_argument("--output-dir", type=Path, help="Directory for manifest scores and predictions")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.image is not None and (args.bbox is None or args.method == "all"):
        parser.error("--image requires --bbox and one method")
    if args.manifest and (args.output_dir is None or args.bbox is not None):
        parser.error("--manifest requires --output-dir and uses bounding boxes from the manifest")

    try:
        config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
        methods = config["methods"]
        method = args.method or config["default_method"]
        if method != "all" and method not in methods:
            parser.error(f"Unknown method {method!r}. Available: {', '.join(methods)}")
        if args.manifest:
            methods = methods if method == "all" else {key: methods[key] for key in dict.fromkeys(("baseline", method))}
            if not shutil.which("java"):
                parser.error("Manifest scoring requires Java on PATH for Stanford tokenization and METEOR")
            output_names = ("scores.json", "summary.csv", "per_region.csv")
            if any((args.output_dir / name).exists() for name in output_names):
                parser.error("Evaluation outputs already exist. Use a different --output-dir")
            regions, digest = read_regions(args.manifest, args.image_root or args.manifest.parent)

        from region_inference import RegionCaptioner
        from visual_features import image_rgb

        model = RegionCaptioner(args.device)
        if args.image:
            result = model.predict(image_rgb(args.image), args.bbox, methods[method])
            print(json.dumps({"method": method, **result}, ensure_ascii=False, indent=2))
            return

        rows = manifest_predictions(model, regions, methods)
        data = {"metadata": {"manifest_sha256": digest, "methods": methods, "latency_kind": "online"},
                "rows": rows, "summary": summarize(rows, methods)}
        scored = score_predictions(data, METRICS, "ptb")
        args.output_dir.mkdir(parents=True, exist_ok=True)
        with (args.output_dir / "scores.json").open("x", encoding="utf-8") as output:
            json.dump(scored, output, ensure_ascii=False, indent=2, allow_nan=False)
            output.write("\n")
        write_csv(args.output_dir / "summary.csv", scored["summary"])
        write_csv(args.output_dir / "per_region.csv", scored["rows"])
        print(f"Scored {len(rows)} predictions across {len(regions)} regions in {args.output_dir}")
    except (OSError, KeyError, ValueError, RuntimeError) as error:
        parser.exit(2, f"Error: {error}\n")


if __name__ == "__main__":
    main()
