"""Freeze an image-disjoint Visual Genome test manifest and prepare its JPEGs."""

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path

from vg_subset import DENSECAP_REVISION, place_image, read_json, read_raw_vg, region_pool, sha256


def prepare_manifest(source_dir, validation_path, count, image_pool, seed):
    split_path = source_dir / "densecap_splits.json"
    split_ids = read_json(split_path)
    train = set(map(int, split_ids["train"]))
    validation = set(map(int, split_ids["val"]))
    test = set(map(int, split_ids["test"]))
    if not all((train, validation, test)) or train & validation or train & test or validation & test:
        raise ValueError("Expected nonempty, image-disjoint DenseCap train, val, and test lists")

    previous = read_json(validation_path)
    seen = {int(region["image_id"]) for region in previous["regions"]}
    if not seen or not seen <= validation or seen & test:
        raise ValueError("The existing validation manifest must use only official validation images")

    images, annotations, sources = read_raw_vg(source_dir, test, image_pool, seed)
    regions = region_pool(images, annotations, count, seed)
    selected = {int(region["image_id"]) for region in regions}
    if selected & seen or not selected <= test:
        raise ValueError("The test sample overlaps the validation images or leaves the test split")
    for region in regions:
        region["image_path"] = f"images/{region['image_id']}.jpg"

    manifest = {
        "dataset": "Visual Genome v1.2 (raw)",
        "split": "test",
        "seed": seed,
        "densecap_split_revision": DENSECAP_REVISION,
        "validation_manifest_sha256": sha256(validation_path),
        "validation_image_count": len(seen),
        "validation_image_overlap": 0,
        "image_pool_size": len(images),
        "image_count": len(selected),
        "region_count": len(regions),
        "area_thresholds": [0.02, 0.10],
        "area_group_counts": dict(Counter(region["area_group"] for region in regions)),
        "source_sha256": {path.name: sha256(path) for path in [split_path, *sources]},
        "regions": regions,
    }
    return manifest, {image_id: images[image_id] for image_id in sorted(selected)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=Path("data/vg_sources"))
    parser.add_argument("--validation-manifest", type=Path, default=Path("data/vg_val_1000/manifest.json"))
    parser.add_argument("--output", type=Path, default=Path("data/vg_test_2000/manifest.json"))
    parser.add_argument("--count", type=int, default=2000)
    parser.add_argument("--image-pool", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20261125)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--proxy", help="Optional HTTP proxy for image downloads")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--download", action="store_true", help="Download only the uniquely selected test JPEGs")
    mode.add_argument("--verify-images", action="store_true", help="Verify all selected JPEGs, with network disabled")
    args = parser.parse_args()
    if min(args.count, args.image_pool, args.workers) <= 0:
        parser.error("--count, --image-pool, and --workers must be positive")

    manifest, images = prepare_manifest(args.source_dir, args.validation_manifest, args.count, args.image_pool, args.seed)
    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
    if output_path.exists():
        if output_path.read_text(encoding="utf-8") != payload:
            raise FileExistsError(f"The frozen manifest is different at {output_path}. Use a new output path.")
    else:
        with output_path.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(payload)

    print(f"Frozen {manifest['region_count']} regions on {len(images)} test images: {output_path}", flush=True)
    print(f"Validation image overlap: {manifest['validation_image_overlap']}", flush=True)
    if args.download or args.verify_images:
        print("Checking selected test JPEGs ...", flush=True)
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            tasks = [pool.submit(place_image, image_id, metadata, output_path.parent, None, args.proxy, args.download)
                     for image_id, metadata in images.items()]
            for task in tasks:
                task.result()
        print("All selected test JPEGs verified", flush=True)
    else:
        missing = sum(not (output_path.parent / "images" / f"{image_id}.jpg").is_file() for image_id in images)
        print(f"Missing local JPEGs: {missing}/{len(images)}. Use --download to fetch the selected files.", flush=True)


if __name__ == "__main__":
    main()
