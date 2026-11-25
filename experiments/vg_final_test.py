"""Freeze a fresh DenseCap Visual Genome test sample and verify its selected JPEGs."""

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path

from vg_subset import AREA_GROUPS, DENSECAP_REVISION, place_image, read_json, read_raw_vg, region_pool, sha256


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = ROOT / "data"


def prepare_manifest(source_dir, exclusion_paths, count, image_pool, seed, data_dir=DEFAULT_DATA_DIR):
    split_path = source_dir / "densecap_splits.json"
    split_ids = read_json(split_path)
    eligible = set(map(int, split_ids["test"]))
    split_sets = [set(map(int, split_ids[key])) for key in ("train", "val")]
    if not eligible or not all(split_sets) or any(eligible & ids for ids in split_sets) or split_sets[0] & split_sets[1]:
        raise ValueError("Expected disjoint nonempty official DenseCap train, val, and test splits")

    excluded = set()
    exclusions = {}
    for path in exclusion_paths:
        old = read_json(path)
        images = {int(row["image_id"]) for row in old["regions"]}
        if not images:
            raise ValueError(f"No image IDs to exclude in {path}")
        excluded.update(images)
        if path.is_relative_to(data_dir):
            name = (Path("data") / path.relative_to(data_dir)).as_posix()
        else:
            name = path.relative_to(ROOT).as_posix() if path.is_relative_to(ROOT) else path.as_posix()
        exclusions[name] = {"sha256": sha256(path), "image_count": len(images)}

    remaining = eligible - excluded
    if len(remaining) < image_pool:
        raise ValueError(f"Only {len(remaining)} eligible test images for --image-pool {image_pool}")
    images, annotations, sources = read_raw_vg(source_dir, remaining, image_pool, seed)
    regions = region_pool(images, annotations, count, seed)
    selected = {int(row["image_id"]) for row in regions}
    expected_groups = {
        group: count // len(AREA_GROUPS) + (index < count % len(AREA_GROUPS))
        for index, group in enumerate(AREA_GROUPS)
    }
    groups = Counter(row["area_group"] for row in regions)
    if groups != expected_groups:
        raise ValueError(f"Could not populate every size group: got {dict(groups)}, expected {expected_groups}. Increase --image-pool.")
    if selected & excluded or not selected <= eligible:
        raise ValueError("The selected test IDs overlap a prior manifest or leave the official test split")
    for region in regions:
        region["image_path"] = f"images/{region['image_id']}.jpg"

    manifest = {
        "dataset": "Visual Genome v1.2 (raw)",
        "split": "test",
        "seed": seed,
        "densecap_split_revision": DENSECAP_REVISION,
        "excluded_manifests": exclusions,
        "excluded_image_count": len(excluded),
        "excluded_images_on_test_split": len(excluded & eligible),
        "excluded_image_overlap": 0,
        "eligible_official_test_images": len(remaining),
        "image_pool_size": len(images),
        "image_count": len(selected),
        "region_count": len(regions),
        "area_thresholds": [0.02, 0.10],
        "area_group_counts": dict(groups),
        "source_sha256": {path.name: sha256(path) for path in [split_path, *sources]},
        "regions": regions,
    }
    return manifest, {image_id: images[image_id] for image_id in sorted(selected)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR,
                        help="Root for the default sources, exclusion manifests, and output")
    parser.add_argument("--source-dir", type=Path, help="Directory containing downloaded Visual Genome sources")
    parser.add_argument("--exclude", action="append", type=Path, default=[],
                        help="Additional viewed manifest to exclude; the two earlier sets are always excluded")
    parser.add_argument("--output", type=Path, help="Default: vg_test_5000/manifest.json inside --data-dir")
    parser.add_argument("--count", type=int, default=5000)
    parser.add_argument("--image-pool", type=int, default=512)
    parser.add_argument("--seed", type=int, default=20261126)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--proxy", help="Optional HTTP proxy for image downloads")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--download", action="store_true", help="Fetch only the selected test JPEGs")
    mode.add_argument("--verify-images", action="store_true", help="Verify selected JPEGs offline")
    args = parser.parse_args()
    if min(args.count, args.image_pool, args.workers) <= 0:
        parser.error("--count, --image-pool, and --workers must be positive")

    data_dir = args.data_dir.resolve()
    source_dir = args.source_dir or data_dir / "vg_sources"
    default_exclusions = (data_dir / "vg_val_1000/manifest.json", data_dir / "vg_test_2000/manifest.json")
    exclusions = tuple(dict.fromkeys(path.resolve() for path in [*default_exclusions, *args.exclude]))
    manifest, images = prepare_manifest(source_dir, exclusions, args.count, args.image_pool, args.seed, data_dir)
    output = (args.output or data_dir / "vg_test_5000/manifest.json").resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
    if output.exists():
        if output.read_text(encoding="utf-8") != payload:
            raise FileExistsError(f"A different frozen manifest already exists at {output}")
    else:
        with output.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(payload)

    print(f"Frozen {manifest['region_count']} regions on {len(images)} official test images: {output}", flush=True)
    print(f"Image overlap with every earlier manifest: {manifest['excluded_image_overlap']}", flush=True)
    print(f"Area groups: {manifest['area_group_counts']}", flush=True)
    if not (args.download or args.verify_images):
        missing = sum(not (output.parent / "images" / f"{iid}.jpg").is_file() for iid in images)
        print(f"Missing local JPEGs: {missing}/{len(images)}. Use --download to fetch the selected files.", flush=True)
        return

    failures = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(place_image, image_id, metadata, output.parent, None, args.proxy, args.download)
                   for image_id, metadata in images.items()]
        for index, (image_id, task) in enumerate(zip(images, futures, strict=True), start=1):
            try:
                task.result()
            except (OSError, ValueError, RuntimeError) as error:
                failures.append((image_id, str(error)))
            if index % 64 == 0:
                print(f"Checked {index}/{len(images)} selected test JPEGs", flush=True)
    if failures:
        details = "\n".join(f"{image_id}: {problem}" for image_id, problem in failures[:8])
        raise RuntimeError(f"{len(failures)}/{len(images)} test JPEGs missing or invalid. First failures:\n{details}")
    print("All selected test JPEGs verified", flush=True)


if __name__ == "__main__":
    main()
