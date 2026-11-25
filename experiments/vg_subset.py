"""Prepare a deterministic Visual Genome v1.2 DenseCap validation subset."""

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import hashlib
import io
import json
import math
import os
from pathlib import Path
import random
import time
from urllib.parse import urlparse
from zipfile import ZipFile

import requests
from PIL import Image


VG_ROOT = "https://homes.cs.washington.edu/~ranjay/visualgenome/data/dataset"
DENSECAP_REVISION = "7c32170f134805debe638806ecb0a22bbcd58c5f"
SOURCES = {
    "densecap_splits.json": (
        f"https://raw.githubusercontent.com/jcjohnson/densecap/{DENSECAP_REVISION}/info/densecap_splits.json"
    ),
    "image_data.json.zip": f"{VG_ROOT}/image_data.json.zip",
    "region_descriptions.json.zip": f"{VG_ROOT}/region_descriptions.json.zip",
}
IMAGE_HOST = "https://cs.stanford.edu/people/rak248"
AREA_GROUPS = ("small", "medium", "large")
DEFAULT_SEED = 20261124


def download(url, destination, proxy, validate):
    if destination.exists():
        validate(destination)
        return

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f"{destination.name}.part")
    proxies = {"http": proxy, "https": proxy} if proxy else None
    for attempt in range(3):
        try:
            with requests.get(url, proxies=proxies, stream=True, timeout=(15, 60)) as response:
                response.raise_for_status()
                with temporary.open("wb") as output:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        output.write(chunk)
            validate(temporary)
            temporary.replace(destination)
            return
        except (OSError, ValueError, requests.RequestException) as error:
            if attempt == 2:
                raise RuntimeError(f"Could not download {url}: {error}") from error
            time.sleep(2**attempt)


def verify_archive(path):
    with ZipFile(path) as archive:
        if not any(name.endswith(".json") for name in archive.namelist()):
            raise ValueError(f"No JSON in {path}")
        damaged_member = archive.testzip()
        if damaged_member:
            raise ValueError(f"Damaged archive member {damaged_member} in {path}")


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_split_ids(path):
    with path.open(encoding="utf-8") as stream:
        splits = json.load(stream)
    validation = set(map(int, splits["val"]))
    held_out = set(map(int, splits["test"]))
    if not validation or not held_out or validation & held_out:
        raise ValueError("Expected nonempty, disjoint DenseCap validation and test splits")
    if validation & set(map(int, splits["train"])):
        raise ValueError("DenseCap validation and training splits overlap")
    return validation


@contextmanager
def source_json(path):
    if path.suffix.lower() == ".zip":
        with ZipFile(path) as archive:
            entries = [member for member in archive.namelist() if Path(member).name == path.stem]
            if len(entries) != 1:
                raise ValueError(f"Expected one {path.stem} inside {path}")
            with archive.open(entries[0]) as entry:
                with io.TextIOWrapper(entry, encoding="utf-8-sig") as stream:
                    yield stream
    else:
        with path.open(encoding="utf-8-sig") as stream:
            yield stream


def read_json(path):
    with source_json(path) as stream:
        return json.load(stream)


def json_array_items(stream):
    """Decode one image record at a time from the large top-level JSON array."""
    decoder = json.JSONDecoder()
    buffer = stream.read(1024 * 1024).lstrip()
    if not buffer.startswith("["):
        raise ValueError("Expected a Visual Genome JSON array")
    buffer = buffer[1:]
    separator = ""
    while True:
        buffer = buffer.lstrip()
        if not buffer:
            buffer = stream.read(1024 * 1024).lstrip()
        if buffer.startswith("]"):
            return
        if separator:
            if not buffer.startswith(separator):
                raise ValueError("Invalid separator in Visual Genome JSON array")
            buffer = buffer[1:].lstrip()
        while True:
            try:
                record, end = decoder.raw_decode(buffer)
                break
            except json.JSONDecodeError:
                chunk = stream.read(1024 * 1024)
                if not chunk:
                    raise
                buffer += chunk
                buffer = buffer.lstrip()
        yield record
        buffer = buffer[end:]
        separator = ","


def clip_box(box, image_width, image_height):
    x, y, width, height = map(float, box)
    if not all(math.isfinite(value) for value in (x, y, width, height)) or width <= 0 or height <= 0:
        return None
    left = max(0.0, x)
    top = max(0.0, y)
    right = min(float(image_width), x + width)
    bottom = min(float(image_height), y + height)
    if left >= right or top >= bottom:
        return None
    return left, top, right - left, bottom - top


def region_pool(images, annotations, count, seed):
    records = {}
    for image_id, region_id, box, caption in annotations:
        if image_id not in images or not isinstance(caption, str) or not caption.strip():
            continue
        image = images[image_id]
        box = clip_box(box, image["width"], image["height"])
        if box is None:
            continue
        key = (image_id, box)
        fraction = box[2] * box[3] / (image["width"] * image["height"])
        group = "small" if fraction < 0.02 else "medium" if fraction < 0.10 else "large"
        record = records.setdefault(key, {
            "region_id": f"{image_id}:{region_id}",
            "image_id": image_id,
            "bbox_xywh": list(box),
            "references": [],
            "area_fraction": fraction,
            "area_group": group,
        })
        if caption.strip() not in record["references"]:
            record["references"].append(caption.strip())

    rng = random.Random(seed)
    grouped = defaultdict(list)
    for record in sorted(records.values(), key=lambda row: row["region_id"]):
        grouped[record["area_group"]].append(record)

    selected, spare = [], []
    for index, group in enumerate(AREA_GROUPS):
        target = count // len(AREA_GROUPS) + (index < count % len(AREA_GROUPS))
        candidates = grouped[group]
        rng.shuffle(candidates)
        selected.extend(candidates[:target])
        spare.extend(candidates[target:])
    if len(selected) < count:
        shortfall = count - len(selected)
        if len(spare) < shortfall:
            raise ValueError(f"Only {len(records)} valid regions for count={count}. Increase --image-pool.")
        selected.extend(rng.sample(spare, shortfall))
    if len({row["region_id"] for row in selected}) != count:
        raise ValueError("Duplicate region IDs associated with different boxes in input")
    return sorted(selected, key=lambda row: (row["image_id"], row["region_id"]))


def image_ids_for_pool(images, validation, sample_size, seed):
    available = sorted(set(images) & validation)
    if not available:
        raise ValueError("No images overlap the official DenseCap validation split")
    return set(random.Random(seed).sample(available, min(sample_size, len(available))))


def read_raw_vg(source_dir, validation, image_pool, seed):
    metadata_path = source_dir / "image_data.json.zip"
    descriptions_path = source_dir / "region_descriptions.json.zip"
    images = {int(row["image_id"]): row for row in read_json(metadata_path)
              if int(row["width"]) > 0 and int(row["height"]) > 0}
    selected_images = image_ids_for_pool(images, validation, image_pool, seed)
    selected_metadata = {key: images[key] for key in selected_images}
    annotations = []
    with source_json(descriptions_path) as stream:
        for item in json_array_items(stream):
            image_id = int(item["id"])
            if image_id in selected_images:
                for region in item["regions"]:
                    box = [region["x"], region["y"], region["width"], region["height"]]
                    annotations.append((image_id, region["region_id"], box, region["phrase"]))
    return selected_metadata, annotations, [metadata_path, descriptions_path]


def read_controlcap(path, validation, image_pool, seed):
    data = read_json(path)
    if any(int(image["id"]) not in validation for image in data["images"]):
        raise ValueError("The ControlCap annotation file contains images outside official validation. Use vg1.2/val.json.")
    images = {int(row["id"]): row for row in data["images"] if row["width"] > 0 and row["height"] > 0}
    selected_images = image_ids_for_pool(images, validation, image_pool, seed)
    selected_metadata = {key: images[key] for key in selected_images}
    annotations = []
    for item in data["annotations"]:
        image_id = int(item["image_id"])
        if image_id in selected_images:
            x1, y1, x2, y2 = item["bbox"]
            annotations.append((image_id, item["id"], [x1, y1, x2 - x1, y2 - y1], item["caption"]))
    return selected_metadata, annotations, [path]


def find_local_image(root, image_id):
    if root:
        for folder in (root, root / "VG_100K", root / "VG_100K_2"):
            candidate = folder / f"{image_id}.jpg"
            if candidate.is_file():
                return candidate
    return None


def verify_image(path, width, height):
    with Image.open(path) as image:
        if image.format != "JPEG" or image.size != (width, height):
            raise ValueError(f"Expected JPEG {width} x {height} at {path}, found {image.format} {image.size}")
        image.verify()


def place_image(image_id, metadata, output_dir, image_root, proxy, fetch):
    width, height = int(metadata["width"]), int(metadata["height"])
    path = find_local_image(image_root, image_id) or output_dir / "images" / f"{image_id}.jpg"
    if path.is_file():
        verify_image(path, width, height)
    elif fetch:
        filename = f"{image_id}.jpg"
        urls = [f"{IMAGE_HOST}/{name}/{filename}" for name in ("VG_100K", "VG_100K_2")]
        actual = urlparse(metadata.get("url", ""))
        if actual.hostname == "cs.stanford.edu" and actual.path.split("/")[-1] == filename:
            preferred = f"https://{actual.hostname}{actual.path}"
            if preferred in urls:
                urls.remove(preferred)
                urls.insert(0, preferred)
        failures = []
        for url in urls:
            try:
                download(url, path, proxy, lambda file: verify_image(file, width, height))
                break
            except RuntimeError as error:
                failures.append(str(error))
        else:
            raise RuntimeError("\n".join(failures))
    else:
        raise FileNotFoundError(f"Image {image_id} missing. Supply --image-root or pass --download.")
    return image_id, Path(os.path.relpath(path, output_dir)).as_posix()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/vg_val_1000/manifest.json"))
    parser.add_argument("--source-dir", type=Path, default=Path("data/vg_sources"))
    parser.add_argument("--controlcap-val", type=Path, help="Use existing converted ControlCap vg1.2/val.json")
    parser.add_argument("--image-root", type=Path, help="Reuse existing Visual Genome JPGs")
    parser.add_argument("--download", action="store_true", help="Fetch metadata and only the selected JPEGs")
    parser.add_argument("--proxy", help="Optional HTTP proxy URL for downloads")
    parser.add_argument("--count", type=int, default=1000)
    parser.add_argument("--image-pool", type=int, default=128)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if min(args.count, args.image_pool, args.workers) <= 0:
        parser.error("--count, --image-pool, and --workers must be positive")
    split_path = args.source_dir / "densecap_splits.json"
    if args.download:
        for name, url in SOURCES.items():
            if args.controlcap_val and name != split_path.name:
                continue
            print(f"Checking {name} ...", flush=True)
            validation = verify_archive if name.endswith(".zip") else read_split_ids
            download(url, args.source_dir / name, args.proxy, validation)
    validation = read_split_ids(split_path)
    if args.controlcap_val:
        images, annotations, sources = read_controlcap(args.controlcap_val, validation, args.image_pool, args.seed)
    else:
        images, annotations, sources = read_raw_vg(args.source_dir, validation, args.image_pool, args.seed)
    records = region_pool(images, annotations, args.count, args.seed)
    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image_ids = sorted({row["image_id"] for row in records})
    print(f"Checking {len(image_ids)} selected images for {len(records)} regions ...", flush=True)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(place_image, image_id, images[image_id], output_path.parent,
                               args.image_root, args.proxy, args.download) for image_id in image_ids]
        relative_paths = dict(future.result() for future in futures)
    for row in records:
        row["image_path"] = relative_paths[row["image_id"]]
    manifest = {
        "dataset": "Visual Genome v1.2 (ControlCap converted)" if args.controlcap_val else "Visual Genome v1.2 (raw)",
        "split": "val",
        "seed": args.seed,
        "densecap_split_revision": DENSECAP_REVISION,
        "image_pool_size": len(images),
        "image_count": len(image_ids),
        "region_count": len(records),
        "area_thresholds": [0.02, 0.10],
        "area_group_counts": dict(Counter(row["area_group"] for row in records)),
        "source_sha256": {path.name: sha256(path) for path in [split_path, *sources]},
        "regions": records,
    }
    payload = json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
    if output_path.exists():
        if output_path.read_text(encoding="utf-8") != payload:
            raise FileExistsError(f"A different manifest already exists at {output_path}")
    else:
        with output_path.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(payload)
    print(f"Prepared {len(records)} validation regions in {output_path}")


if __name__ == "__main__":
    main()
