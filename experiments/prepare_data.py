"""Recreate the three published Visual Genome manifests from the official sources."""

import argparse
from pathlib import Path
import subprocess
import sys

from vg_subset import sha256


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
EXPECTED = {
    "vg_val_1000": "ed5a611309d4794b1cb85e7a90eca3cf9e88dffada64e5428c83dd0739e94783",
    "vg_test_2000": "3e462273493cb86ba53be3192481e1c0dff01b06c2973be3e6c2db8b501eefe2",
    "vg_test_5000": "0492e52c445b484dfa3848a1b7c7aa858915d02a2b046e4eab5750c8297d74b1",
}


def run_step(script, options, manifest, digest):
    subprocess.run([sys.executable, str(SCRIPT_DIR / script), *map(str, options)], cwd=ROOT, check=True)
    actual = sha256(manifest)
    if actual != digest:
        raise ValueError(f"{manifest}: expected the published SHA-256 {digest}, got {actual}")
    print(f"Confirmed published manifest {manifest}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data",
                        help="Expected location of the three dataset directories and their images")
    parser.add_argument("--source-dir", type=Path, help="Override the directory containing the three source files")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--download", action="store_true", help="Fetch the annotation archives and selected JPEGs")
    mode.add_argument("--verify", action="store_true", help="Check existing files offline (default)")
    parser.add_argument("--workers", type=int, default=8, help="Number of concurrent image checks or downloads")
    parser.add_argument("--proxy", help="Optional HTTP proxy for source and image downloads")
    args = parser.parse_args()
    if args.workers <= 0:
        parser.error("--workers must be positive")

    data_dir = args.data_dir.resolve()
    source_dir = args.source_dir.resolve() if args.source_dir else data_dir / "vg_sources"
    shared = ["--source-dir", source_dir, "--workers", args.workers]
    if args.proxy:
        shared.extend(["--proxy", args.proxy])
    validation = data_dir / "vg_val_1000/manifest.json"
    previous = data_dir / "vg_test_2000/manifest.json"
    final = data_dir / "vg_test_5000/manifest.json"
    fetch = ["--download"] if args.download else []
    test_mode = fetch or ["--verify-images"]
    try:
        run_step("vg_subset.py", [*shared, "--output", validation, *fetch], validation, EXPECTED["vg_val_1000"])
        run_step("vg_heldout.py", [*shared, "--validation-manifest", validation, "--output", previous, *test_mode],
                 previous, EXPECTED["vg_test_2000"])
        run_step("vg_final_test.py", [*shared, "--data-dir", data_dir, *test_mode], final, EXPECTED["vg_test_5000"])
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        parser.exit(1, f"Dataset preparation failed: {error}\n")


if __name__ == "__main__":
    main()
