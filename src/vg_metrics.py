"""Compute paired Visual Genome caption metrics from multiscale_experiment results."""

import argparse
from collections import defaultdict
import csv
from functools import partial
from importlib.metadata import PackageNotFoundError, version
import json
import math
from pathlib import Path
import re
import shutil
from statistics import mean
import subprocess
import sys
from threading import Lock


AREA_GROUPS = ("small", "medium", "large")
BLEU_KEYS = tuple(f"BLEU_{index}" for index in range(1, 5))
METRICS = ("CIDEr", "BLEU", "ROUGE_L", "METEOR", "SPICE", "RefPAC-S")
PAC_CHECKPOINT_URL = "https://drive.google.com/file/d/1F-0Pma-vfJPAiDzeyl-iEdSXZIO1cDae/view"


def paired_rows(rows):
    modes = defaultdict(dict)
    for row in rows:
        identifier = str(row["region_id"])
        refs = row["references"]
        if not isinstance(refs, list) or not refs or any(not isinstance(text, str) or not text.strip() for text in refs):
            raise ValueError(f"Region {identifier} has invalid references")
        if row["area_group"] not in AREA_GROUPS or not isinstance(row["caption"], str):
            raise ValueError(f"Invalid area group or caption for region {identifier}")
        mode = row["mode"]
        if identifier in modes[mode]:
            raise ValueError(f"Duplicate prediction for {mode}/{identifier}")
        modes[mode][identifier] = row
    if "baseline" not in modes or not modes["baseline"]:
        raise ValueError("Paired evaluation requires baseline predictions")
    baseline = modes["baseline"]
    for mode, by_region in modes.items():
        if by_region.keys() != baseline.keys():
            raise ValueError(f"Region IDs differ between baseline and {mode}")
        for identifier, row in by_region.items():
            expected = baseline[identifier]
            for key in ("image_id", "image_path", "bbox_xywh", "references", "area_group"):
                if row[key] != expected[key]:
                    raise ValueError(f"Paired field {key} differs for {mode}/{identifier}")
    return modes, list(baseline)


def tokenizer(mode):
    if mode == "ptb":
        if not shutil.which("java"):
            raise RuntimeError("Add a Java runtime to PATH for Stanford PTB tokenization")
        from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer

        tokenize = PTBTokenizer().tokenize

        def checked_tokenize(captions):
            output = tokenize(captions)
            if output.keys() != captions.keys() or any(len(output[key]) != len(values) for key, values in captions.items()):
                raise RuntimeError("Stanford PTB tokenizer did not return a tokenized row for every caption")
            return output

        return checked_tokenize

    def simple_tokenize(captions):
        return {key: [" ".join(re.findall(r"\w+(?:'\w+)?", row["caption"].lower())) for row in values]
                for key, values in captions.items()}

    return simple_tokenize


def make_scorers(metrics):
    scorers = {}
    if "CIDEr" in metrics:
        from pycocoevalcap.cider.cider import Cider

        scorers["CIDEr"] = Cider()
    if "BLEU" in metrics:
        from pycocoevalcap.bleu.bleu import Bleu

        scorers["BLEU"] = Bleu(4)
    if "ROUGE_L" in metrics:
        from pycocoevalcap.rouge.rouge import Rouge

        scorers["ROUGE_L"] = Rouge()
    if "METEOR" in metrics:
        from pycocoevalcap.meteor import meteor

        class LocalMeteor(meteor.Meteor):
            def __init__(self):
                # METEOR 1.5 double-escapes spaces when resolving its default paraphrase URL.
                self.meteor_cmd = ["java", "-Xmx2G", "-jar", meteor.METEOR_JAR,
                                   "-", "-", "-stdio", "-l", "en", "-norm", "-a", "data/paraphrase-en.gz"]
                self.lock = Lock()
                self.meteor_p = subprocess.Popen(
                    self.meteor_cmd, cwd=Path(meteor.__file__).parent,
                    stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                )

            def __del__(self):
                process = getattr(self, "meteor_p", None)
                if process is not None:
                    if process.poll() is None:
                        process.kill()
                    process.wait()
                    process.stdin.close()
                    process.stdout.close()

        scorers["METEOR"] = LocalMeteor()
    if "SPICE" in metrics:
        from pycocoevalcap.spice.spice import Spice

        scorers["SPICE"] = Spice()
    return scorers


def load_refpac(checkpoint, device):
    if checkpoint is None or not checkpoint.is_file():
        raise RuntimeError(f"RefPAC-S requires --pac-checkpoint downloaded from {PAC_CHECKPOINT_URL}")
    import torch
    from clip.model import build_model

    upstream = Path(__file__).resolve().parent / "upstream" / "Patch-ioner"
    sys.path.insert(0, str(upstream))
    from pacsMetric import pac_score

    # Preserve the upstream RefPAC score and avoid Windows worker subprocesses.
    pac_score.extract_all_captions = partial(pac_score.extract_all_captions, num_workers=0, batch_size=64)
    weights = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model = build_model(weights["state_dict"]).float().to(device).eval().requires_grad_(False)

    def refpac(references, candidates):
        with torch.inference_mode():
            return pac_score.RefPACScore(model, references, candidates, device,
                                         [len(text.split()) for text in candidates])

    return refpac


def finite(value):
    return float(value) if math.isfinite(float(value)) else None


def text_scores(scorers, refs, candidates, groups):
    corpus = {group: {} for group in ("all", *AREA_GROUPS)}
    instance = defaultdict(dict)
    for metric, scorer in scorers.items():
        total, per_region = scorer.compute_score(refs, candidates)
        if metric == "BLEU":
            corpus["all"].update(zip(BLEU_KEYS, map(finite, total)))
            for index, key in enumerate(BLEU_KEYS):
                for identifier, score in zip(refs, per_region[index]):
                    instance[identifier][key] = finite(score)
        else:
            order = refs
            if metric == "SPICE":
                order = sorted(refs)
                per_region = [value["All"]["f"] for value in per_region]
            corpus["all"][metric] = finite(total)
            for identifier, score in zip(order, per_region, strict=True):
                instance[identifier][metric] = finite(score)
        for group in AREA_GROUPS:
            ids = groups[group]
            if metric in ("BLEU", "METEOR"):
                keys = BLEU_KEYS if metric == "BLEU" else (metric,)
                if ids:
                    group_refs = {identifier: refs[identifier] for identifier in ids}
                    group_cands = {identifier: candidates[identifier] for identifier in ids}
                    group_total, _ = scorer.compute_score(group_refs, group_cands)
                    values = map(finite, group_total) if metric == "BLEU" else (finite(group_total),)
                else:
                    values = (None,) * len(keys)
                corpus[group].update(zip(keys, values))
            else:
                # For CIDEr, each size group shares document frequencies from the full subset.
                valid = [instance[identifier][metric] for identifier in ids
                         if instance[identifier][metric] is not None]
                corpus[group][metric] = mean(valid) if valid else None
    return corpus, instance


def score_predictions(results, metrics, tokenizer_name, pac_checkpoint=None, device="cpu"):
    modes, order = paired_rows(results["rows"])
    baseline = modes["baseline"]
    groups = {group: [key for key in order if baseline[key]["area_group"] == group] for group in AREA_GROUPS}
    scorers = make_scorers(metrics)
    if scorers:
        tokenize = tokenizer(tokenizer_name)
        refs = tokenize({key: [{"caption": value} for value in baseline[key]["references"]] for key in order})
    refpac = load_refpac(pac_checkpoint, device) if "RefPAC-S" in metrics else None
    by_summary = {(row["mode"], row["area_group"]): row for row in results.get("summary", [])}
    summary = []
    output_rows = []
    for mode, predictions in modes.items():
        if scorers:
            candidates = tokenize({key: [{"caption": predictions[key]["caption"]}] for key in order})
            corpus, instance = text_scores(scorers, refs, candidates, groups)
        else:
            corpus = {group: {} for group in ("all", *AREA_GROUPS)}
            instance = defaultdict(dict)
        if refpac:
            full_references = [baseline[key]["references"] for key in order]
            captions = [predictions[key]["caption"] for key in order]
            total, per_region = refpac(full_references, captions)
            corpus["all"]["RefPAC-S"] = finite(total)
            for identifier, score in zip(order, per_region):
                instance[identifier]["RefPAC-S"] = finite(score)
            for group, ids in groups.items():
                valid = [instance[identifier]["RefPAC-S"] for identifier in ids
                         if instance[identifier]["RefPAC-S"] is not None]
                corpus[group]["RefPAC-S"] = mean(valid) if valid else None
        for group, values in corpus.items():
            timing = by_summary.get((mode, group), {})
            count = len(order) if group == "all" else len(groups[group])
            summary.append({**timing, "mode": mode, "area_group": group, "regions": count, **values})
        output_rows.extend({**predictions[key], **instance[key]} for key in order)
    try:
        coco_version = version("pycocoevalcap") if scorers else None
    except PackageNotFoundError:
        coco_version = None
    return {
        "metadata": {**results.get("metadata", {}), "metrics": list(metrics), "tokenizer": tokenizer_name,
                     "pycocoevalcap_version": coco_version,
                     "cider_idf_corpus": "all paired validation regions" if "CIDEr" in scorers else None},
        "summary": summary,
        "rows": output_rows,
    }


def write_csv(path, rows):
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("x", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, ensure_ascii=False) if isinstance(value, (list, dict)) else value
                             for key, value in row.items()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--metrics", nargs="+", choices=METRICS, default=["CIDEr", "BLEU", "ROUGE_L"])
    parser.add_argument("--tokenizer", choices=("ptb", "simple"), default="ptb",
                        help="Use official Stanford PTB tokenization, or simple diagnostic tokenization.")
    parser.add_argument("--pac-checkpoint", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu", help="Device for RefPAC-S")
    args = parser.parse_args()
    chosen = tuple(dict.fromkeys(args.metrics))
    if any(metric in chosen for metric in ("METEOR", "SPICE")) and not shutil.which("java"):
        parser.error("METEOR and SPICE require a Java runtime in PATH")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    paths = [args.output_dir / name for name in ("scores.json", "summary.csv", "per_region.csv")]
    if any(path.exists() for path in paths):
        parser.error("Metric output files already exist. Use a new output directory.")
    predictions = json.loads(args.predictions.read_text(encoding="utf-8"))
    try:
        results = score_predictions(predictions, chosen, args.tokenizer, args.pac_checkpoint, args.device)
    except ModuleNotFoundError as error:
        parser.exit(2, f"Missing metric dependency: {error}. Install pycocoevalcap==1.2 for text metrics "
                       "and github.com/openai/CLIP for RefPAC-S.\n")
    with paths[0].open("x", encoding="utf-8") as stream:
        json.dump(results, stream, ensure_ascii=False, allow_nan=False, indent=2)
        stream.write("\n")
    write_csv(paths[1], results["summary"])
    write_csv(paths[2], results["rows"])
    print(f"Scored {len(results['rows'])} predictions in {args.output_dir}")


if __name__ == "__main__":
    main()
