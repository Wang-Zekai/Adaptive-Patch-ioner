"""Generate the missing equal-weight captions and rescore the paired ablation."""

import argparse
from contextlib import closing
import hashlib
from itertools import groupby
import json
from pathlib import Path
import shutil
import sqlite3

import yaml

from evaluate import ROOT, read_regions, score_predictions, write_csv


EXISTING_MODES = ('baseline', 'size_only', 'current_best')
REGION_FIELDS = ('image_id', 'image_path', 'bbox_xywh', 'references', 'area_group')


def load_archive(path, digest, regions, methods):
    saved = json.loads(path.read_bytes())
    metadata = saved['metadata']
    if metadata['manifest_sha256'] != digest or metadata['tokenizer'] != 'ptb':
        raise ValueError('Archive manifest hash or tokenizer does not match this evaluation')

    for mode in EXISTING_MODES:
        previous = dict(metadata['configs'][mode])
        current = dict(methods[mode])
        for key in ('family', 'history_id'):
            previous.pop(key, None)
        if mode != 'baseline':
            previous.setdefault('original_pool', {'method': 'official'})
            previous.setdefault('detail_pool', {'method': 'overlap'})
        if previous != current:
            raise ValueError(f'Archived config differs from the evaluation config for {mode}')

    expected = {str(region['region_id']): region for region, _ in regions}
    rows = {mode: {} for mode in EXISTING_MODES}
    for item in saved['rows']:
        mode = item['mode']
        key = str(item['region_id'])
        if mode not in rows or key not in expected:
            continue
        region = expected[key]
        if key in rows[mode] or any(item[field] != region[field] for field in REGION_FIELDS):
            raise ValueError(f'Duplicate or mismatched archive row for {mode}/{key}')
        rows[mode][key] = {**region, 'mode': mode, 'caption': item['caption']}

    for mode, by_region in rows.items():
        if by_region.keys() != expected.keys():
            raise ValueError(f'Missing archive regions for {mode}')
    return [rows[mode][key] for mode in EXISTING_MODES for key in expected]


def caption_batch(model, image, boxes, config, original_grid):
    import torch

    from multiscale import clip_box
    from size_aware import compose, detail_specs, original_specs
    from visual_features import tensor_signals

    visual = model.visual
    boxes = [clip_box(box, image.size) for box in boxes]
    with torch.inference_mode():
        if original_grid is None:
            size = model.captioner.model.resize_dim
            original_grid = visual.encode([visual.transform(image, size)])
        original_patches, original_shape = original_grid
        signals = tensor_signals([visual.get_geometry_signals(image, box) for box in boxes])
        original = torch.cat([
            visual.pool_original_variant(original_patches, original_shape, image, box, pool)
            for box, pool in zip(boxes, original_specs(config, signals), strict=True)
        ])
        cropped = []
        for box, address in zip(boxes, detail_specs(config, signals), strict=True):
            if address is None:
                raise ValueError('The uniform pooling experiment requires a detail crop for every region')
            scale, resolution, preserve_aspect, pool = address
            layout = 'letterbox' if preserve_aspect else 'square'
            tensor, target = visual.prepare_refined_detail(image, box, scale, resolution, layout)
            cropped.append((tensor, target, pool))
        zoomed, zoom_shape = visual.encode([tensor for tensor, _, _ in cropped])
        detail = torch.cat([
            visual.pool_variant(zoomed[index:index + 1], target, zoom_shape, pool)
            for index, (_, target, pool) in enumerate(cropped)
        ])
        fused, _ = compose(config, original, detail, signals)
        if not torch.isfinite(fused).all().item() or not (fused.norm(dim=-1) > 0).all().item():
            raise ValueError('Expected finite nonzero region features')
        frozen = model.captioner.model
        prefix = frozen.im_proj.project(fused, normalize=frozen.normalize)
        captions = model.decap.decoding_batched(
            frozen.decoder, prefix, decoding_method=frozen.decoding_method,
            tokenizer=model.tokenizer, decoder_family=model.family,
        )
    return [caption.strip() for caption in captions], original_grid


def generate_uniform(regions, config, device, checkpoint, signature, batch_size=1):
    from region_inference import RegionCaptioner
    from visual_features import image_rgb

    with closing(sqlite3.connect(checkpoint)) as db:
        with db:
            db.execute('CREATE TABLE IF NOT EXISTS run (signature TEXT NOT NULL)')
            db.execute('CREATE TABLE IF NOT EXISTS predictions (region_id TEXT PRIMARY KEY, caption TEXT NOT NULL)')
            recorded = db.execute('SELECT signature FROM run').fetchone()
            if recorded is None:
                db.execute('INSERT INTO run VALUES (?)', (signature,))
            elif recorded[0] != signature:
                raise ValueError('Checkpoint belongs to another run. Choose a new output directory.')

        captions = dict(db.execute('SELECT region_id, caption FROM predictions'))
        print(f'Resuming with {len(captions)} saved A captions out of {len(regions)}', flush=True)
        model = None
        pending = ((r, path) for r, path in regions if str(r['region_id']) not in captions)
        for path, group in groupby(pending, key=lambda item: item[1]):
            if model is None:
                model = RegionCaptioner(device)
            image = image_rgb(path)
            group = [region for region, _ in group]
            original_grid = None
            for start in range(0, len(group), batch_size):
                batch = group[start:start + batch_size]
                boxes = [r['bbox_xywh'] for r in batch]
                if batch_size == 1:
                    prediction = model.predict(image, boxes[0], config)
                    if not prediction['detail_active']:
                        raise RuntimeError(f'Expected detail branch to run for region {batch[0]["region_id"]}')
                    new_captions = [prediction['caption']]
                else:
                    new_captions, original_grid = caption_batch(model, image, boxes, config, original_grid)
                records = [(str(r['region_id']), text) for r, text in zip(batch, new_captions, strict=True)]
                with db:
                    db.executemany('INSERT INTO predictions VALUES (?, ?)', records)
                previous_count = len(captions)
                captions.update(records)
                if len(captions) // 100 != previous_count // 100 or len(captions) == len(regions):
                    print(f'A captions ready: {len(captions)}/{len(regions)}', flush=True)
        return [{**r, 'mode': config['id'], 'caption': captions[str(r['region_id'])]} for r, _ in regions]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, default=ROOT / 'data/vg_test_5000/manifest.json')
    parser.add_argument('--image-root', type=Path, help='Base directory for image_path in the manifest')
    parser.add_argument('--archived-scores', type=Path, required=True, help='scores.json from the published test run')
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--limit', type=int, help='Only evaluate the first N regions for a smoke test')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--batch-size', type=int, default=1, help='Batch crops within each image')
    parser.add_argument('--check', action='store_true', help='Verify all archive IDs and image paths without inference')
    args = parser.parse_args()
    if args.limit is not None and args.limit <= 0:
        parser.error('--limit must be positive')
    if args.batch_size <= 0:
        parser.error('--batch-size must be positive')
    if not args.check and args.output_dir is None:
        parser.error('--output-dir is required for inference')

    try:
        config = yaml.safe_load((ROOT / 'configs/pooling_ablation.yaml').read_text(encoding='utf-8'))['methods']
        manifest = args.manifest.resolve()
        regions, digest = read_regions(manifest, args.image_root or manifest.parent)
        archived_rows = load_archive(args.archived_scores, digest, regions, config)
        print(f'Verified {len(regions)} regions, image paths, and {len(archived_rows)} cached captions', flush=True)
        if args.check:
            return
        if not shutil.which('java'):
            raise RuntimeError('Add Java to PATH for the original PTB tokenizer before running inference')

        regions = regions[:args.limit]
        ids = {str(region['region_id']) for region, _ in regions}
        archived_rows = [row for row in archived_rows if str(row['region_id']) in ids]
        output = args.output_dir
        output.mkdir(parents=True, exist_ok=True)
        mode_a = config['uniform_detail']
        source_files = sorted((ROOT / 'src').glob('*.py'))
        source_hashes = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in source_files}
        metadata = {
            'manifest_sha256': digest, 'manifest': str(manifest), 'evaluated_regions': len(regions),
            'device': args.device, 'batch_size': args.batch_size, 'methods': config, 'source_hashes': source_hashes,
            'fresh_modes': ['uniform_detail'], 'archived_modes': list(EXISTING_MODES),
        }
        signature = json.dumps(metadata, sort_keys=True)
        uniform = generate_uniform(regions, mode_a, args.device, output / 'uniform.sqlite3', signature, args.batch_size)
        result = score_predictions({'metadata': metadata, 'rows': archived_rows + uniform}, ('CIDEr',), 'ptb')
        result['metadata']['cider_idf_corpus'] = 'all paired regions in this run'
        (output / 'scores.json').write_text(json.dumps(result, ensure_ascii=False, allow_nan=False), encoding='utf-8')
        write_csv(output / 'summary.csv', result['summary'])
        write_csv(output / 'per_region.csv', result['rows'])
        print(f'Scored {len(regions)} regions in {output}')
    except (OSError, KeyError, ValueError, RuntimeError, sqlite3.Error, ImportError) as error:
        parser.exit(2, f'Error: {error}\n')


if __name__ == '__main__':
    main()
