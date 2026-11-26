# Method and evaluation

## Region features

1. Encode the full image with the original Patch-ioner visual encoder.
2. Optionally crop a 1.2× box around the target and resize the crop to 518 × 518 pixels. Map the original target coordinates into the crop, and pool only patches intersecting that target.
3. Combine the visual features before querying the full DeCap text memory and passing its output to the unchanged decoder.

The detail feature uses a normalized patch weight proportional to the patch's overlap with the target times its semantic similarity term. That term is the exponential of the patch's cosine similarity to the patch closest to the target center, shifted by 1 and divided by 0.25.

For size-aware fusion, the size signal is the geometric mean of two quantities clamped to [0, 1]: the number of covered patches divided by 150 and the target's shorter side in patch units divided by 10. Thresholds 0.24 and 0.58 set detail weights 0.3, 0.8, and 0.5. The weights sum to one across the original and detail features.

## Frozen presets

| Preset | Behavior |
| --- | --- |
| Baseline | Upstream full-image inference |
| Fixed detail | Original plus detail, with a fixed 0.3 detail weight |
| Previous best | Size-aware detail weight and overlap–semantic pooling in the detail branch |
| Quality | Previous best plus an original-image pool with 75% overlap-only features and 25% overlap–semantic features |
| Balanced | Quality with a pre-crop selection rule that uses patch support, mean cosine similarity, and variance of similarity |
| Efficient | Previous best with a pre-crop selection rule based on the dispersion of target patch features |
| Large-target fallback | Previous best, with the baseline selected for boxes covering at least 10% of the image |

The stored thresholds were calibrated before the held-out test. The inference interface takes image pixels and target coordinates. References are used by the separate evaluator after captions are generated.

## Evaluation

The refinement study screened 75 configurations on 240 validation regions, then evaluated 31 on 1,000 validation regions. Nine frozen configurations were subsequently evaluated on 5,000 regions from 512 official DenseCap test images, image-disjoint from both the validation data and the earlier 2,000-region test. Size groups use area-ratio thresholds of 2% and 10%. Reference metrics use Stanford PTB tokenization. Reported paired CIDEr intervals resample images 10,000 times and keep document frequencies fixed.

The 96-region timing audit was run after warmup in rotating method order. The CUDA peak is allocated GPU memory, which differs from the amount reserved by the allocator. In the audit, two online captions differed by one word from their batch-evaluation captions. Both differences occurred in the same region. The original batch scores are reported in the tables.

Test and validation CSVs include CIDEr, BLEU-1 through BLEU-4, ROUGE-L, and METEOR. RefPAC-S was not evaluated in this study. Visual Genome images, annotations, and the experimental prediction cache are not distributed here.

## Final-test detail-pooling ablation

The ablation uses the same 5,000 regions and 512 images as the published test. All three variants use the official full-image pool, a 1.2× detail crop resized to 518 × 518, and the same size-aware fusion weights. Each variant pools the patches with positive target overlap.

| Variant | Detail-crop pooling |
| --- | --- |
| A (`uniform_detail`) | Equal weight per patch |
| B (`size_only`) | Weight proportional to target overlap area |
| C (`current_best`) | Overlap area multiplied by the center-reference semantic term |

The run generates A with crop batches of up to eight and uses the published predictions for B and C. All predictions are paired by region and rescored with Stanford PTB tokenization and the same CIDEr document frequencies. The two differences use 10,000 paired bootstrap draws at the image level. The size-group intervals use no multiple-comparison adjustment.

Overall, B − A is +0.00140 CIDEr (95% interval [−0.00228, +0.00500]) and C − B is +0.01125 ([−0.00190, +0.02421]). The largest C − B gain is on small targets (+0.03292, [+0.01593, +0.05014]). See the [full results](../results/pooling_ablation.csv) and [reproduction commands](../README.md#detail-pooling-ablation).
