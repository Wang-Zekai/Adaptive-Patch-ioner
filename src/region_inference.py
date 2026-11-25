"""Public entry point for the frozen policies used in the final experiments."""

from time import perf_counter

import torch

from multiscale import clip_box, load_captioner
from refinement_features import RefinementFeatures
from size_aware import compose, detail_specs, original_specs
from visual_features import tensor_signals


MIB = 1024**2
VISUAL_SIGNALS = {"mean_cosine", "variance_cosine", "dispersion", "aggregation_change", "coherence"}


class RegionCaptioner:
    """Share a single frozen model and its complete text memory across calls."""

    def __init__(self, device="cuda:0"):
        self.captioner = load_captioner(device)
        self.device = self.captioner.device
        self.visual = RefinementFeatures(self.captioner, batch_size=1)

        # load_captioner places the pinned upstream source first on sys.path.
        from src.decap import decap

        self.decap = decap
        model = self.captioner.model
        self.tokenizer = model._get_decoder_tokenizer()
        self.family = (model.decoder_config or {}).get("decoder_family")
        self.family = self.family.lower() if self.family else None

    def _fused(self, image, box, config):
        size = self.captioner.model.resize_dim
        patches, shape = self.visual.encode([self.visual.transform(image, size)])
        gate = config.get("gate", {})
        needed = {name for name, _ in gate.get("components", ())}
        if "fixed_signal" in gate:
            needed.add(gate["fixed_signal"])
        get_signals = (
            self.visual.get_signals(patches, shape, image, box)
            if needed & VISUAL_SIGNALS
            else self.visual.get_geometry_signals(image, box)
        )
        signals = tensor_signals([get_signals])
        original = self.visual.pool_original_variant(patches, shape, image, box, original_specs(config, signals)[0])
        address = detail_specs(config, signals)[0]
        if address is None:
            return original, 0.0, False

        scale, resolution, preserve_aspect, pool = address
        layout = "letterbox" if preserve_aspect else "square"
        tensor, target = self.visual.prepare_refined_detail(image, box, scale, resolution, layout)
        zoom_patches, zoom_shape = self.visual.encode([tensor])
        area_fraction = box.w * box.h / (image.width * image.height)
        detail = self.visual.pool_variant(zoom_patches, target, zoom_shape, pool, area_fraction=area_fraction)
        fused, weight = compose(config, original, detail, signals)
        return fused, weight.item(), True

    @torch.inference_mode()
    def predict(self, image, bbox, config):
        """Caption a single xywh box, with synchronized model-only latency."""
        box = clip_box(bbox, image.size)
        cuda = self.device.type == "cuda"
        if cuda:
            torch.cuda.synchronize(self.device)
            torch.cuda.reset_peak_memory_stats(self.device)
        start = perf_counter()
        if config["id"] == "baseline":
            caption = self.captioner.predict(image, bbox, "baseline")["caption"]
            weight, used_detail = 0.0, False
        else:
            fused, weight, used_detail = self._fused(image, box, config)
            model = self.captioner.model
            if not torch.isfinite(fused).all().item() or not (fused.norm(dim=-1) > 0).all().item():
                raise ValueError("Expected finite nonzero region features")
            prefix = model.im_proj.project(fused.clone(), normalize=model.normalize)
            captions = self.decap.decoding_batched(
                model.decoder, prefix, decoding_method=model.decoding_method,
                tokenizer=self.tokenizer, decoder_family=self.family,
            )
            caption = captions[0]
        if cuda:
            torch.cuda.synchronize(self.device)
        elapsed_ms = (perf_counter() - start) * 1e3
        return {
            "caption": caption.strip(),
            "bbox_xywh": list(box.xywh),
            "detail_active": used_detail,
            "detail_weight": weight,
            "inference_ms": elapsed_ms,
            "peak_allocated_mib": torch.cuda.max_memory_allocated(self.device) / MIB if cuda else None,
        }
