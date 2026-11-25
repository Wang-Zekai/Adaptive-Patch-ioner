"""Multi-scale inference for the pinned DINOv2 + DeCap Memory Patch-ioner model."""

from dataclasses import dataclass
import math
import os
from pathlib import Path
import sys

from PIL import Image
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
ASSET_ROOT = Path(os.environ.get("PATCH_IONER_ASSET_ROOT", ROOT)).resolve()
UPSTREAM = ASSET_ROOT / "upstream"
ASSETS = ASSET_ROOT / "assets"
MODEL_DIR = ASSETS / "talk2dino-decap"
DINO_CACHE = ASSETS / "torch" / "hub"


BRANCHES = ("original", "detail", "context")
DETAIL_SCALES = (1.0, 1.2, 1.5)
CONTEXT_SCALES = (1.5, 2.0, 3.0)
DEFAULT_WEIGHTS = {
    "baseline": (1.0, 0.0, 0.0),
    "detail_only": (0.0, 1.0, 0.0),
    "detail_fusion": (0.7, 0.3, 0.0),
    "context_fusion": (0.8, 0.0, 0.2),
    "multiscale_fusion": (0.6, 0.3, 0.1),
}
MODES = tuple(DEFAULT_WEIGHTS)


@dataclass(frozen=True)
class Box:
    x: float
    y: float
    w: float
    h: float

    @property
    def xywh(self):
        return (self.x, self.y, self.w, self.h)

    def scaled(self, sx, sy):
        return Box(self.x * sx, self.y * sy, self.w * sx, self.h * sy)

    def expanded(self, scale):
        dx = self.w * (scale - 1) / 2
        dy = self.h * (scale - 1) / 2
        return Box(self.x - dx, self.y - dy, self.w * scale, self.h * scale)


def clip_box(bbox, image_size):
    """Validate xywh and clip partially visible boxes to the image."""
    x, y, w, h = map(float, bbox)
    if not all(map(math.isfinite, (x, y, w, h, x + w, y + h))) or w <= 0 or h <= 0:
        raise ValueError(f"Expected finite xywh with positive width and height: {bbox}")
    width, height = image_size
    if x >= 0 and y >= 0 and x + w <= width and y + h <= height and x + w > x and y + h > y:
        return Box(x, y, w, h)
    left, top = max(0.0, x), max(0.0, y)
    right, bottom = min(float(width), x + w), min(float(height), y + h)
    if right <= left or bottom <= top:
        raise ValueError(f"Box does not intersect the {width}x{height} image: {bbox}")
    return Box(left, top, right - left, bottom - top)


def detail_geometry(box, image_size, scale):
    """Return the actual integer PIL crop and target xywh within that crop."""
    outer = clip_box(box.expanded(scale).xywh, image_size)
    left, top = math.floor(outer.x), math.floor(outer.y)
    right, bottom = math.ceil(outer.x + outer.w), math.ceil(outer.y + outer.h)
    crop = (left, top, min(right, image_size[0]), min(bottom, image_size[1]))
    return crop, Box(box.x - left, box.y - top, box.w, box.h)


def patch_overlap(box, grid_shape, patch_size):
    """Return per-cell intersection areas on the resized image in pixel units."""
    rows, cols = grid_shape
    x = torch.arange(cols, dtype=torch.float64) * patch_size
    y = torch.arange(rows, dtype=torch.float64) * patch_size
    dx = (x.add(patch_size).clamp(max=box.x + box.w) - x.clamp(min=box.x)).clamp_min(0)
    dy = (y.add(patch_size).clamp(max=box.y + box.h) - y.clamp(min=box.y)).clamp_min(0)
    return dy[:, None] * dx[None, :]


def context_patch_weights(target, outer, grid_shape, patch_size):
    """Exclude every cell touching the target, even if it overlaps the ring too."""
    outer_area = patch_overlap(outer, grid_shape, patch_size)
    return outer_area.masked_fill(patch_overlap(target, grid_shape, patch_size) > 0, 0)


def weighted_pool(patches, area):
    total = area.sum()
    if total == 0:
        return None
    weights = (area.flatten() / total).to(device=patches.device, dtype=patches.dtype)
    return (patches * weights[None, :, None]).sum(dim=1)


def resolve_weights(mode, weights):
    if mode not in DEFAULT_WEIGHTS:
        raise ValueError(f"Unknown mode: {mode}")
    if weights is None:
        return DEFAULT_WEIGHTS[mode]
    if mode in ("baseline", "detail_only"):
        raise ValueError(f"{mode} has fixed weights")
    values = tuple(map(float, weights))
    if len(values) != len(BRANCHES) or any(not math.isfinite(w) or w < 0 for w in values):
        raise ValueError("Weights must contain three finite nonnegative numbers: original detail context")
    largest = max(values)
    if largest == 0:
        raise ValueError("At least one fusion weight must be positive")
    scaled = tuple(weight / largest for weight in values)
    total = sum(scaled)
    return tuple(weight / total for weight in scaled)


def fuse_features(features, weights):
    active = [(name, weight) for name, weight in zip(BRANCHES, weights) if weight and features.get(name) is not None]
    if not active:
        active = [("original", 1.0)]
    total = sum(weight for _, weight in active)
    effective = tuple(next((weight / total for name, weight in active if name == branch), 0.0) for branch in BRANCHES)
    if len(active) == 1:
        return features[active[0][0]], effective
    return sum(features[name] * (weight / total) for name, weight in active), effective


class MultiScaleCaptioner:
    def __init__(self, model, device, extract_bboxes_feats, adjust_bbox):
        if model.backbone_type != "DINO" or model.groupnet is not None or model.im_proj is None:
            raise ValueError("Expected the pinned DINOv2 + DeCap Memory Patch-ioner variant")
        self.model = model.eval().requires_grad_(False)
        self.device = torch.device(device)
        self.extract_bboxes_feats = extract_bboxes_feats
        self.adjust_bbox = adjust_bbox
        self.patch_size = model.patch_size

    def prepare_image(self, image):
        return self.model.image_transforms_no_crop(image).unsqueeze(0).to(self.device)

    def _patches(self, tensor):
        # This is the DINO branch of the upstream forward, before memory lookup.
        patches = self.model.dino(tensor, is_training=True)["x_norm_patchtokens"]
        shape = tuple(size // self.patch_size for size in tensor.shape[-2:])
        if patches.shape != (1, math.prod(shape), self.model.embed_dim):
            raise RuntimeError(f"Unexpected DINO patch shape: {tuple(patches.shape)} for grid {shape}")
        return patches, shape

    def upstream_boxes(self, image, box, tensor):
        height, width = tensor.shape[-2:]
        boxes = torch.tensor([[self.adjust_bbox(image, box.xywh, width, height)]], dtype=torch.float32)
        # A subpixel box can round its origin onto the far image edge in float32.
        upper = torch.nextafter(torch.tensor([width, height], dtype=boxes.dtype), torch.zeros(2))
        boxes[..., :2] = torch.minimum(boxes[..., :2], upper)
        return boxes

    def _original_feature(self, patches, boxes):
        # The upstream function performs in-place quantization on its box tensor.
        return self.extract_bboxes_feats(
            patches, boxes.clone(), gaussian_avg=False, patch_size=self.patch_size
        ).reshape(1, self.model.embed_dim)

    @torch.inference_mode()
    def extract_features(self, image, box, branches=BRANCHES, *, detail_scale=1.2, context_scale=2.0):
        if detail_scale not in DETAIL_SCALES or context_scale not in CONTEXT_SCALES:
            raise ValueError(f"Supported detail scales {DETAIL_SCALES} and context scales {CONTEXT_SCALES}")
        features = {}
        if "original" in branches or "context" in branches:
            original = self.prepare_image(image)
            patches, shape = self._patches(original)
            if "original" in branches:
                features["original"] = self._original_feature(patches, self.upstream_boxes(image, box, original))
            if "context" in branches:
                height, width = original.shape[-2:]
                sx, sy = width / image.width, height / image.height
                outer = clip_box(box.expanded(context_scale).xywh, image.size)
                weights = context_patch_weights(box.scaled(sx, sy), outer.scaled(sx, sy), shape, self.patch_size)
                features["context"] = weighted_pool(patches, weights)
                if features["context"] is None and "original" not in features and "detail" not in branches:
                    features["original"] = self._original_feature(patches, self.upstream_boxes(image, box, original))
            del original, patches

        if "detail" in branches:
            crop, target = detail_geometry(box, image.size, detail_scale)
            crop_image = image.crop(crop)
            tensor = self.prepare_image(crop_image)
            patches, shape = self._patches(tensor)
            height, width = tensor.shape[-2:]
            target = target.scaled(width / crop_image.width, height / crop_image.height)
            features["detail"] = weighted_pool(patches, patch_overlap(target, shape, self.patch_size))
            if features["detail"] is None:
                raise RuntimeError("Zoomed target has no overlapping visual patches")
        return features

    @torch.inference_mode()
    def predict(self, image, bbox, mode, *, detail_scale=1.2, context_scale=2.0, weights=None):
        values = resolve_weights(mode, weights)
        if image.mode != "RGB":
            image = image.convert("RGB")
        box = clip_box(bbox, image.size)
        context_available = None
        if mode == "baseline":
            tensor = self.prepare_image(image)
            boxes = self.upstream_boxes(image, box, tensor)
            caption = self.model(tensor, bboxes=boxes, get_cls_capt=False, gaussian_avg=False)["bbox_capts"][0][0]
            effective_weights = values
        else:
            branches = tuple(name for name, value in zip(BRANCHES, values) if value > 0)
            features = self.extract_features(image, box, branches, detail_scale=detail_scale, context_scale=context_scale)
            if "context" in features:
                context_available = features["context"] is not None
            fused, effective_weights = fuse_features(features, values)
            if not torch.isfinite(fused).all().item() or torch.linalg.vector_norm(fused).item() <= 0:
                raise RuntimeError("The fused feature is nonfinite or has zero norm")
            # Im2TxtProjector normalizes its input in place. Keep visual features intact.
            caption = self.model.caption_tokens(fused.clone())[0]
        return {
            "caption": caption,
            "bbox_xywh": box.xywh,
            "area_fraction": box.w * box.h / (image.width * image.height),
            "context_available": context_available,
            "effective_weights": effective_weights,
        }


def load_captioner(device="cuda:0"):
    """Load the existing local official weights and full memory, used by every mode."""
    device = torch.device(device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable in the current Conda environment")
    os.environ["IM2TXT_MEMORY_PATH"] = str(MODEL_DIR)
    os.environ["TORCH_HOME"] = str(DINO_CACHE.parent)
    os.environ["HF_HOME"] = str(ASSETS / "huggingface")
    os.environ["HF_HUB_CACHE"] = str(ASSETS / "huggingface" / "hub")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ.pop("TRANSFORMERS_CACHE", None)
    sys.path.insert(0, str(UPSTREAM / "Patch-ioner"))

    from src import Patchioner
    from src.bbox_utils import adjust_bbox_for_transform_no_scale, extract_bboxes_feats

    config = yaml.safe_load((MODEL_DIR / "config.yaml").read_text(encoding="utf-8"))
    decoder = MODEL_DIR / "coco_karpathy-009.pt"
    memory = MODEL_DIR / "coco_captions_text_embeddings-B16-ViT-B.16-591753.h5"
    dino = DINO_CACHE / "checkpoints" / "dinov2_vitb14_reg4_pretrain.pth"
    for path, size in ((decoder, 273331640), (memory, 1869966480), (dino, 346393545)):
        if not path.is_file() or path.stat().st_size != size:
            raise FileNotFoundError(f"Missing or incomplete official asset: {path}")
    config["decap_weights"] = str(decoder)
    with torch.inference_mode():
        model = Patchioner.from_config(config, device=str(device))
        return MultiScaleCaptioner(model, device, extract_bboxes_feats, adjust_bbox_for_transform_no_scale)
