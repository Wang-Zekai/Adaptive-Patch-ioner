"""Single-region extraction of the official frozen DINO visual features."""

from functools import lru_cache
import math

from PIL import Image
import torch
from torchvision import transforms as T

from multiscale import clip_box, detail_geometry


@lru_cache(maxsize=4)
def alternate_transform(size):
    return T.Compose([
        T.Resize((size, size), interpolation=T.InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])


def image_rgb(path):
    with Image.open(path) as file:
        return file.convert("RGB")


class VisualFeatures:
    def __init__(self, captioner, batch_size):
        from boundary_semantic import aggregate_patches, count_and_coherence

        self.captioner = captioner
        self.batch_size = batch_size
        self.device = captioner.device
        self.patch_size = captioner.patch_size
        self.dimension = captioner.model.embed_dim
        self.pool = aggregate_patches
        self.signals = count_and_coherence

    def transform(self, image, size):
        if size == self.captioner.model.resize_dim:
            return self.captioner.model.image_transforms_no_crop(image)
        if size <= 0 or size % self.patch_size:
            raise ValueError(f"Resolution must be a positive multiple of {self.patch_size}: {size}")
        return alternate_transform(size)(image)

    @torch.inference_mode()
    def encode(self, batch):
        tensor = torch.stack(batch).to(self.device)
        shape = tuple(dim // self.patch_size for dim in tensor.shape[-2:])
        try:
            patches = self.captioner.model.dino(tensor, is_training=True)["x_norm_patchtokens"]
        except torch.cuda.OutOfMemoryError:
            if len(batch) == 1:
                raise
        else:
            if patches.shape != (len(batch), math.prod(shape), self.dimension):
                raise RuntimeError(f"Unexpected patch shape {patches.shape} for grid {shape}")
            return patches, shape
        del tensor
        torch.cuda.empty_cache()
        midpoint = len(batch) // 2
        left, shape = self.encode(batch[:midpoint])
        right, _ = self.encode(batch[midpoint:])
        return torch.cat((left, right)), shape

    def pool_original(self, patches, shape, image, box, method):
        size = self.captioner.model.resize_dim
        if method == "official":
            dim_marker = torch.empty((0, 0, size, size))
            boxes = self.captioner.upstream_boxes(image, box, dim_marker)
            return self.captioner._original_feature(patches, boxes)
        resized = box.scaled(size / image.width, size / image.height)
        return self.pool(patches, resized, shape, self.patch_size, method)

    def pool_context(self, patches, shape, image, box, scale, method):
        size = self.captioner.model.resize_dim
        sx, sy = size / image.width, size / image.height
        outer = clip_box(box.expanded(scale).xywh, image.size).scaled(sx, sy)
        value = self.pool(patches, box.scaled(sx, sy), shape, self.patch_size, method, outer_box=outer)
        return value if value is not None else patches.new_zeros((1, self.dimension))

    def prepare_detail(self, image, box, scale, size):
        bounds, target = detail_geometry(box, image.size, scale)
        crop = image.crop(bounds)
        resized = target.scaled(size / crop.width, size / crop.height)
        return self.transform(crop, size), resized

    def get_signals(self, patches, shape, image, box):
        size = self.captioner.model.resize_dim
        resized = box.scaled(size / image.width, size / image.height)
        stats = self.signals(patches, resized, shape, self.patch_size)
        stats['area_fraction'] = box.w * box.h / (image.width * image.height)
        return stats

def tensor_signals(stats):
    return {name: torch.tensor([row[name] for row in stats], dtype=torch.float32) for name in stats[0]}
