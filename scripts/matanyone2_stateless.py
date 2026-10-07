from pathlib import Path
import math
import urllib.request

import torch
import torch.nn.functional as F

from matanyone2.model.transformer.object_transformer import QueryTransformer
from matanyone2.model.utils.memory_utils import do_softmax, get_similarity
from matanyone2.utils.tensor_utils import aggregate
from matanyone2.utils.get_default_model import get_matanyone2_model


CHECKPOINT_URL = "https://github.com/pq-yang/MatAnyone2/releases/download/v1.0.0/matanyone2.pth"


def ensure_checkpoint(path: str | Path = "models/matanyone2.pth") -> Path:
    ckpt_path = Path(path)
    if not ckpt_path.exists():
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(CHECKPOINT_URL, ckpt_path)
    return ckpt_path


def load_model(ckpt: str | Path = "models/matanyone2.pth", device: str = "cuda"):
    ckpt_path = ensure_checkpoint(ckpt)
    return get_matanyone2_model(str(ckpt_path), device=torch.device(device)).eval()


class EncodeImageAndKey(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, image):
        ms_features, pix_feat = self.model.encode_image(image)
        key, shrinkage, selection = self.model.transform_key(ms_features[0])
        return (*ms_features, pix_feat, key, shrinkage, selection)


class MemoryReadout(torch.nn.Module):
    def __init__(self, top_k: int | None = 30):
        super().__init__()
        self.top_k = top_k

    def forward(self, query_key, query_selection, memory_key, memory_shrinkage, memory_value):
        # memory_key: B*CK*T*H*W
        # memory_value: B*N*CV*T*H*W
        bs, num_objects, value_dim, _, h, w = memory_value.shape
        similarity = get_similarity(
            memory_key.flatten(start_dim=2),
            memory_shrinkage.flatten(start_dim=2),
            query_key.flatten(start_dim=2),
            query_selection.flatten(start_dim=2),
        )
        affinity = do_softmax(similarity, top_k=self.top_k, inplace=False)
        value = memory_value.flatten(start_dim=3)
        value = value.view(bs, num_objects * value_dim, -1)
        readout = torch.bmm(value, affinity)
        return readout.view(bs, num_objects, value_dim, h, w)


class MemoryReadoutQueryMajor(torch.nn.Module):
    """MemoryReadout(top_k=None) laid out as plain attention for TensorRT.

    The anisotropic L2 similarity -a^2 + 2ab - b^2 (scaled by shrinkage) becomes one
    dot product over K = 2*CK + 1 (padded to a multiple of 8), so the distance is
    accumulated inside a single GEMM, and the softmax runs over the contiguous memory
    axis instead of a strided one.
    """

    def forward(self, query_key, query_selection, memory_key, memory_shrinkage, memory_value):
        bs, num_objects, value_dim, _, h, w = memory_value.shape
        ck = memory_key.shape[1]
        mk = memory_key.flatten(start_dim=2)  # B, CK, N
        ms = memory_shrinkage.flatten(start_dim=2)  # B, 1, N
        qk = query_key.flatten(start_dim=2)  # B, CK, HW
        qe = query_selection.flatten(start_dim=2)  # B, CK, HW

        b_sq = (qe * qk.pow(2)).sum(1, keepdim=True)
        query = torch.cat([qe, 2 * qk * qe, -b_sq], dim=1)
        memory = torch.cat([-mk.pow(2), mk, torch.ones_like(ms)], dim=1) * (ms / math.sqrt(ck))
        pad = (-query.shape[1]) % 8
        query = F.pad(query, (0, 0, 0, pad))
        memory = F.pad(memory, (0, 0, 0, pad))

        affinity = torch.softmax(query.transpose(1, 2) @ memory, dim=-1)  # B, HW, N
        value = memory_value.flatten(start_dim=3).view(bs, num_objects * value_dim, -1)
        readout = value @ affinity.transpose(1, 2)
        return readout.view(bs, num_objects, value_dim, h, w)


def _dense_aux_mask(self, logits, selector, seg_pass=False):
    """QueryTransformer._get_aux_mask without the data-dependent torch.where.

    Unblocks rows that would block every pixel, like the original, but with a dense
    op so the ONNX graph has no NonZero (which forces a device-to-host sync per run).
    """
    prob = logits.sigmoid() if selector is None else logits.sigmoid() * selector
    logits = aggregate(prob, dim=1)
    foreground_mask = (logits[:, 1:] >= logits.max(dim=1, keepdim=True)[0]).flatten(start_dim=2)
    aux_foreground_mask = (~foreground_mask).unsqueeze(2).unsqueeze(2).repeat(
        1, 1, self.num_heads, self.num_queries // 2, 1).flatten(start_dim=0, end_dim=2)
    aux_background_mask = foreground_mask.unsqueeze(2).unsqueeze(2).repeat(
        1, 1, self.num_heads, self.num_queries // 2, 1).flatten(start_dim=0, end_dim=2)
    aux_mask = torch.cat([aux_foreground_mask, aux_background_mask], dim=1)
    return aux_mask & ~aux_mask.all(dim=-1, keepdim=True)


def use_static_aux_mask():
    QueryTransformer._get_aux_mask = _dense_aux_mask


class MemoryReadoutWithUncertainty(torch.nn.Module):
    def __init__(self, model, top_k: int | None = None, query_major: bool = False):
        super().__init__()
        self.model = model
        self.read = MemoryReadoutQueryMajor() if query_major else MemoryReadout(top_k=top_k)

    def forward(
        self,
        query_key,
        query_selection,
        memory_key,
        memory_shrinkage,
        memory_value,
        last_pix_feat,
        pix_feat,
        last_pred_mask,
        last_msk_value,
    ):
        visual_readout = self.read(query_key, query_selection, memory_key, memory_shrinkage, memory_value)
        uncert_output = self.model.pred_uncertainty(
            last_pix_feat,
            pix_feat,
            last_pred_mask,
            visual_readout[:, 0] - last_msk_value[:, 0],
        )
        uncert_prob = uncert_output["prob"].unsqueeze(1)
        return visual_readout * uncert_prob + last_msk_value * (1 - uncert_prob)


class MemorySimilarity(torch.nn.Module):
    def forward(self, query_key, query_selection, memory_key, memory_shrinkage):
        return get_similarity(
            memory_key.flatten(start_dim=2),
            memory_shrinkage.flatten(start_dim=2),
            query_key.flatten(start_dim=2),
            query_selection.flatten(start_dim=2),
        )


class MemoryTopKSoftmax(torch.nn.Module):
    def __init__(self, top_k: int | None = 30):
        super().__init__()
        self.top_k = top_k

    def forward(self, similarity):
        return do_softmax(similarity, top_k=self.top_k, inplace=False)


class MemoryValueReadout(torch.nn.Module):
    def forward(self, affinity, memory_value):
        bs, num_objects, value_dim, _, h, w = memory_value.shape
        value = memory_value.flatten(start_dim=3)
        value = value.view(bs, num_objects * value_dim, -1)
        readout = torch.bmm(value, affinity)
        return readout.view(bs, num_objects, value_dim, h, w)


class PixelFusion(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, pix_feat, pixel_memory, sensory, last_mask):
        return self.model.pixel_fusion(pix_feat, pixel_memory, sensory, last_mask)


class ReadMemoryAndPixelFusion(torch.nn.Module):
    def __init__(self, model, top_k: int | None = None):
        super().__init__()
        self.model = model
        self.read = MemoryReadout(top_k=top_k)

    def forward(
        self,
        query_key,
        query_selection,
        memory_key,
        memory_shrinkage,
        memory_value,
        pix_feat,
        sensory,
        last_mask,
    ):
        pixel_memory = self.read(query_key, query_selection, memory_key, memory_shrinkage, memory_value)
        return self.model.pixel_fusion(pix_feat, pixel_memory, sensory, last_mask)


class FullMemoryRead(torch.nn.Module):
    def __init__(self, model, top_k: int | None = None, query_major: bool = False):
        super().__init__()
        self.model = model
        self.read = MemoryReadoutWithUncertainty(model, top_k=top_k, query_major=query_major)

    def forward(
        self,
        query_key,
        query_selection,
        memory_key,
        memory_shrinkage,
        memory_value,
        last_pix_feat,
        pix_feat,
        last_pred_mask,
        last_msk_value,
        sensory,
        obj_memory,
    ):
        visual_readout = self.read(
            query_key,
            query_selection,
            memory_key,
            memory_shrinkage,
            memory_value,
            last_pix_feat,
            pix_feat,
            last_pred_mask,
            last_msk_value,
        )
        pixel_readout = self.model.pixel_fusion(pix_feat, visual_readout, sensory, last_pred_mask)
        memory_readout, _ = self.model.readout_query(pixel_readout, obj_memory)
        return memory_readout


class Segment(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, f16, f8, f4, f2, f1, memory_readout, sensory):
        new_sensory, logits, prob = self.model.segment(
            [f16, f8, f4, f2, f1],
            memory_readout,
            sensory,
            update_sensory=True,
            clamp_mat=True,
        )
        return new_sensory, logits, prob


class SegmentAlphaOnly(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, f16, f8, f4, f2, f1, memory_readout, sensory):
        new_sensory, logits, _ = self.model.segment(
            [f16, f8, f4, f2, f1],
            memory_readout,
            sensory,
            update_sensory=True,
            clamp_mat=True,
        )
        return new_sensory, logits[:, 1:]


class SegmentAlphaOnlyNoF16(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, f8, f4, f2, f1, memory_readout, sensory):
        batch = f8.shape[0]
        f16 = torch.empty((batch, 1024, f8.shape[-2] // 2, f8.shape[-1] // 2), device=f8.device, dtype=f8.dtype)
        new_sensory, logits, _ = self.model.segment(
            [f16, f8, f4, f2, f1],
            memory_readout,
            sensory,
            update_sensory=True,
            clamp_mat=True,
        )
        return new_sensory, logits[:, 1:]


class EncodeMask(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, image, pix_feat, sensory, masks):
        image = (image - self.model.pixel_mean) / self.model.pixel_std
        mask_value, new_sensory = self.model.mask_encoder(
            image,
            pix_feat,
            sensory,
            masks,
            None,
            deep_update=True,
            chunk_size=-1,
        )
        object_summaries, object_logits = self.model.object_summarizer(masks, mask_value, False)
        return mask_value, new_sensory, object_summaries


class EncodeMaskShallow(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, image, pix_feat, sensory, masks):
        image = (image - self.model.pixel_mean) / self.model.pixel_std
        mask_value, _ = self.model.mask_encoder(
            image,
            pix_feat,
            sensory,
            masks,
            None,
            deep_update=False,
            chunk_size=-1,
        )
        return mask_value


class FirstFrameReadMemory(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, pix_feat, last_msk_value, sensory, last_mask, obj_memory):
        pixel_readout = self.model.pixel_fusion(pix_feat, last_msk_value, sensory, last_mask)
        memory_readout, _ = self.model.readout_query(pixel_readout, obj_memory)
        return memory_readout
