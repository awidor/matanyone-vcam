import numpy as np
import torch


_BG_CACHE = {}
_OUT_RGB_CACHE = {}


def composite_bgr_on_cpu(frame_bgr, alpha, color):
    """Composite BGR uint8 frame over a solid BGR color using a torch alpha tensor.

    Keeps the existing semantics (alpha clamped to [0, 1], uint8 BGR output) while
    avoiding normalized float frame/background temporaries.
    """
    a = alpha[0, 0].detach().clamp(0, 1).cpu().numpy()[..., None]
    bg = np.asarray(color, dtype=np.float32)
    out = np.empty_like(frame_bgr, dtype=np.float32)
    np.subtract(frame_bgr, bg, out=out, casting="unsafe")
    out *= a
    out += bg
    return out.astype(np.uint8)


def _bg_rgb_tensor(color, image_rgb):
    key = (image_rgb.device.type, image_rgb.device.index, image_rgb.dtype, tuple(color))
    bg = _BG_CACHE.get(key)
    if bg is None:
        bg = torch.tensor(
            (color[2] / 255.0, color[1] / 255.0, color[0] / 255.0),
            dtype=image_rgb.dtype,
            device=image_rgb.device,
        ).view(3, 1, 1)
        _BG_CACHE[key] = bg
    return bg


def _out_rgb_tensor(image_rgb):
    key = (image_rgb.device.type, image_rgb.device.index, image_rgb.dtype, tuple(image_rgb.shape))
    out = _OUT_RGB_CACHE.get(key)
    if out is None:
        out = torch.empty_like(image_rgb[0])
        _OUT_RGB_CACHE[key] = out
    return out


def composite_rgb_tensor_to_bgr(image_rgb, alpha, color):
    """Composite from the already-uploaded RGB image tensor and return BGR uint8."""
    bg = _bg_rgb_tensor(color, image_rgb)
    out_rgb = _out_rgb_tensor(image_rgb)
    torch.sub(image_rgb[0], bg, out=out_rgb)
    a = alpha[0].detach()
    a.clamp_(0, 1)
    out_rgb.mul_(a)
    out_rgb.add_(bg)
    out_bgr = out_rgb[[2, 1, 0]].permute(1, 2, 0).mul_(255.0).clamp_(0, 255).byte()
    return out_bgr.cpu().numpy()
