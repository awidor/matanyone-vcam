import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

REPO_VENDOR = Path(__file__).resolve().parents[1] / "vendor" / "MatAnyone2"
sys.path.insert(0, str(REPO_VENDOR))

from matanyone2.inference.inference_core import InferenceCore
from matanyone2.model.utils.memory_utils import do_softmax, get_similarity
from matanyone2.utils.inference_utils import gen_dilate, gen_erosion
from matanyone2.utils.tensor_utils import pad_divide_by, unpad
from matanyone2_stateless import load_model


class StatelessMatAnyone:
    def __init__(self, model, *, top_k=30, mem_every=5):
        self.model = model
        self.top_k = top_k
        self.mem_every = mem_every
        self.curr_ti = -1
        self.last_mem_ti = 0
        self.memory_key = None
        self.memory_shrinkage = None
        self.memory_value = None
        self.obj_memory = None
        self.sensory = None
        self.last_mask = None
        self.last_pix_feat = None
        self.last_msk_value = None
        self.pad = None

    def _append_memory(self, key, shrinkage, mask_value, obj_value):
        key = key.unsqueeze(2)
        shrinkage = shrinkage.unsqueeze(2)
        mask_value = mask_value.unsqueeze(3)
        if self.memory_key is None:
            self.memory_key = key
            self.memory_shrinkage = shrinkage
            self.memory_value = mask_value
            self.obj_memory = obj_value.unsqueeze(2)
        else:
            self.memory_key = torch.cat([self.memory_key, key], dim=2)
            self.memory_shrinkage = torch.cat([self.memory_shrinkage, shrinkage], dim=2)
            self.memory_value = torch.cat([self.memory_value, mask_value], dim=3)
            self.obj_memory[:, :, 0, :, :-1] = self.obj_memory[:, :, 0, :, :-1] + obj_value[:, :, :, :-1]
            self.obj_memory[:, :, 0, :, -1:] = self.obj_memory[:, :, 0, :, -1:] + obj_value[:, :, :, -1:]

        max_tokens = (5 - 1) * key.shape[-2] * key.shape[-1]
        if self.memory_key.shape[-3] * self.memory_key.shape[-2] * self.memory_key.shape[-1] > max_tokens:
            # Keep the first frame and the most recent non-permanent memory.
            max_frames = 5
            self.memory_key = torch.cat([self.memory_key[:, :, :1], self.memory_key[:, :, -(max_frames - 1):]], dim=2)
            self.memory_shrinkage = torch.cat(
                [self.memory_shrinkage[:, :, :1], self.memory_shrinkage[:, :, -(max_frames - 1):]], dim=2
            )
            self.memory_value = torch.cat(
                [self.memory_value[:, :, :, :1], self.memory_value[:, :, :, -(max_frames - 1):]], dim=3
            )

    def _read_memory(self, pix_feat, key, selection):
        similarity = get_similarity(
            self.memory_key.flatten(start_dim=2),
            self.memory_shrinkage.flatten(start_dim=2),
            key.flatten(start_dim=2),
            selection.flatten(start_dim=2),
        )
        affinity = do_softmax(similarity, top_k=self.top_k, inplace=False)
        bs, num_objects, value_dim, _, h, w = self.memory_value.shape
        value = self.memory_value.flatten(start_dim=3).view(bs, num_objects * value_dim, -1)
        visual_readout = torch.bmm(value, affinity).view(bs, num_objects, value_dim, h, w)

        uncert_output = self.model.pred_uncertainty(
            self.last_pix_feat,
            pix_feat,
            self.last_mask,
            visual_readout[:, 0] - self.last_msk_value[:, 0],
        )
        uncert_prob = uncert_output["prob"].unsqueeze(1)
        visual_readout = visual_readout * uncert_prob + self.last_msk_value * (1 - uncert_prob)
        pixel_readout = self.model.pixel_fusion(pix_feat, visual_readout, self.sensory, self.last_mask)
        memory_readout, _ = self.model.readout_query(pixel_readout, self.obj_memory)
        return memory_readout

    def step(self, image, mask=None):
        self.curr_ti += 1
        image, self.pad = pad_divide_by(image, 16)
        image = image.unsqueeze(0)
        ms_feat, pix_feat = self.model.encode_image(image)
        key, shrinkage, selection = self.model.transform_key(ms_feat[0])

        is_mem_frame = mask is not None or (self.curr_ti - self.last_mem_ti >= self.mem_every)
        if mask is not None:
            mask, _ = pad_divide_by(mask, 16)
            self.last_mask = mask.unsqueeze(0).unsqueeze(0).float() / 255.0
            pred_prob_with_bg = torch.cat([1 - self.last_mask, self.last_mask], dim=1)[0]
            self.sensory = torch.zeros(
                (1, 1, self.model.sensory_dim, key.shape[-2], key.shape[-1]), device=image.device
            )
        else:
            memory_readout = self._read_memory(pix_feat, key, selection)
            self.sensory, _, pred_prob_with_bg = self.model.segment(
                ms_feat, memory_readout, self.sensory, chunk_size=-1, update_sensory=True
            )
            pred_prob_with_bg = pred_prob_with_bg[0]
            self.last_mask = pred_prob_with_bg[1:].unsqueeze(0)

        self.last_pix_feat = pix_feat
        if is_mem_frame:
            mask_value, self.sensory, obj_value, _ = self.model.encode_mask(
                image, pix_feat, self.sensory, self.last_mask, deep_update=True, chunk_size=-1, need_weights=False
            )
            self._append_memory(key, shrinkage, mask_value, obj_value)
            self.last_mem_ti = self.curr_ti
            self.last_msk_value = mask_value
        else:
            self.last_msk_value, _, _, _ = self.model.encode_mask(
                image, pix_feat, self.sensory, self.last_mask, deep_update=False, chunk_size=-1, need_weights=False
            )

        return unpad(pred_prob_with_bg, self.pad)


def read_frames(video_path, count):
    cap = cv2.VideoCapture(str(video_path))
    frames = []
    while len(frames) < count:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0)
    cap.release()
    return frames


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", default="vendor/MatAnyone2/inputs/video/test-sample2.mp4")
    parser.add_argument("--mask", default="vendor/MatAnyone2/inputs/mask/test-sample2.png")
    parser.add_argument("--frames", type=int, default=8)
    parser.add_argument("--tolerance", type=float, default=0.005)
    parser.add_argument("--stateless-top-k", type=int, default=30)
    parser.add_argument("--stateless-full-softmax", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda")
    model = load_model(device="cuda")
    original = InferenceCore(model, cfg=model.cfg, device=device)
    stateless_top_k = None if args.stateless_full_softmax else args.stateless_top_k
    stateless = StatelessMatAnyone(model, top_k=stateless_top_k, mem_every=model.cfg.mem_every)

    frames = [frame.to(device) for frame in read_frames(args.video, args.frames)]
    mask = np.array(Image.open(args.mask).convert("L"))
    mask = gen_dilate(mask, 10, 10)
    mask = gen_erosion(mask, 10, 10)
    mask = torch.from_numpy(mask).float().to(device)

    diffs = []
    with torch.inference_mode():
        for index, frame in enumerate(frames):
            if index == 0:
                ref = original.step(frame, mask, objects=[1])
                got = stateless.step(frame, mask)
            else:
                ref = original.step(frame)
                got = stateless.step(frame)
            diff = (ref - got).abs().mean().item()
            diffs.append(diff)
            print(f"frame={index} mean_abs_diff={diff:.6f}")

    mean = float(np.mean(diffs))
    print(f"overall_mean_abs_diff={mean:.6f}")
    if mean > args.tolerance:
        raise SystemExit(f"Parity failed: {mean:.6f} > {args.tolerance}")


if __name__ == "__main__":
    main()
