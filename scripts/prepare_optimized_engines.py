from pathlib import Path
import shutil


FILES = {
    "engines/submodules/encode_image_fp16.engine": "engines/optimized/encode_image_fp16.engine",
    "engines/profile/read_memory_t5_full_fp16.engine": "engines/optimized/read_memory_fp16.engine",
    "engines/submodules/pixel_fusion_fp16.engine": "engines/optimized/pixel_fusion_fp16.engine",
    "engines/profile/segment_alpha_only_fp16.engine": "engines/optimized/segment_fp16.engine",
    "engines/submodules/encode_mask_fp16.engine": "engines/optimized/encode_mask_fp16.engine",
    "engines/submodules/first_frame_read_memory_fp16.engine": "engines/optimized/first_frame_read_memory_fp16.engine",
}


def main() -> None:
    for src, dst in FILES.items():
        src_path = Path(src)
        dst_path = Path(dst)
        if not src_path.exists():
            raise FileNotFoundError(src_path)
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_path, dst_path)
        print(f"{src_path} -> {dst_path}")


if __name__ == "__main__":
    main()
