import argparse
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="output/first_frame.png")
    parser.add_argument("--ffmpeg-dir", default="C:/Tools/ffmpeg-2026-05-06-git-f2e5eff3ff-full_build")
    args = parser.parse_args()

    ffmpeg = str(Path(args.ffmpeg_dir) / "bin" / "ffmpeg.exe")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            args.input,
            "-frames:v",
            "1",
            "-vf",
            "scale=1280:720",
            str(output),
        ],
        check=True,
    )
    print(f"wrote={output}")


if __name__ == "__main__":
    main()
