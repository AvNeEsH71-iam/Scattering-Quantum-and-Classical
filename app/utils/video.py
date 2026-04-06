from __future__ import annotations

import os
import shutil
from datetime import datetime


def get_default_output_dir() -> str:
    videos_dir = os.path.join(os.path.expanduser("~"), "Videos", "ScatteringExplorer")
    if os.path.isdir(os.path.dirname(videos_dir)):
        os.makedirs(videos_dir, exist_ok=True)
        return videos_dir

    fallback_dir = os.path.join(os.path.expanduser("~"), "ScatteringExplorer")
    os.makedirs(fallback_dir, exist_ok=True)
    return fallback_dir


def build_temp_output_path(simulation_name: str, work_dir: str) -> str:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{simulation_name.lower()}_{stamp}.mp4"
    return os.path.join(work_dir, filename)


def copy_video(src_path: str, dst_path: str) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(dst_path)), exist_ok=True)
    shutil.copy2(src_path, dst_path)
    return dst_path
