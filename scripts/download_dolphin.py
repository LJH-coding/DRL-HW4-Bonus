import os
from pathlib import Path
import platform

from common import download_file, extract_zip, extract_tar

DOLPHIN_WIN_URL = "https://github.com/VIPTankz/Wii-RL/releases/download/dolphin/dolphin0.zip"
DOLPHIN_MAC_ARM_URL = "https://github.com/unexploredtest/dolphin/releases/download/dolphin-wii-rl/DolphinMacArm.tar.gz"
DOLPHIN_MAC_X86_URL = "https://github.com/unexploredtest/dolphin/releases/download/dolphin-wii-rl/DolphinMacx86.tar.gz"
ZIP_NAME = "Dolphin.zip"
TAR_NAME = "Dolphin.tar.gz"

def _download_and_extract(url: str, extract_to: Path):
    url_lower = url.lower()
    if url_lower.endswith(".zip"):
        download_file(url, ZIP_NAME)
        extract_zip(ZIP_NAME, extract_to)
        os.remove(ZIP_NAME)
        return
    if url_lower.endswith(".tar.gz") or url_lower.endswith(".tgz"):
        download_file(url, TAR_NAME)
        extract_tar(TAR_NAME, extract_to)
        os.remove(TAR_NAME)
        return

    raise RuntimeError(
        "Unsupported archive format for Dolphin. "
        "Use .zip, .tar.gz, or .tgz."
    )

def main():
    current_directory = Path.cwd()

    if(platform.system() == "Windows"):
        _download_and_extract(DOLPHIN_WIN_URL, current_directory)
    elif(platform.system() == "Darwin" and platform.machine() == "arm64"):
        _download_and_extract(DOLPHIN_MAC_ARM_URL, current_directory)
    elif(platform.system() == "Darwin" and platform.machine() == "x86_64"):
        _download_and_extract(DOLPHIN_MAC_X86_URL, current_directory)
    else:
        raise RuntimeError(f"The operating system '{platform.system()}' is not supported.")

    print("Done!")


if __name__ == "__main__":
    main()