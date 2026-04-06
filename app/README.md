# Scattering Explorer App

Desktop GUI application for classical and quantum scattering simulations with MP4 export.

## Structure

```
app/
├── main.py
├── simulations/
├── ui/
├── utils/
├── assets/
├── requirements.txt
└── ScatteringExplorer.spec
```

## Prerequisites

- Windows 10/11
- Python 3.10+
- Pip

## Install Dependencies

From repository root:

```bat
python -m pip install -r app\requirements.txt
```

## Run From Source

```bat
python app\main.py
```

## Build Standalone EXE

### Option A: Root build script (recommended)

```bat
build.bat
```

This produces:

- `dist/ScatteringExplorer.exe`

### Option B: Build from app folder with spec file

```bat
cd app
build.bat
```

This produces:

- `app/dist/ScatteringExplorer.exe`

## PyInstaller Command Used

The main build script uses the required one-file, windowed mode:

```bat
pyinstaller --onefile --windowed main.py --name ScatteringExplorer --add-data "classical_scattering.py;." --add-data "quantum_scattering.py;." --add-data "assets;assets" --collect-binaries imageio_ffmpeg --hidden-import PyQt5.QtMultimedia --hidden-import PyQt5.QtMultimediaWidgets
```

## Output Location for Generated MP4

Generated simulation videos are saved to:

- `%USERPROFILE%\Videos\ScatteringExplorer`

If `%USERPROFILE%\Videos` is unavailable, fallback:

- `%USERPROFILE%\ScatteringExplorer`

## Notes

- The executable is windowed (`--windowed`), so no console window opens.
- The executable can be launched directly by double-clicking `ScatteringExplorer.exe`.
- FFmpeg is bundled through `imageio-ffmpeg` integration for reliable MP4 encoding.
