@echo off
setlocal

set PYTHON_EXE=%~dp0.venv\Scripts\python.exe
if not exist "%PYTHON_EXE%" set PYTHON_EXE=python
set ROOT_DIR=%~dp0

echo [1/3] Cleaning previous PyInstaller artifacts...
if exist "%ROOT_DIR%build" rmdir /s /q "%ROOT_DIR%build"
if exist "%ROOT_DIR%dist" rmdir /s /q "%ROOT_DIR%dist"
if exist "%ROOT_DIR%app\build" rmdir /s /q "%ROOT_DIR%app\build"
if exist "%ROOT_DIR%app\dist" rmdir /s /q "%ROOT_DIR%app\dist"

echo [2/3] Building ScatteringExplorer.exe...
pushd "%ROOT_DIR%app"
"%PYTHON_EXE%" -m PyInstaller --onefile --windowed main.py --name ScatteringExplorer --add-data "classical_scattering.py;." --add-data "quantum_scattering.py;." --add-data "assets;assets" --collect-binaries imageio_ffmpeg --hidden-import PyQt5.QtMultimedia --hidden-import PyQt5.QtMultimediaWidgets
if errorlevel 1 (
    popd
    echo Build failed.
    exit /b 1
)
popd

echo [3/3] Finalizing dist output...
if not exist "%ROOT_DIR%dist" mkdir "%ROOT_DIR%dist"
copy /y "%ROOT_DIR%app\dist\ScatteringExplorer.exe" "%ROOT_DIR%dist\ScatteringExplorer.exe" > nul

echo Build completed: dist\ScatteringExplorer.exe
endlocal
