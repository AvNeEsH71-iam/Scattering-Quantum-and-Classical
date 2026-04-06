@echo off
setlocal

set PYTHON_EXE=%~dp0..\.venv\Scripts\python.exe
if not exist "%PYTHON_EXE%" set PYTHON_EXE=python

echo Cleaning app build artifacts...
if exist build rmdir /s /q build
if exist dist rmdir /s /q dist

echo Building ScatteringExplorer.exe from spec...
"%PYTHON_EXE%" -m PyInstaller --clean ScatteringExplorer.spec --collect-binaries imageio_ffmpeg
if errorlevel 1 (
    echo Build failed.
    exit /b 1
)

echo Build completed: dist\ScatteringExplorer.exe
endlocal
