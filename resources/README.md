# Whisper GUI Application - Required DLL Files

This application requires several DLL files to function properly. These files must be placed in the root directory of the application.

## Required DLL Files

### Core Files
These files are required for basic functionality:

- `whisper.dll` (or one of the alternative names: `whisper_cpp.dll`, `whisper-cpp.dll`, `whisperdll.dll`)
- `ggml.dll`
- `ggml-base.dll`
- `ggml-cpu.dll`

### GPU Acceleration
For GPU acceleration support (optional):

- `ggml-vulkan.dll`

## Notes on DLL Files

1. The application will automatically look for these files in the root directory.
2. If the Vulkan DLL is not found, the application will operate in CPU-only mode.
3. These DLLs are architecture-specific. Make sure to use the correct version (32-bit or 64-bit) matching your Python environment.
4. The Whisper DLL and related files can be obtained from the Whisper.cpp project.

## Model Files

In addition to DLL files, you'll need to download a Whisper model file. Common model sizes include:
- tiny
- base
- small
- medium
- large

Models can be downloaded from the official Whisper repository or from Whisper.cpp compatible sources.