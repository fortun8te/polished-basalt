@echo off
rem Build the rtx5080-win fork of Basalt into its OWN directories. Never touches the production
rem source (ai200-20261026) or the running server. CPU-only compile: run it while the GPU is idle
rem or accept a slower build; it does not use the GPU.
rem Copy this repo's working tree to FORK_SRC first (git archive or robocopy, excluding .git).
setlocal
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
if errorlevel 1 exit /b %errorlevel%
set "ROOT=C:\Users\pc-agent\AgentWorkspace\ai200-20261026"
set "FORK_SRC=C:\Users\pc-agent\AgentWorkspace\basalt-5080\src"
set "FORK_BUILD=C:\Users\pc-agent\AgentWorkspace\basalt-5080\build"
set "BASALT_PY=%ROOT%\.venv\Scripts\python.exe"
set "CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1"
set "PATH=%CUDA_PATH%\bin;%PATH%"
"C:\Program Files\CMake\bin\cmake.exe" -S "%FORK_SRC%" -B "%FORK_BUILD%" -G Ninja -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON ^
  -DCMAKE_MAKE_PROGRAM="%ROOT%\.venv\Scripts\ninja.exe" -DCMAKE_C_COMPILER=cl.exe -DCMAKE_CXX_COMPILER=cl.exe ^
  -DCMAKE_CUDA_COMPILER="%CUDA_PATH%\bin\nvcc.exe" -DCMAKE_CUDA_HOST_COMPILER="%VCToolsInstallDir%bin\Hostx64\x64\cl.exe" ^
  -DCUDAToolkit_ROOT="%CUDA_PATH%" -DCMAKE_CUDA_ARCHITECTURES=120 -DPython3_EXECUTABLE="%BASALT_PY%" -DIQ_FIXTURE_PYTHON="%BASALT_PY%"
if errorlevel 1 exit /b %errorlevel%
"C:\Program Files\CMake\bin\cmake.exe" --build "%FORK_BUILD%" --target basalt basalt-probe basalt-bench ^
  file_expert_source_test external_ple_test verified_file_test model_file_test mapped_file_test path_utf8_test ^
  platform_compiler_test platform_cpu_test platform_system_test iq_avx2_test iq_signs_test iq_avx512_signs_test pool_native_test ^
  tokenizer_test dec_gdn_parity --parallel 4
exit /b %errorlevel%
