@echo off
rem ============================================================================
rem  StartLux-Decision on the GTX 1050 (3 GB) - GPU inference server
rem    llama.cpp Vulkan build, weights + KV cache both on the GPU (~1286 MiB).
rem    -np 4 matters: with -np 1 the single slot is overwritten by every new
rem    question and the prefix cache never survives; with 4 slots the three
rem    questions of a request each keep their own warm cache (3.9 s -> 0.32 s).
rem  Leave this window open; it is the inference engine.
rem ============================================================================
setlocal
set HERE=%~dp0
set LLAMA=%HERE%llama\llama-server.exe
set MODEL=%HERE%models\StartLux-Decision-0.8B-Q8_0-GGUF\StartLux-Decision-0.8B-Q8_0.gguf

if not exist "%LLAMA%" ( echo llama-server.exe not found at %LLAMA% & pause & exit /b 1 )
if not exist "%MODEL%" ( echo model not found at %MODEL% & pause & exit /b 1 )

echo Starting llama-server on http://127.0.0.1:8081  (Vulkan1 = GTX 1050)
"%LLAMA%" -m "%MODEL%" --device Vulkan1 -ngl 99 -c 32768 -np 4 --jinja --host 127.0.0.1 --port 8081
