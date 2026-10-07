@echo off
rem ============================================================================
rem  StartLux-Decision 2B Q4_K_M on the GTX 1050 (3 GB)
rem    1215 MB weights fit; KV cache stays in system RAM. -c 32768 still fits.
rem    Do NOT drop --no-kv-offload: the Vulkan device is lost while allocating
rem    the GPU KV cache on this driver/card.
rem ============================================================================
setlocal
set HERE=%~dp0
set LLAMA=%HERE%llama\llama-server.exe
set MODEL=%HERE%models\StartLux-Decision-2B-Q4_K_M-GGUF\StartLux-Decision-2B-Q4_K_M.gguf

if not exist "%MODEL%" ( echo model not found at %MODEL% & pause & exit /b 1 )

echo Starting llama-server on http://127.0.0.1:8081  (Vulkan1 = GTX 1050, 2B Q4_K_M)
"%LLAMA%" -m "%MODEL%" --device Vulkan1 -ngl 99 -c 32768 -np 4 --no-kv-offload --jinja --host 127.0.0.1 --port 8081
