@echo off
rem ============================================================================
rem  CPU-only fallback (no GPU): 0.8B Q8_0 on the processor.
rem    Measured on this machine: ~684 ms per decision vs ~413 ms with the GPU.
rem    Use it when the GPU is busy, or to reproduce the GPU numbers.
rem ============================================================================
setlocal
set HERE=%~dp0
set LLAMA=%HERE%llama\llama-server.exe
set MODEL=%HERE%models\StartLux-Decision-0.8B-Q8_0-GGUF\StartLux-Decision-0.8B-Q8_0.gguf

echo Starting llama-server on http://127.0.0.1:8082  (CPU only, -ngl 0)
"%LLAMA%" -m "%MODEL%" -ngl 0 -c 32768 -np 1 --jinja --host 127.0.0.1 --port 8082
