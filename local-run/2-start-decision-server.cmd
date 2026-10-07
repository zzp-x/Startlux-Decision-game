@echo off
rem ============================================================================
rem  StartLux-Decision decision service (/v1/systemone) - no torch, no transformers
rem    Reads the option-letter log-probabilities from llama-server and turns them
rem    into the typed answers.  Start 1-start-llama-*.cmd first, in another window.
rem
rem  --concurrency must be <= the llama-server -np value, otherwise the questions
rem  are queued instead of being batched into one forward pass.
rem
rem  --max-length is the per-request prompt budget.  With -c 32768 -np 4 the
rem  server gives each slot 32768/4 = 8192 tokens, so that is the real ceiling;
rem  raise -c (costs KV memory) or lower -np for longer evidence.
rem ============================================================================
setlocal
set HERE=%~dp0
set PY=C:\Users\Libai\.workbuddy\binaries\python\versions\3.13.12\python.exe
set MODEL=%HERE%models\StartLux-Decision-0.8B-Q8_0-GGUF

echo Starting the decision server on http://127.0.0.1:8090  (engine: %MODEL%)
"%PY%" -X utf8 "%HERE%gguf-local\startlux_local.py" --model-dir "%MODEL%" --llama http://127.0.0.1:8081 --port 8090 --concurrency 4 --max-length 8192
pause
