"""StartLux-Decision on a llama.cpp GGUF file, with the GPU doing the forward pass.

This is a drop-in replacement for startlux_decision.gguf_server that needs **no torch, no transformers**:
the chat template comes from llama-server (/apply-template), the tokens from /tokenize, and the
next-token log-probabilities of the option letters from /completion.  Prompt rendering, the option-letter
readout, the per-type temperatures and the wide-choice rounds are the package's own jevfmt code, loaded
straight from the Startlux-Decision checkout, so answers match the reference engine question by question.

    python startlux_local.py            # server on 127.0.0.1:8090   (POST /v1/systemone)
    python startlux_local.py --ask "..."  (see tools/ for examples)

Model dir must hold decision_config.json + config.json (they ship next to the .gguf).
"""
import argparse
import importlib.util
import json
import math
import os
import sys
import time
import threading
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
MAX_OPTIONS = 26
DEFAULT_REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                            "..", "..", "Startlux-Decision"))


def load_jevfmt(repo):
    """Load startlux_decision/jevfmt.py by path: the package __init__ pulls torch, jevfmt itself is stdlib-only."""
    path = os.path.join(repo, "startlux_decision", "jevfmt.py")
    spec = importlib.util.spec_from_file_location("jevfmt", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def choice_confidence(p):
    n = len(p)
    return 1.0 if n < 2 else min(1.0, max(0.0, (n * max(p) - 1) / (n - 1)))


def score_confidence(p):
    n = len(p)
    m = max(range(n), key=p.__getitem__)
    spread = sum(v * abs(i - m) for i, v in enumerate(p))
    even = sum(abs(i - (n - 1) / 2) for i in range(n)) / n
    return max(0.0, 1 - spread / even) if even else 1.0


class LocalDecision:
    def __init__(self, model_dir, llama="http://127.0.0.1:8081", repo=None,
                 max_length=32768, cache_prompt=True, verbose=False, concurrency=4):
        self.J = load_jevfmt(repo or DEFAULT_REPO)
        cfg = json.load(open(os.path.join(model_dir, "decision_config.json"), encoding="utf-8"))
        mc = json.load(open(os.path.join(model_dir, "config.json"), encoding="utf-8"))
        self.letters = list(cfg["letter_token_ids"])
        self.temperature = {k: float(v) for k, v in cfg["temperature_by_type"].items()}
        wide = cfg.get("wide_choice", {})
        self.group, self.keep, self.residual = wide.get("group", 25), wide.get("keep", 3), wide.get("residual", 1e-3)
        self.vocab = int(mc.get("text_config", mc).get("vocab_size", 262144))
        self.base = llama.rstrip("/")
        self.max_length = int(max_length)
        self.cache_prompt = bool(cache_prompt)
        self.verbose = verbose
        self.concurrency = max(1, int(concurrency))
        self.calls = 0
        self.tokens = 0
        self._lock = threading.Lock()

    # ---------------------------------------------------------------- llama-server plumbing
    def _post(self, path, obj, timeout=900):
        req = urllib.request.Request(self.base + path, json.dumps(obj).encode(),
                                     {"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())

    def health(self):
        try:
            with urllib.request.urlopen(self.base + "/health", timeout=10) as r:
                return json.loads(r.read())
        except Exception as e:
            return {"error": repr(e)}

    def render_ids(self, row, order):
        """The row's prompt exactly as the checkpoint's chat template renders it -> (ids, text)."""
        msgs, _ = self.J.messages(row, order)
        text = self._post("/apply-template",
                          {"messages": msgs, "chat_template_kwargs": {"enable_thinking": False}})["prompt"]
        if not text.endswith(self.J.THINK_OFF_SUFFIX):
            raise ValueError("chat template lacks the thinking-off assistant prefix (--jinja missing?)")
        ids = self._post("/tokenize", {"content": text, "add_special": False})["tokens"]
        if len(ids) > self.max_length:
            raise ValueError(f"length {len(ids)} > {self.max_length}")
        return ids, text

    def _letter_logprobs(self, ids, count):
        """Next-token log-probabilities of the option letters at the last prompt position.  The server does a full
        softmax; dividing the log-probs by the temperature and normalising over the options equals doing it on the
        logits."""
        targets = self.letters[:count]
        tried = 0
        # Ladder of top-k sizes.  Costs on a GTX 1050 with a warm cache: k=256 ~56 ms, k=8192 ~108 ms,
        # k=n_vocab (248320) ~2250 ms -- the full-vocab path dominates everything else, so it is the last
        # resort only.  Measured worst rank of an option letter on the official support-ticket request is
        # 531 (of 248320), so 8192 covers every case seen with a 15x margin.
        for n in (256, 8192, self.vocab):
            body = {"prompt": ids, "n_predict": 1, "temperature": -1, "n_probs": n,
                    "cache_prompt": self.cache_prompt}
            out = self._post("/completion", body)
            rows = out.get("completion_probabilities") or out.get("probs")
            got = {t["id"]: t["logprob"] for t in rows[0]["top_logprobs"]}
            # NOTE: out["tokens_evaluated"] is the prompt LENGTH, not the work done.  The real work is
            # timings.prompt_n (tokens actually computed); timings.cache_n was reused from the KV cache.
            tim = out.get("timings") or {}
            tried += int(tim.get("prompt_n") or len(ids))
            with self._lock:                      # process-lifetime diagnostics only
                self.calls += 1
                self.tokens += int(tim.get("prompt_n") or len(ids))
            if all(t in got for t in targets):
                if self.verbose:
                    print(f"    [llama] prompt_n={tim.get('prompt_n')} "
                          f"cached={tim.get('cache_n')} prompt_ms={tim.get('prompt_ms', 0):.1f}",
                          file=sys.stderr)
                # per-call stats are returned, not accumulated: usage must describe one request
                return [got[t] for t in targets], {"prompt": len(ids), "evaluated": tried,
                                                   "cached": int(tim.get("cache_n") or 0)}
        raise RuntimeError("option letters missing from llama-server's log-probabilities")

    # ---------------------------------------------------------------- readout
    def _probs(self, rows):
        def one(r):
            order = [o["id"] for o in r["options"]]
            ids, _ = self.render_ids(r, order)
            logp, stat = self._letter_logprobs(ids, len(order))
            return logp, self.temperature.get(r["type"], 1.0), stat

        # Rows are independent, so release them together: llama-server batches concurrent requests into one
        # forward pass (measured 2013 ms sequential vs 1332 ms for three questions on a GTX 1050, i.e. 630 t/s
        # against the card's 675 t/s ceiling).  Needs llama-server started with -np >= len(rows).
        if len(rows) > 1 and self.concurrency > 1:
            with ThreadPoolExecutor(max_workers=min(self.concurrency, len(rows))) as ex:
                res = list(ex.map(one, rows))
        else:
            res = [one(r) for r in rows]

        out, stats = [], {"prompt": 0, "evaluated": 0, "cached": 0}
        for logp, t, stat in res:
            m = max(logp)
            e = [math.exp((x - m) / t) for x in logp]
            s = sum(e)
            out.append([x / s for x in e])
            for key in stats:
                stats[key] += stat[key]
        return out, stats

    # ---------------------------------------------------------------- wide choice lists
    def _wide(self, state, spec):
        keys = list(spec["criteria"])
        n_groups = -(-len(keys) // self.group)
        size, extra = divmod(len(keys), n_groups)
        groups, start = [], 0
        for g in range(n_groups):
            end = start + size + (1 if g < extra else 0)
            groups.append(keys[start:end])
            start = end
        rows = [self.J.from_systemone(state, dict(spec, criteria={k: spec["criteria"][k] for k in g})) for g in groups]
        first, stats = self._probs(rows)
        first_p = {k: p for g, ps in zip(groups, first) for k, p in zip(g, ps)}
        finalists = [k for g, ps in zip(groups, first) for k, _ in sorted(zip(g, ps), key=lambda x: -x[1])[:self.keep]]
        fin_spec = dict(spec, criteria={k: spec["criteria"][k] for k in finalists})
        if len(finalists) > MAX_OPTIONS:
            final_p, more = self._wide(state, fin_spec)
            for key in stats:
                stats[key] += more[key]
        else:
            one_p, more = self._probs([self.J.from_systemone(state, fin_spec)])
            final_p = dict(zip(finalists, one_p[0]))
            for key in stats:
                stats[key] += more[key]
        rest = [k for k in keys if k not in set(finalists)]
        mass = sum(first_p[k] for k in rest) or 1.0
        probs = {k: final_p[k] * (1 - self.residual) for k in finalists}
        probs.update({k: self.residual * first_p[k] / mass for k in rest})
        z = sum(probs.values())
        return {k: v / z for k, v in probs.items()}, stats

    # ---------------------------------------------------------------- public API
    def _answer(self, row, p, question):
        ids = [o["id"] for o in row["options"]]
        if row["type"] == "noul":
            return {"type": "noul", "noul": p[ids.index("true")]}
        if row["type"] == "score":
            levels = question.get("criteria") or []
            levels = [levels[k] for k in self.J.score_keys(levels)] if isinstance(levels, dict) else list(levels)
            return {"type": "score", "score": sum(i * v for i, v in enumerate(p)),
                    "confidence": score_confidence(p),
                    "legend": {str(i): (levels[i] if i < len(levels) else str(i)) for i in range(len(p))},
                    "probabilities": {str(i): v for i, v in enumerate(p)}}
        dist = dict(zip(ids, p))
        return {"type": "choice", "choice": max(dist, key=dist.get), "confidence": choice_confidence(p),
                "probabilities": dist}

    def decide(self, state, questions):
        J = self.J
        answers, rows = {}, []
        stats = {"prompt": 0, "evaluated": 0, "cached": 0}
        for k, q in questions.items():
            t = q.get("type", "choice")
            crit = q.get("criteria") or {}
            more = None
            if t == "choice" and isinstance(crit, dict) and len(crit) == 1:
                only = next(iter(crit))
                answers[k] = {"type": "choice", "choice": only, "confidence": 1.0, "probabilities": {only: 1.0}}
            elif t == "choice" and isinstance(crit, dict) and len(crit) > MAX_OPTIONS:
                p, more = self._wide(state, q)
                answers[k] = {"type": "choice", "choice": max(p, key=p.get),
                              "confidence": choice_confidence(list(p.values())), "probabilities": p}
            else:
                rows.append((k, J.from_systemone(state, q)))
            for key in (more or ()):
                stats[key] += more[key]
        if rows:
            probs, more = self._probs([r for _, r in rows])
            for (k, row), p in zip(rows, probs):
                answers[k] = self._answer(row, p, questions[k])
            for key in stats:
                stats[key] += more[key]
        # per request: input_tokens = prompt length, evaluated_tokens = tokens the server really computed,
        # cached_tokens = the part it reused from the KV cache (input - cached ~= evaluated).
        return answers, {"input_tokens": stats["prompt"], "evaluated_tokens": stats["evaluated"],
                         "cached_tokens": stats["cached"], "output_tokens": 0}


LocalDecision._J_UNUSED = None   # (kept for clarity: jevfmt is loaded per instance as self.J)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                        "models", "StartLux-Decision-0.8B-Q8_0-GGUF"))
    ap.add_argument("--llama", default="http://127.0.0.1:8081")
    ap.add_argument("--repo", default=DEFAULT_REPO)
    ap.add_argument("--port", type=int, default=8090)
    ap.add_argument("--max-length", type=int, default=32768)
    ap.add_argument("--no-cache-prompt", action="store_true")
    ap.add_argument("--concurrency", type=int, default=4,
                    help="questions released in parallel; keep <= llama-server -np so they batch into one forward")
    ap.add_argument("--name", default=None)
    ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args()

    eng = LocalDecision(a.model_dir, a.llama, repo=os.path.abspath(a.repo),
                        max_length=a.max_length, cache_prompt=not a.no_cache_prompt, verbose=a.verbose,
                        concurrency=a.concurrency)
    name = a.name or os.path.basename(os.path.abspath(a.model_dir))
    print(f"[startlux-local] model={name} llama={a.llama} letters={eng.letters[:3]}..{eng.letters[-1]} "
          f"vocab={eng.vocab} temp={eng.temperature}", flush=True)

    class Handler(BaseHTTPRequestHandler):
        def _send(self, code, obj):
            data = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path.rstrip("/") == "/v1/models":
                return self._send(200, {"models": [{"name": name, "description":
                                                    "StartLux-Decision typed decision model (local GGUF / llama.cpp)"}]})
            self._send(200, {"status": "ok", "model": name})

        def do_POST(self):
            try:
                req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
                t0 = time.perf_counter()
                answers, usage = eng.decide(req.get("state"), req["questions"])
                usage["wall_ms"] = round((time.perf_counter() - t0) * 1000, 1)
                self._send(200, {"answers": answers, "usage": usage, "model": name})
            except (ValueError, KeyError) as e:
                self._send(422, {"error": str(e), "detail": [{"loc": ["body"], "msg": str(e), "type": "value_error"}]})
            except Exception as e:
                self._send(500, {"error": repr(e)})

        def log_message(self, *args):
            pass

    class Server(ThreadingHTTPServer):
        request_queue_size = 1024
        daemon_threads = True

    print(f"[startlux-local] listening on http://127.0.0.1:{a.port}/v1/systemone", flush=True)
    Server(("127.0.0.1", a.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
