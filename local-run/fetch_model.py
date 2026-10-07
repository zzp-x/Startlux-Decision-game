"""Range downloader for the GGUF repositories, tuned for the links this machine gets (~25-40 MB/s).

Writes straight into the destination file, so there are no part files left behind.
Resume: re-running continues from the chunks already recorded in <dest>.progress.json.

  python fetch_model.py <repo> <file.gguf>            # e.g. StartLux-Decision-2B-Q8_0-GGUF StartLux-Decision-2B-Q8_0.gguf
  python fetch_model.py <repo> --config-only          # just config.json + decision_config.json

Every StartLux GGUF repo on ModelScope ships the .gguf plus config.json and decision_config.json;
those two JSONs are all this setup needs next to the weights.
"""
import json
import os
import sys
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

MS = "https://modelscope.cn/models/StartLuxAI/{repo}/resolve/master/{name}"
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
HERE = os.path.dirname(os.path.abspath(__file__))
LOCK = threading.Lock()


def total_size(url):
    req = urllib.request.Request(url, headers={**UA, "Range": "bytes=0-0"})
    with urllib.request.urlopen(req, timeout=60) as r:
        cr = r.headers.get("Content-Range")
        if cr and "/" in cr:
            return int(cr.split("/")[-1])
        cl = r.headers.get("Content-Length")
        return int(cl) if cl else 0


def fetch(url, dest, start, end, i, done, log):
    if i in done:
        return True
    for attempt in range(60):
        try:
            req = urllib.request.Request(url, headers={**UA, "Range": f"bytes={start}-{end}"})
            with urllib.request.urlopen(req, timeout=120) as r:
                buf = r.read()
            if len(buf) != end - start + 1:
                raise IOError(f"short read {len(buf)} != {end - start + 1}")
            with open(dest, "r+b") as f:
                f.seek(start)
                f.write(buf)
            with LOCK:
                done.add(i)
            return True
        except Exception as e:
            log(f"  chunk {i} attempt {attempt+1} {type(e).__name__}: {e}")
            time.sleep(1.5)
    return False


def get_json(url, dest):
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=60) as r:
        data = r.read()
    json.loads(data)
    open(dest, "wb").write(data)
    return len(data)


def main():
    repo, name = sys.argv[1], sys.argv[2]
    workers = int(sys.argv[3]) if len(sys.argv) > 3 else 8
    chunk_mb = int(sys.argv[4]) if len(sys.argv) > 4 else 8
    d = os.path.join(HERE, "models", repo)
    os.makedirs(d, exist_ok=True)

    for f in ("config.json", "decision_config.json"):
        p = os.path.join(d, f)
        if os.path.exists(p) and os.path.getsize(p) > 50:
            continue
        print(f, get_json(MS.format(repo=repo, name=f), p), "bytes")
    if name == "--config-only":
        return

    dest = os.path.join(d, name)
    prog = dest + ".progress.json"
    log = lambda m: print(m, flush=True)          # noqa: E731

    url = MS.format(repo=repo, name=name)
    n = total_size(url)
    print(f"{name}: {n} bytes ({n/1e6:.1f} MB)")
    if not n:
        return
    cs = chunk_mb << 20
    ranges = [(s, min(s + cs - 1, n - 1)) for s in range(0, n, cs)]
    done = set(json.load(open(prog))) if os.path.exists(prog) else set()
    if not os.path.exists(dest) or os.path.getsize(dest) != n:
        with open(dest, "wb") as f:
            f.truncate(n)
        done = set()
    print(f"chunks={len(ranges)} resume_from={len(done)}")
    t0 = time.time()
    with ThreadPoolExecutor(workers) as ex:
        futs = [ex.submit(fetch, url, dest, s, e, i, done, log) for i, (s, e) in enumerate(ranges)]
        for k, f in enumerate(futs):
            f.result()
            if (k + 1) % 40 == 0:
                with LOCK:
                    json.dump(sorted(done), open(prog, "w"))
                print(f"  {k+1}/{len(ranges)} chunks, {time.time()-t0:.0f}s", flush=True)
    ok = all(f.result() for f in futs)
    el = time.time() - t0
    with LOCK:
        json.dump(sorted(done), open(prog, "w"))
    print(f"DONE ok={ok} size={os.path.getsize(dest)} in {el:.0f}s ({n/1e6/max(el,1):.1f} MB/s)")
    if ok:
        os.remove(prog)


if __name__ == "__main__":
    main()
