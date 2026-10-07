# The official 4esv/jev-mario stack, pinned exactly as upstream's pyproject.toml asks for it.
#
# Why this image exists: nes-py==8.2.1 has no Windows wheel under *any* Python version (PyPI carries only
# an sdist plus a macOS wheel for every release up to 8.2.1), and its setup.py hardcodes CC='g++' and
# compiles the C++ core with a plain setuptools Extension -- no CMake, no MSVC path.  So the official pins
# can only be built where g++ exists, which is a Linux userland.  In here they build from source and
# branch.py then runs exactly as upstream wrote it, with no compatibility shim anywhere.
FROM python:3.12-slim

RUN apt-get update \
 && apt-get install -y --no-install-recommends build-essential \
 && rm -rf /var/lib/apt/lists/*

# exact pins from jev-mario-main/pyproject.toml (requires-python >=3.12,<3.13)
RUN pip install --no-cache-dir --upgrade pip setuptools wheel \
 && pip install --no-cache-dir \
      gym-super-mario-bros==7.4.0 \
      nes-py==8.2.1 \
      gym==0.26.2 \
      "numpy<2" \
      "httpx>=0.27" \
      "imageio>=2.34" \
      "pillow>=10"

WORKDIR /work
COPY jev-mario-main/ /work/jev-mario-main/
COPY mario_launch.py mario_par.py /work/

# The model server stays on the host: this build of llama.cpp is a Vulkan one and the Linux VM has no
# Vulkan passthrough, so the container runs the emulator side only and reaches the host for decisions.
ENTRYPOINT ["python", "-u", "mario_launch.py", "branch.py"]
CMD ["--bot", "jev", "--level", "1-1", \
     "--url", "http://host.docker.internal:8090/v1/systemone", \
     "--label", "startlux-0.8b-docker"]

# Raw upstream entry point, no launcher in the way (needs the runs volume to write the gif; the harness
# resolves runs/ relative to play.py, so the workdir has to be the harness directory):
#   docker run --rm --workdir /work/jev-mario-main \
#     -v "$PWD/jev-mario-main/runs:/work/jev-mario-main/runs" jev-mario:upstream-pins \
#     python -u branch.py --bot jev --level 1-1 --url http://host.docker.internal:8090/v1/systemone
#
# Add `--parallel 6` to the launcher form to fan a decision's 11 options out over 6 processes; the
# container sees all the host cores, so the same speedup applies inside it.
