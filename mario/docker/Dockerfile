# 官方 4esv/jev-mario 技术栈，依赖版本严格钉死为上游 pyproject.toml 所要求的那套。
#
# 这个镜像存在的理由：nes-py==8.2.1 在**任何** Python 版本下都没有 Windows wheel
# （PyPI 上到 8.2.1 为止每个版本只有 sdist 加一个 macOS wheel），而它的 setup.py 把
# CC='g++' 写死了，用普通 setuptools Extension 编译 C++ 核心——没有 CMake，也没有 MSVC 分支。
# 所以官方那套 pin 只能在有 g++ 的地方构建，也就是 Linux userland。在这里它们从源码编译，
# 之后 branch.py 完全按上游写法运行，全程不需要任何兼容垫片。
FROM python:3.12-slim

RUN apt-get update \
 && apt-get install -y --no-install-recommends build-essential \
 && rm -rf /var/lib/apt/lists/*

# 版本号逐条来自 jev-mario-main/pyproject.toml（requires-python >=3.12,<3.13）
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

# 模型服务留在宿主机上：这份 llama.cpp 是 Vulkan 构建，而 Linux 虚拟机没有 Vulkan 直通，
# 所以容器只跑模拟器这一侧，决策通过 host.docker.internal 回宿主机取。
ENTRYPOINT ["python", "-u", "mario_launch.py", "branch.py"]
CMD ["--bot", "jev", "--level", "1-1", \
     "--url", "http://host.docker.internal:8090/v1/systemone", \
     "--label", "startlux-0.8b-docker"]

# 不经启动器、直接跑上游入口的写法（需要挂 runs 卷才能写出 gif；harness 是按 play.py 的
# 位置解析 runs/ 的，所以工作目录必须是 harness 目录）：
#   docker run --rm --workdir /work/jev-mario-main \
#     -v "$PWD/jev-mario-main/runs:/work/jev-mario-main/runs" jev-mario:upstream-pins \
#     python -u branch.py --bot jev --level 1-1 --url http://host.docker.internal:8090/v1/systemone
#
# 想并行就在启动器形式后面加 `--parallel 6`，把一次决策的 11 个选项扇到 6 个进程上；
# 容器能看到宿主机的全部核心，所以同样的加速在里面一样成立。
