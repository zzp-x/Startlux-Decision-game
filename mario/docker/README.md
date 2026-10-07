# Docker：用官方 pin 精确复现 harness 运行环境

## 这个目录解决什么

`jev-mario-main/pyproject.toml` 把依赖钉死在：

```
requires-python = ">=3.12,<3.13"
gym-super-mario-bros==7.4.0   nes-py==8.2.1   gym==0.26.2   numpy<2   httpx   imageio   pillow
```

这套 pin 在 Windows 上装不上，而且**不是版本没挑对**：

| 事实 | 证据 |
| --- | --- |
| `nes-py` 全版本（0.0.0 到 8.2.1）都没有 Windows wheel | PyPI 上每个 release 都只有 2 个文件：sdist + macOS wheel |
| 有 Windows wheel 的只有 `9.0.0` / `9.0.1`，且仅 cp313 / cp314 | `nes_py-9.0.1-cp313-cp313-win_amd64.whl` |
| `nes-py` 的构建脚本本身就要求 g++ | `setup.py` 里硬编码 `os.environ['CC'] = 'g++'`，用 setuptools `Extension` 直接编译，`-std=c++1y -O3`，**不走 CMake** |

所以想在 Windows 上用官方 pin，等于要给一个 g++-only 的 C++ 扩展在 MSVC 下造出一条不存在的构建路径。
换 Linux 用户态是**唯一**可行路线——这就是这个镜像的全部意义。镜像里 `branch.py` 按上游原文运行，
兼容垫片自动失效（垫片只在栈真的拒绝那个 kwarg 时才起作用）。

## 怎么用

前提：决策服务要监听 `0.0.0.0`。默认只绑 `127.0.0.1`，容器进不来：

```bash
cd local-run
python -X utf8 gguf-local/startlux_local.py \
    --model-dir models/StartLux-Decision-0.8B-Q8_0-GGUF \
    --llama http://127.0.0.1:8081 --host 0.0.0.0 --port 8090
```

然后：

```bash
cd mario
docker compose -f docker/docker-compose.yml build
docker compose -f docker/docker-compose.yml run --rm jev-mario
```

`runs/` 通过卷挂回宿主，录像和决策日志照常落在 `mario/jev-mario-main/runs/`。
镜像里也带了 `mario_launch.py` / `mario_par.py`，所以容器内一样能加 `--parallel 6` 用上宿主的 12 个核。

## 本机现状：引擎起不来（所以这里没有构建过）

这台机器上 **Docker 引擎当前不可用**，镜像因此**未经构建验证**——Dockerfile 的依赖集是从
sdist 的构建脚本推出来的，不是跑出来的。证据：

| 检查 | 结果 |
| --- | --- |
| `docker --version` | `29.4.1`（CLI 已装） |
| `docker info` | `failed to connect to the docker API at npipe:////./pipe/dockerDesktopLinuxEngine` |
| Docker Desktop 是否安装 | `C:\Program Files\Docker\Docker\Docker Desktop.exe` 存在 |
| 启动引擎（16:59:54） | 日志显示 `launching com.docker.backend.exe` / `backend process started`，随后进程消失 |
| `%LOCALAPPDATA%\Docker\log\vm\` | 目录**从未生成** —— Linux VM 一次都没起来 |
| `%APPDATA%\Docker\settings.json` | `"wslEngineEnabled": true`（配置走 WSL2 后端） |
| 本会话调用 `wsl.exe` | 被拦截：`PROGRAM BLOCKED BY SECURITY POLICY ... Command Security → Program Blacklist`，且无法在会话内批准或绕过 |

配置指向 WSL2 后端，而 `wsl.exe` 在本机被安全策略拉黑，Linux VM 起不来 → 引擎拿不到
`dockerDesktopLinuxEngine` 管道。要在这台机器上跑通，需要先在
「安全中心 → 命令安全 → 程序黑名单」里放行 `wsl.exe`，重启 Docker Desktop，再用
`docker compose ... build` 验证；或者把镜像构建放到一台 Linux 机器上。

## 为什么模型不放进容器

本机 llama.cpp 是 **Vulkan** 构建（Pascal 架构不依赖 CUDA 13），而 WSL2 里没有 Vulkan 直通；
把模型服务塞进容器会直接失去 GPU。所以分工是：**容器跑模拟器，模型留在宿主**，容器通过
`host.docker.internal:8090` 访问。顺带的好处是容器只吃 CPU，正好和模拟器这条纯 CPU 的路径对齐。
