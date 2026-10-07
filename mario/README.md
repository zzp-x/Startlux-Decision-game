# 用本地 StartLux-Decision 跑官方马里奥（jev-mario）

官方 README 里的马里奥视频（`media/mario_1-1_27b.mp4`）**不在 Startlux-Decision 仓库里**，
它是外部 harness **`4esv/jev-mario`** 跑出来的。那个 harness 支持 `--url`，可以指向任何
说 System One 协议的服务，所以能直接接我们本地 8090 上的 StartLux-Decision。

结果已跑完：**x = 2370 / 旗帜约 3160，未通关**，26 次决策，49 分 30 秒。

> **出处声明**：`jev-mario-main/` 是 [4esv/jev-mario](https://github.com/4esv/jev-mario) 的原样副本，
> 随本仓库分发只是为了克隆下来就能直接玩。它的 `branch.py` / `live.py` / `play.py` /
> `pyproject.toml` / `README.md` 与上游**逐字节一致**（sha256 核对过），我们一行都没有改 ——
> 兼容垫片、URL 重定向、进度探针、`--parallel` 全部写在同级的 `mario_launch.py` / `mario_par.py` 里。
> `runs/` 里 `*-jev-*` / `*-kev-*` / `*-laya-*` 的 GIF 是上游作者的录像，`*-startlux-*` 才是本模型跑出来的。
> **上游没有声明任何许可证**；若作者有异议，删掉 `jev-mario-main/` 即可，其余代码不受影响。

## 怎么跑

```bash
# 一条命令搞定：起服务 + 跑马里奥（等价于双击 ../run.cmd --mario）
python ../run_local.py --mario                  # 串行，约 50 分钟
python ../run_local.py --mario --parallel 6     # 11 个选项并行，约 18 分钟

# 或者手动分三步：
# 1) 推理引擎 + 决策服务（两个窗口，见 ../local-run/README.md）
../local-run/1-start-llama-0.8b-gpu.cmd      # llama-server, 8081
../local-run/2-start-decision-server.cmd     # /v1/systemone, 8090

# 2) 官方分支 harness（模型决策时模拟器暂停；延迟只影响总时长）
P=C:/Users/Libai/.workbuddy/binaries/python/envs/mario/Scripts/python.exe
"$P" -u mario_launch.py branch.py --bot jev --level 1-1 \
     --url http://127.0.0.1:8090/v1/systemone --label startlux-0.8b

# 实时模式（模拟器不等模型，按实测延迟前进）
"$P" -u mario_launch.py live.py --bot jev --level 1-1
```

产物写在 `jev-mario-main/runs/`：`<level>-<mode>-<bot>-<stamp>.gif`、同名 `.log.jsonl`、
以及一行 `results.jsonl`。

## 环境（Windows 上的关键点）

官方 `pyproject.toml` 钉的是 Python 3.12 + `nes-py==8.2.1` / `gym==0.26.2` / `numpy<2`，
**在 Windows 上装不起来，而且和挑哪个 Python 版本无关**：

- **`nes-py` 的 Windows wheel 只存在于 9.0.0 和 9.0.1**，且只有 cp313 / cp314（`nes_py-9.0.1-cp313-cp313-win_amd64.whl`）。
  从 0.0.0 到 8.2.1 的每一个 release，PyPI 上只有两个文件：sdist + macOS wheel。
- 根因在构建脚本里：`nes-py 8.2.1` 的 `setup.py` **硬编码 `os.environ['CC'] = 'g++'`**，用 setuptools
  `Extension` 直接编译（`-std=c++1y -O3`，**不走 CMake**）。也就是说这套依赖在构建层面就是 g++-only，
  在 Windows 上等于要给一个只认 g++ 的 C++ 扩展造一条 MSVC 路径。
- 想让官方 pin 真正跑起来，唯一路线是 Linux 用户态 —— 见 [`docker/`](docker/README.md)。

改用的组合（`C:\Users\Libai\.workbuddy\binaries\python\envs\mario`，Python 3.13）：

| 包 | 版本 | 说明 |
|---|---|---|
| `nes-py` | **9.0.1** | 有 `cp313-win_amd64` 预编译 wheel，无需编译器 |
| `gym-super-mario-bros` | **9.1.0** | 纯 Python，改基于 gymnasium |
| `gymnasium` | 1.4.0 | 替代 gym |
| `numpy` | 2.5.3 | 3.13 上没有 numpy 1.x wheel |
| `pyglet` 1.5.21 / `httpx` / `imageio` / `pillow` | | harness 依赖 |

```bash
C:/Users/Libai/.workbuddy/binaries/python/versions/3.13.12/python.exe -m venv C:/Users/Libai/.workbuddy/binaries/python/envs/mario
C:/Users/Libai/.workbuddy/binaries/python/envs/mario/Scripts/python.exe -m pip install \
  --index-url https://mirrors.aliyun.com/pypi/simple \
  "nes-py==9.0.1" "gym-super-mario-bros==9.1.0" "gymnasium>=1.0.0" numpy httpx imageio pillow
```

> pip 源注意：清华源在本机对 pip 返回 **403**（curl/urllib 正常），换阿里云即可。

**ROM 是随 `gym-super-mario-bros` 包内置的**（`_roms/super-mario-bros.nes` 等四个），不用自己找。

## `mario_launch.py` 做了什么（官方代码零改动）

`jev-mario-main/` 里每个文件都和上游逐字节相同。启动器只做两件事：

1. **兼容垫片**：`gym-super-mario-bros 9.x` 基于 gymnasium，不再接受 harness 传的
   `apply_api_compatibility=True`。垫片把 `gym_super_mario_bros.make` 包一层、丢掉这个 kwarg。
   丢它是安全的 —— 环境本来就是新 API（`reset()` 返回 `(obs, info)`、`step()` 返回 5 元组），
   且已验证 `_backup/_restore` **逐帧精确**（备份→跑 60 帧→还原，x 完全回到原值）。
2. **指向本地服务**：`branch.py` 本来就有 `--url`；`play.py` / `live.py` 把 `play.JEV_URL`
   写死并要 `TYPESAFE_API_KEY`，启动器在导入后覆盖这两处，而不是改官方文件。
   另外挂了一个只读探针，包住 `httpx.Client.post`，把每次决策实时打出来。

## 测出来的数字

**接口契约**（11 个选项的 choice，和 harness 发的一模一样）：

```
OK  choice = 'run and jump right'   latency = 672 ms   input_tokens = 2246
    30.9%  run and jump right     11.7%  hop right     10.3%  run right
    sum(probabilities) = 1.0000 over 11 options
```

**模拟器成本**（branch harness 的硬开销，与模型无关）：

| 项 | 耗时 |
|---|---|
| 单个动作的后果模拟（1 秒 + 36 条两步延续） | **11.3 s** |
| 一次决策（11 个动作全模拟） | **137.9 s** |
| 整场 1-1（26 次决策） | **49.5 min** |

模拟器实测约 **575 帧/秒**（1.74 ms/帧），瓶颈是 nes-py 里 C++ 的逐帧模拟 + PPU 渲染：
`self.screen` 是零拷贝视图，`_get_observation` 不做额外拷贝，所以 Python 侧没有可削减的东西。
**上游 README 说 15–20 分钟/关，这台 i7-9750H 要 50 分钟 —— 是模拟器比作者机器慢，不是模型。**

**模型侧延迟**：branch 的提示词（现状摘要 + 11 条实测后果 + RULES/GRID_LEGEND）约 2.7 s/次；
`live.py` 的 3 字段请求（choice + noul + score，6463 token）约 2.3 s/次。

## 并行评估：整关 49 分 30 秒 → 17 分 57 秒

上面那张表里 137.9 s 的模拟开销是可以拆掉的。`branch.py` 评估一次决策的方式是：取一次模拟器
快照，然后对 11 个选项**逐个**执行「把这个选项演一秒（HORIZON=60 帧，按住直到落地）+ 从落点再演
6×6=36 条两步延续」。**11 个选项都从同一个快照出发，彼此独立** —— 这就是可以并行的那一层。

难在状态怎么传给别的进程。看起来现成的路是快照：`nes-py 9.0.1` 提供了 `NESEnv.dump_state()`，但它
返回的是 Cython 对象 `nes_py._native.NativeStateSnapshot`，**既不能 pickle（`no default __reduce__
due to non-trivial __cinit__`）、也没有 buffer 协议、没有 `tobytes()`**，出不了进程。

所以 worker 改成**重放**：模拟器是确定性的，而父进程会记下它对自己 env 施加的每一步真实输入
（`Sim.step(action, record)` 在真实步进时带 record、模拟分支时 record=None，这个区别正好把真实路径
标出来了）。worker 从 `reset()` 重放这段按键历史就能落到逐位相同的状态；又因为决策只会往后追加
历史，已经重放过的 worker 只需要补放新增的那几十帧。定位完成后，worker 调用的是**上游自己的
`Sim.outcome()`**，所以选项语义按构造就没变——这个模块没有自己写任何模拟逻辑。

`--parallel N` 打开它（不加就是上游原样）。**推荐 6（物理核数），原因见下面的 GPU 那条**：

```bash
"$P" -u mario_launch.py branch.py --bot jev --level 1-1 --parallel 6 \
     --url http://127.0.0.1:8090/v1/systemone --label startlux-0.8b-par
```

**A/B 实测**（`probe_par.py --at 4`：重放真实日志到第 5 次决策的状态，x=594、512 帧按键历史，
在该状态上用两种路径评估同样 11 个选项）：

| 配置 | 单决策模拟耗时 | 加速 | 跨进程状态一致 | 11 个选项结果一致 |
|---|---|---|---|---|
| 串行（上游原样） | 144.5 s | 1.00x | — | 基准 |
| 并行 6 进程 | 41.5 s | 3.48x | ✅ | ✅ |
| 并行 11 进程 | 34.3 s | **4.22x** | ✅ | ✅ |
| 并行 12 进程 | 35.5 s | 4.08x | ✅ | ✅ |

两点如实说明：

- 「一致」是逐字段比出来的，不是估计：每个 worker 重放后回报自己 RAM 与屏幕的哈希，与父进程相同；
  11 个选项的 `dead / flag / dx / frames / on_ground / alive_paths / best_gain / best_path / enemies_ahead / wall / gap`
  全部相等。串行评估跑完，父进程状态也零扰动（哈希不变）——上游 `outcome()` 自带 rewind。
- 4.22x 低于 11 倍理想值，原因是硬件而不是进程池：11 个 CPU-bound 进程挤在 6 个物理核（12 线程）上，
  每个大约只拿到 0.55 个核。所以「进程数 = 物理核数 × 2 以内」就是这台机器的上限。
  这也是**推断**（从并行度与实测曲线的形状推出来的），没有单独测「单 worker 池」这个对照。

**端到端**（整关 1-1，同一个 0.8B 模型、同一个决策服务，唯一变量是 `--parallel`）：

| | 串行（上游原样） | 并行 6 进程 |
|---|---|---|
| 每次决策 | 114.2 s | **41.4 s**（模拟段 38.5 s + 模型 ~2.7 s） |
| 整关 1-1 | 49 min 30 s | **17 min 57 s**（2.76x） |
| 结果 | best_x=2370、flag=false、frames=2857 | 完全相同 |
| 决策日志 | 26 次 | **26/26 逐行一致**：frame / x / choice / 概率 / 模型看到的 11 条后果文本 / `ride` 字段 |

> **并行会打掉 GPU 服务，这点必须知道。** 第一次用 `--parallel 11` 跑，并行版第一次模型调用时
> llama-server 直接死了：`89.09.180.272 E ggml_vulkan: device lost on Vulkan1`，决策服务随即对 harness
> 返回 500（`HTTPStatusError`）。串行那次 26 次调用都没事，差别就是多出来的 11 个进程。这张 3 GB
> Pascal 卡的 prompt eval 本来就在 2.1 s 上下、贴着驱动的超时边界，CPU 被铺满之后 GPU 提交侧被拖过
> 边界，驱动直接重置设备。**所以降到 `--parallel 6`（等于物理核数，留一半逻辑核），并用与原次完全
> 相同的参数重启 llama-server**（保证前后两次运行的决策可比），之后 26 次调用全程稳定。

## 结果与解读

```
{"level": "1-1", "bot": "branch-startlux-0.8b", "best_x": 2370, "flag": false,
 "frames": 2857, "api_calls": 26, "latency_p50": 2.699, "cost_usd": 0.0}
```

选择的动作分布：`stand` 10 次、`run and jump right` 9 次、`run then jump` 3 次、
`bounce on the spring behind` 2 次、`jump right` 1 次、`jump in place` 1 次。

三件事值得说：

1. **x = 2370 正好是 harness 自带 `--bot search`（纯机械搜索、不调用任何模型）在 1-1 上的成绩。**
   上游 README 的表：`1-1  Jev flag / search 2370`。0.8B 停在了同一个位置 —— 那里是「1 格高墙
   在 1 格外 + 2 格宽坑在 5 格外」的地形，11 个选项里没有一个能活过 3 秒，属于选项集的硬停滞点。
2. **概率几乎是平的**：26 次决策里最高一次只有 0.33，典型 0.13–0.22，而 11 选 1 的均匀分布是 0.09。
   也就是「略好于随机」—— 和上游 README 对 Kev-0.8B 的描述完全一致
   （"ranks them slightly better than chance"）。0.8B 的容量就是这个水平。
3. **`stand` 被选了 10 次却没原地不动**：harness 在所选动作本身没有前进时，会连它的最佳延续
   一起执行（日志里的 `ride` 字段）。所以它靠 harness 兜底还是推进到了 2370。

同口径的上游对照（branch 模式、1-1）：

| 控制方 | 到达距离 |
|---|---|
| Jev（TypeSafe，云端） | 通关（3160） |
| **StartLux-Decision-0.8B（本机 GPU）** | **2370** |
| harness 自带 search 基线（无模型） | 2370 |
| Kev-0.8B | 1672 |
| Laya | 1139 |

`live.py` 实时模式在这台机器上不成立：实测延迟 2.26 s，而 `live.py` 让模拟器按延迟实时前进，
于是每次决策 Mario 要盲跑 136 帧（2.26 s），5 次决策就撞死在 x=684。这个模式需要 ~0.4 s 的延迟
（上游 Jev 是 0.38 s）。要用它就得上更快的机器或更小的提示词。

## 单次决策：抓一份真实请求，用 demo_call 打它

不必开整局也能看"模型在马里奥里到底被问了什么"。`capture_request.py` 直接调上游 `Sim.outcome()`
跑完 11 个选项（和 `branch.py` 同一条代码路径），把模型真正会 POST 的 body 原样落盘：

```bash
P=/c/Users/Libai/.workbuddy/binaries/python/envs/mario/Scripts/python.exe
"$P" -u capture_request.py --at 1                  # 写 mario-demo-request.json
../local-run/demo_call.py --file mario-demo-request.json
```

`--at N` = 先按住 `run right` 走 N 次决策再抓。注意 `run right` 会径直撞上第一只板栗仔
（x≈315 处死），`--at` 给大了脚本会明确报错让你调小。

抓 frame 136（`--at 1`，x=201）那一次，`criteria` 与录像的 `outcomes` **逐条完全一致**，实测回答
也复现了录像：

| | |
|---|---|
| 请求体 | 4599 字节 / 11 个选项 / `input_tokens=1417` |
| 实测回答 | `bounce on the spring behind`（p=0.175）—— 录像 frame 136 的选择与 `p=0.18` |
| 延迟 | 冷 4755 ms；同请求重发全命中缓存 89 ms |

11 选项的完整分布很说明问题：唯一**必死**的选项 `run right`（"dies in 0.6 s"）拿到最低的 0.017，
模型确实读懂了"会死"；但它选中的 `bounce on the spring behind` 也只有 0.175——均匀分布是 0.091，
只是略高，谈不上有把握。

（`live.py` 问的是**另一套问题**：9 个原子动作 `stand`/`walk right`/`jump right`/`run right`/
`run and jump right`/`jump in place`/`walk left`/`hop right`/`back off for a run-up`，附带
`RULES + GRID_LEGEND` 和 13x20 的字符网格。`branch.py` 这套是"先模拟再描述"，
选项文本是仿真出来的结果句，没有网格。）

## 文件

```
mario/
  mario_launch.py           启动器：兼容垫片 + 本地 URL 重定向 + 决策进度探针 + --parallel
  capture_request.py        抓一份真实的 branch 决策请求 -> mario-demo-request.json
  mario-demo-request.json   抓好的样例请求，可直接喂给 ../local-run/demo_call.py
  mario_par.py              --parallel 的实现：进程池 + 按键历史重放（原因见上文）
  jev-mario-main/           官方 harness，逐字节未改动
    branch.py               分支 harness（官方 README 的演示用的是这个）
    play.py live.py         直接控制 / 实时（共用 play.py 的 RAM→网格解析）
    runs/                   产物（含上游作者随仓库自带的三份对照录像：Jev/Kev/Laya）
  docker/                   用官方 pin 复现环境的镜像（本机引擎起不来，未经构建验证）
  make_compare.py           把两份 runs/ 录像拼成左右对比 GIF（用法：python make_compare.py [倍数]）
  probe1.py probe2.py       依赖与模拟器探测
  probe_par.py              A/B 探针：串行 vs 并行的等价性与加速比
  validate.py               接口契约 + 模拟器耗时
  inspect_run.py            决策日志分析
  jev-mario.tar.gz          上游快照
```

已生成的对比片：`runs/1-1-compare-jev-vs-startlux-0.8b.gif`（左：作者 Jev 通关录像；右：本机
StartLux-0.8B x=2370 的录像，逐帧对齐，512x258，约 5.8 MB，30 fps 播放即 2 倍速）。
