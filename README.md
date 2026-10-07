# StartLux-Decision 本地运行

在这台机器（GTX 1050 / 3 GB，Windows）上把 StartLux-Decision 跑起来，并对外提供
`POST /v1/systemone` 决策接口。**双击 `run.cmd` 就够了。**

```
双击 run.cmd   ->   llama-server (显卡推理)   127.0.0.1:8081
                   决策服务 /v1/systemone     127.0.0.1:8090
                   （8~10 秒后自动跑一次示例调用）
```

想关掉就双击 `stop.cmd`。

---

## 首次准备

本仓库**只提交外围代码**（一键运行脚本、本地决策引擎、马里奥接入、文档），
**不含 Startlux-Decision 原始仓库与 GGUF 权重，也不含 llama.cpp 的二进制**。跑起来前补三样：

```bash
# 1) 上游仓库 —— 本地引擎会用它的 startlux_decision/jevfmt.py 做提示词渲染
git clone https://github.com/StartLuxLabs/Startlux-Decision.git

# 2) llama.cpp 的 Vulkan 预编译（91 MB，约 40 MB 是 ggml-vulkan.dll，不适合入库）
#    https://github.com/ggml-org/llama.cpp/releases/tag/b11417
#        下载 llama-b11417-bin-win-vulkan-x64.zip
#        把压缩包里的内容解压到 local-run/llama/

# 3) GGUF 权重 —— 0.8B Q8_0，约 774 MiB
python local-run/fetch_model.py StartLux-Decision-0.8B-Q8_0-GGUF StartLux-Decision-0.8B-Q8_0.gguf
```

三样齐了以后目录长这样，然后双击 `run.cmd`：

```
Startlux-Decision/                                  ← 步骤 1
local-run/llama/llama-server.exe                    ← 步骤 2
local-run/models/StartLux-Decision-0.8B-Q8_0-GGUF/  ← 步骤 3
```

许可：StartLux-Decision 代码 Apache-2.0、**权重 CC BY-NC 4.0（非商用）**；
llama.cpp 是 MIT。本仓库自身未附许可。

---

## 关于第三方代码的出处

- **`mario/jev-mario-main/`** 来自 **[4esv/jev-mario](https://github.com/4esv/jev-mario)**，
  原样随仓库分发以便直接玩：`branch.py` / `live.py` / `play.py` / `pyproject.toml` /
  `README.md` 与上游**逐字节一致**（已用 sha256 核对），我们只在外部用启动器包装，
  没有改动它的任何一个文件。`runs/` 里的 `*-jev-*` / `*-kev-*` / `*-laya-*` 演示 GIF 是上游作者
  的录像，`*-startlux-*` 才是我们用本模型跑出来的。
  **上游仓库没有声明任何许可证**，这里属于带出处的转发；如果作者有异议，删除该目录即可，
  其余代码不受影响（改从上游自行克隆）。
- **`local-run/llama/`** 是 llama.cpp 官方 release 的原样解压产物（MIT），本仓库不分发。
- **`Startlux-Decision/`** 上游仓库（Apache-2.0）本仓库不分发，按上面的步骤 1 自行克隆。

---

## 目录

| 文件 | 作用 |
|---|---|
| **`run.cmd`** | **双击启动全部服务**，并跑一次示例调用 |
| **`stop.cmd`** | 双击停止全部服务 |
| `run_local.py` | 真正的编排逻辑（纯标准库，可带参数调用） |
| `run.sh` | 同样的东西，给 git-bash / WSL 用 |
| `local-run/` | 决策服务、调用示例、下载脚本（`llama/` 与 `models/` 按上文自行补齐） |
| `mario/` | 用本模型玩官方马里奥 demo 的 harness |
| `Startlux-Decision/` | 上游仓库（**只被读取，从不修改**，按上文自行克隆） |

服务默认是**游离**的：`run.cmd` 的窗口关掉、或者按回车退出，服务继续在后台跑，
直到你执行 `stop.cmd`。如果你希望"关窗口就等于停服务"，用 `run.cmd --follow`。

---

## 常用命令

```bash
python run_local.py                       # 启动 + 示例调用（等价于双击 run.cmd）
python run_local.py --follow              # 前台监视，Ctrl+C 停止全部
python run_local.py --status              # 看现在谁在跑、哪个档位、健康与否
python run_local.py --stop                # 停止
python run_local.py --restart             # 先停再起（换档位时用）
python run_local.py --model 2b            # 换 2B Q4_K_M 引擎
python run_local.py --model cpu           # 纯 CPU 推理（端口 8082）
python run_local.py --selftest            # 启动后跑四类问题自测
python run_local.py --bench               # 启动后跑延迟基准
python run_local.py --mario               # 启动后跑官方马里奥 1-1
python run_local.py --mario --parallel 6  # 马里奥 + 11 个选项并行评估
```

### 引擎档位

| `--model` | 模型 | 设备 | 显存 | 单次决策 | 说明 |
|---|---|---|---|---|---|
| **`0.8b`**（默认） | 0.8B Q8_0 | GPU | **1286 MiB** | **~390 ms** | 推荐，权重和 KV 缓存都在显存里 |
| `2b` | 2B Q4_K_M | GPU | 1334 MiB | ~842 ms | KV 缓存在内存（脚本已自动加 `--no-kv-offload`） |
| `cpu` | 0.8B Q8_0 | CPU | 0 | ~684 ms | 显卡被占用时用；端口 8082 |

换档位要加 `--restart`，否则脚本会拒绝——它不会偷偷复用一个不是你想要的引擎：

```bash
python run_local.py --model 2b --restart
```

---

## 怎么调用这个模型

### HTTP（推荐）

```bash
curl -s http://127.0.0.1:8090/v1/systemone \
  -H "Content-Type: application/json" \
  -d '{"state":{"pair":"BTC/USDT","rsi":38},
       "questions":{"q":{"type":"bool","instructions":"Is the move news driven?",
                         "criteria":{"true":"a news event explains it","false":"ordinary flow"}}}}'
```

请求体只有两块：`state` 是模型要读的证据（任意 JSON），`questions` 是它要回答的问题。
问题有三种类型，与上游 `startlux_decision/jevfmt.py` 一致：

| `type` | `criteria` 形状 | 回答 |
|---|---|---|
| `choice` | `{"选项id": "判据文本", ...}` | 选中的选项 id + 各选项概率 |
| `noul`（别名 `bool`） | `{"true": "...", "false": "..."}` | "true" 成立的概率 |
| `score` | `["档位文本", ...]`（有序） | 期望档位序号 + `legend` 档位表 |

### 自带客户端

```bash
cd local-run
python demo_call.py                              # 内置示例
python demo_call.py --file demo-request.json      # 4 个问题：choice + noul + score + choice
python demo_call.py --file demo-request.json --repeat 3   # 看冷/热延迟
```

上面这份 4 问题请求（727 prompt token）实测：

| 场景 | 客户端往返 | input / evaluated / cached tokens |
|---|---|---|
| 新证据（缓存未命中） | **~1.3 s** | 727 / 727 / 0 |
| 同一请求再发一次 | **~0.28 s** | 727 / **16** / **711** |

`usage` 三个计数都是**单次请求**口径：`input_tokens` 是提示词长度，`evaluated_tokens`
是服务端真正算掉的 token，`cached_tokens` 是从 KV 缓存复用掉的。注意 **`tokens_evaluated`
这个 llama.cpp 原始字段是提示词总长、不是实算量**，别拿它判断缓存有没有命中。

### 进程内直调（不起 HTTP）

```python
import sys; sys.path.insert(0, "local-run/gguf-local")
from startlux_local import LocalDecision
eng = LocalDecision("local-run/models/StartLux-Decision-0.8B-Q8_0-GGUF",
                    "http://127.0.0.1:8081",
                    repo="Startlux-Decision", max_length=8192, concurrency=4)
answers, usage = eng.decide({"ticket_id": "TCK-1", "message": "..."},
                            {"urgency": {"type": "choice", "instructions": "How urgent?",
                                         "criteria": {"low": "...", "high": "..."}}})
```

### 接第三方 harness

任何说 System One 协议的程序都可以直接指过来，不用改代码：

```
--url http://127.0.0.1:8090/v1/systemone
```

---

## 马里奥 demo

官方 README 里的那段马里奥视频不在 Startlux-Decision 仓库里，它是外部 harness
**`4esv/jev-mario`** 跑出来的。我们已经把它接上本地模型：

```bash
python run_local.py --mario                  # 串行，约 50 分钟
python run_local.py --mario --parallel 6     # 11 个选项并行，约 18 分钟
```

产物在 `mario/jev-mario-main/runs/`：`<关卡>-branch-<标签>-<时间戳>.gif`、同名
`.log.jsonl`、以及一行 `results.jsonl`。

本机实测结果：**x = 2370（旗杆约 3160），未通关，26 次决策**。0.8B 在 "按哪种走法更接近
目标" 这个排序上接近随机（最高概率只有 0.33），和上游 README 对 0.8B 的评价一致——这是
模型规模的问题，不是配置问题。

`--parallel` 的实测收益与安全上限：

| 并行进程 | 26 次决策耗时 | 加速比 |
|---|---|---|
| 1（原样） | 144.5 s | 1.00× |
| 6 | 41.5 s | 3.48× |
| 11 | 34.3 s | 4.22× |

整关总时长从 **49 分 30 秒降到 17 分 57 秒**，且 **26/26 次决策与串行版本逐行完全一致**
（帧号、x 坐标、选项、概率、各选项结局、是否骑乘）。但 11 进程时 llama-server 会出现
`device lost`，**6 是安全上限**。

---

## 故障排查

| 现象 | 原因与处理 |
|---|---|
| `端口 8081 被 PID xxxx 占用但不健康` | 上次没退干净。`python run_local.py --stop`，或 `--restart` |
| `端口 8081 上运行的是 xxx 档，不是你要的 yyy` | 换档位必须加 `--restart`（顺带也用来换模型） |
| 启动很慢、卡在"加载模型" | 首次要从磁盘读 774 MiB 权重，8~10 秒属正常；超过 `--timeout`（默认 300 s）才算异常 |
| `device lost on Vulkan1` | 显存被抢或并发过高。2B 必须带 `--no-kv-offload`（脚本已内置）；马里奥 `--parallel` 不要超过 6 |
| 找不到 Python | 设环境变量 `STARTLUX_PY` 指向你的 `python.exe` |
| 马里奥报找不到 Python | jev-mario 用的是另一个隔离环境，设 `STARTLUX_MARIO_PY`；重建方法见 `mario/README.md` |
| 想看详细日志 | `local-run/logs/` 下按档位分文件：`llama-server.0.8b.log`、`decision-server.0.8b.log` |

---

## 一些设计取舍

- **只用自己的显卡能跑的东西。** 走 llama.cpp 的 Vulkan 后端而不是 CUDA，所以 Pascal 老卡
  （GTX 1050，compute 6.1）也能用。代价是没有张量核心、fp16/bf16 被禁，全程 FP32。
- **每个请求 4 个槽（`-np 4`），决策服务并发度与之对齐。** 不是随便填的：`-np 1` 时单个槽
  会被每个新问题覆写、前缀缓存永远不存活；4 个槽才能让同一次请求里的多个问题各自驻留
  自己的缓存，把官方口径的 3 字段延迟从 3.9 s 压到 322 ms。`--concurrency` 必须 ≤ `-np`，
  否则问题会排队而不是合并成一次前向。
- **读选项字母只取 top-k，不取全词表。** 全词表（248320）要 2250 ms，纯粹花在算 26 万个
  softmax 并序列化 10 MB JSON；`n_probs=8192` 只要 108 ms。因为最终只在这几个选项之间
  归一化，用 top-k 是**精确等价**而不是近似（实测字母最坏排名是第 531 名）。
- **上游 `Startlux-Decision/` 目录零改动。** 本地引擎只是把它的 `jevfmt.py`（纯标准库）
  加载进来做提示词渲染，前向换成 llama.cpp。马里奥那边用兼容垫片接住 gymnasium 的 API
  差异，harness 文件保持与上游逐字节一致。

比官方 H200 慢约 100 倍**是算术不是配置**：官方 12.2 ms 处理 891 token = 73,000 t/s，
这台机器实测 675 t/s，比值 108×。细节推导见 `local-run/README.md`。

## 代码约定

改这个仓库之前先看这三条，不然容易改出问题：

1. **注释、文档字符串一律中文**；**运行时字符串一律英文**——打印输出、日志格式、API 字段名、
   发给模型的提示词与选项文本、CLI 的 `--help`。这些是程序的对外表现，改它会变动行为，
   也会让既有文档里引用的输出失效。
2. **`.cmd` 保持纯 ASCII + CRLF**，一个中文字符都不要加。cmd.exe 按 OEM 代码页
   （中文系统上是 GBK）解析批处理文件，UTF-8 中文会被解成乱码；多字节序列在
   `if ... ( ... )` 块里还有已知的解析失败。所有面向用户的中文一律由 `run_local.py` 打印，
   想看中文说明就 `run.cmd --help`。仓库里的 `.gitattributes` 用 `*.cmd -text` 钉死了行尾，
   别删它——`core.autocrlf` 是每台机器各自的设置，靠它保不住 CRLF。
3. **`Startlux-Decision/` 和 `mario/jev-mario-main/` 只读**。这两块是上游代码，必须与上游
   逐字节一致（已核验）。要改行为就在外围包一层：马里奥那边是 `mario_launch.py` 的兼容垫片
   与 `mario_par.py` 的外部打补丁，决策引擎那边是 `local-run/gguf-local/startlux_local.py`。
   这两个目录也刻意不入发布仓库。

更深入的实测数据、五个踩坑的完整记录、以及 GPU/CPU 两路互为对照的正确性说明：
**`local-run/README.md`**。马里奥 harness 的接入细节与 Windows 依赖为什么装不上：
**`mario/README.md`**。
