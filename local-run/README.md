# StartLux-Decision 本地运行手册(GTX 1050 / 3 GB)

**结论:这台机器能上显卡的规模是 0.8B(Q8_0,推荐)和 2B(Q4_K_M,可用)。** 见下面的实测表。

## 硬件

| 项目 | 值 |
|---|---|
| GPU | NVIDIA GeForce GTX 1050 3072 MiB(Pascal,compute 6.1,驱动 582.66) |
| 集显 | Intel UHD Graphics 630(与独显一起被 Vulkan 枚举为 Vulkan0) |
| CPU | Intel Core i7-9750H,6 核 12 线程 |
| 内存 | 15.9 GB |

推理引擎用的是 **llama.cpp Vulkan 构建**(`llama/`,`b11417`),不依赖 CUDA,因此 Pascal 老卡也能吃满。权重是 ModelScope 上 `StartLuxAI/` 官方发布的 GGUF。

## 怎么跑

**最省事的方式是双击仓库根目录的 `run.cmd`**：它会自动挑显卡、按下面的参数把两个服务都
起好、做一次健康检查和冒烟测试，再跑一遍示例调用。停止用 `stop.cmd`。详见根目录 `README.md`。

想手动控制、或者只想起其中一个，再用下面这四个入口（双击即可，或按顺序手动执行）：

```
1-start-llama-0.8b-gpu.cmd      # 0.8B Q8_0  → GPU,32K 上下文,端口 8081   ← 推荐
1b-start-llama-2b-gpu.cmd       # 2B Q4_K_M  → GPU,32K 上下文,端口 8081
1c-start-llama-cpu.cmd          # 0.8B Q8_0  → 纯 CPU 对照,端口 8082
2-start-decision-server.cmd     # 决策服务  → 端口 8090(/v1/systemone),需先起上面任一个
```

决策服务不依赖 torch / transformers:它把 `Startlux-Decision/startlux_decision/jevfmt.py`(纯标准库)直接加载进来做提示词渲染,再用 llama-server 的 `/apply-template`、`/tokenize`、`/completion` 拿到选项字母的 log-probability,除以类型温度后归一化。这就是 `model.py` 里 `decide()` 的原始算法,只是把前向换成了 llama.cpp。

### 试一下

```bash
curl -s http://127.0.0.1:8090/v1/systemone -H "Content-Type: application/json" -d "{\"state\":{\"pair\":\"BTC/USDT\",\"rsi\":38},\"questions\":{\"q\":{\"type\":\"bool\",\"instructions\":\"Is the move news driven?\",\"criteria\":{\"true\":\"a news event explains it\",\"false\":\"ordinary flow\"}}}}"
```

自带脚本:

```bash
cd gguf-local
python selftest.py --model-dir ../models/StartLux-Decision-0.8B-Q8_0-GGUF   # 四类问题全跑一遍
python bench_local.py --llama http://127.0.0.1:8081 --model-dir ../models/StartLux-Decision-0.8B-Q8_0-GGUF
```

## 怎么调用（POST /v1/systemone）

请求体就两块：`state`（模型要读的证据，任意 JSON）和 `questions`（它要回答的问题）。问题有三种类型，与 `startlux_decision/jevfmt.py` 一致：

| `type` | `criteria` 形状 | 回答 |
|---|---|---|
| `choice` | `{"选项id": "判据文本", ...}` | 选中的选项 id + 各选项概率 |
| `noul`（别名 `bool`） | `{"true": "...", "false": "..."}` | "true" 成立的概率 |
| `score` | `["档位文本", ...]`（有序） | 期望档位序号 + `legend` 档位表 |

现成可跑的客户端（零依赖，标准库）：

```bash
cd local-run
python demo_call.py                            # 内置示例请求
python demo_call.py --file demo-request.json    # 自定义请求（4 个问题：choice+noul+score+choice）
python demo_call.py --file demo-request.json --repeat 3   # 连发 3 次看冷/热延迟
```

上面这份 4 问题请求（727 prompt token）实测：

| 场景 | 客户端往返 | 服务端 | input / evaluated / cached tokens |
|---|---|---|---|
| 新证据（缓存未命中） | **2290 ms** | 2160 ms | 727 / 727 / 0 |
| 同一请求再发一次 | **276 ms** | 273 ms | 727 / **16** / **711** |

`usage` 三个计数字段都是**单次请求**口径：`input_tokens` = 提示词长度；`evaluated_tokens` = 服务端真正算掉的 token（取自 llama.cpp 的 `timings.prompt_n`）；`cached_tokens` = 从 KV 缓存复用掉的部分（`input - cached ≈ evaluated`）。注意 **`tokens_evaluated` 这个字段是提示词总长而不是实算量**，别拿它判断缓存有没有命中。

进程内直调（不起 HTTP 服务）：

```python
import sys; sys.path.insert(0, "local-run/gguf-local")
from startlux_local import LocalDecision
eng = LocalDecision("local-run/models/StartLux-Decision-0.8B-Q8_0-GGUF", "http://127.0.0.1:8081",
                    repo="<仓库路径>/Startlux-Decision", max_length=8192, concurrency=4)
answers, usage = eng.decide({"ticket_id": "TCK-1", "message": "..."},
                            {"urgency": {"type": "choice", "instructions": "How urgent?",
                                         "criteria": {"low": "...", "high": "..."}}})
```

接第三方 harness 时不用写代码，直接指 URL：

```bash
--url http://127.0.0.1:8090/v1/systemone
```

## 实测结果

延迟用仓库自带的 `eval/latency.py`(20 次预热 + 200 次计时),**和官方同一份脚本、同一份请求**(support ticket:choice + noul + score 三字段)。

| 口径 | GTX 1050 | 官方 H200 | 倍数 |
|---|---|---|---|
| 3 字段,同一请求重复 | **322 ms**(p50 319 / p95 372) | 12.2 ms | 26× |
| 1 个 yes/no 字段,同一请求重复 | **110 ms**(p50 110 / p95 128) | 8.3 ms | 13× |
| 3 字段,**输入变化**(缓存全失效) | **1111 ms**(p50 1122,12 组不同票据) | 12.2 ms(实算 891 token) | **~91×** |
| 3 字段,官方同形请求(964 token)、输入变化 | **~1400 ms** | 12.2 ms | **~106×** |

前两行脚本口径一致、数字可比,但**对我有利**:请求逐字重复时我的 KV 缓存全部命中,实际只重算约 12 个 token;而官方的 CUDA Graph 是定长重放,每次都实打实跑完 891 个 token。**后两行才是同工同料**——第 3 行用 12 组随机票据实测,第 4 行是官方同形提示词按每问题 467 ms 累加。

模型与显存(一次请求 = 同一份证据 + 4 个问题,提示词约 630 token):

| 模型 | 放哪 | 显存占用 | 结论 |
|---|---|---|---|
| **0.8B Q8_0** | **GPU 全量(32K 上下文)** | **1286 MiB** | **推荐**,KV 也能放显存 |
| 0.8B Q8_0 | 纯 CPU | 0 | 可用,每问题约 684 ms |
| **2B Q4_K_M** | **GPU(KV 在内存,32K)** | **1334 MiB** | **可用**,每问题约 842 ms |
| 2B Q8_0 | 纯 CPU | 0 | 显存放不下 |
| 2B Q8_0 | GPU | — | 加载失败 |
| 4B Q4_K_M | GPU | 2655 MiB / 只剩 327 MiB | 加载不稳定 |
| 4B Q8_0 / 9B / 27B / 35B-A3B | — | — | 3 GB 装不下,只能 CPU |

## 为什么比官方慢约 100 倍

**这是算术,不是配置问题。** 官方 12.2 ms 处理 891 个 token,等于 **73,000 t/s**;这台机器实测 **675 t/s**。比值 **108×**,和上面第三行的 106× 吻合。

`llama-bench` 实测(0.8B Q8_0,pp891 = 与官方"3 字段"同量级):

| | prefill | 生成 | 说明 |
|---|---|---|---|
| GPU(GTX 1050 / Vulkan) | **675 t/s** | 62.5 t/s | `pp296` = 710 t/s |
| CPU(i7-9750H,-ngl 0) | 571 t/s | 27.4 t/s | GPU 只快 1.18× |
| 官方(H200,bf16) | **73,000 t/s** | — | 12.2 ms / 891 token |

差距的来源,按权重排序:

1. **没有张量核心,而且 fp16/bf16 被禁用。** llama.cpp 启动时直接报告:`GTX 1050 | fp16: 0 | bf16: 0 | matrix cores: none`。Pascal 的 GP107 单元 FP16 只有 FP32 的 1/64,所以计算全程退回 FP32。H200 走 bf16 张量核心,差距在这里,不是内存带宽。
2. **这是纯 prefill 负载。** 模型不生成 token,一次决策 = 一次前向,全部成本是矩阵乘。prefill 在 CPU 上本来就强(6 核 i7 有 571 t/s),所以**显卡只快 1.18 倍**。显卡的价值是**把 CPU 让出来**,不是绝对速度。
3. **官方还有两项我没有的优化。** 一是 CUDA Graph(按 (问题数, 长度) 录制,消掉每层 kernel 发射开销——官方自己给的对照:4B 关图 90.3 ms vs 开图 26.0 ms,差 3.5 倍);二是读字母 logits **不需要走 HTTP**,而我的路径每个问题要 4 次 HTTP 往返(`/apply-template`、`/tokenize`、`/completion`)。

这台机器上**已经没有可压的空间**:840 个 token 用 1332 ms 跑完(630 t/s),是这块卡 675 t/s 上限的 94%。

## 五个坑(实测踩到的)

1. **KV 缓存放显存会让 2B/4B 崩。** 2B 只要把 KV 也放显存,加载到 13 秒就 `vk::Queue::submit: ErrorDeviceLost`——哪怕 Q4_K_M 只占 1215 MiB、显存还剩 2.9 GB。加 `--no-kv-offload` 后正常,连 32K 上下文都跑得动。0.8B 不受影响。所以 `1b-start-llama-2b-gpu.cmd` 里这个参数不能删。
2. **llama.cpp 不会对模板化提示词做"部分前缀复用",于是同一份证据下的多个问题全都要全量重算。** 实测对照:

   | 场景 | 共享前缀 | 结果 |
   |---|---|---|
   | 模板提示词,**逐字重复** | 全部 | ✅ `cache_n=325`,只重算 4 个 token |
   | 模板提示词,两个轮流发 | 321 / 329 | ❌ `cache_n=0`,每次全量 467 ms |
   | 纯文本,两个轮流发 | 300 / 301 | ✅ `cache_n=297` |

   也就是说:共享 321 个 token **没用**,共享 300 个 token **有用**——差异来自提示词里的 `<|im_start|>` 等控制 token。后果是顺序发 N 个问题时,N 次全量 prefill(3 问题 ≈ 1400 ms)。**这是上游行为,不是本仓库代码的问题,值得反馈。**

3. **绕过第 2 条的办法是并发 + 多槽。** 用 `-np 4` 让 3 个问题**同时**进入 3 个槽,每个槽各自驻留自己的缓存;后续请求(即使证据变了、前缀相同)就能命中。`--concurrency` 必须 ≤ `-np`,否则问题会排队而不是合并前向。这一条把官方口径的 3 字段从 3.9 s 压到 322 ms。
4. **读出头别问全词表。** llama.cpp 的 `n_probs` 填 `vocab_size`(248320)代价是 **2250 ms**——纯粹为了算 26 万个 softmax 并序列化约 10 MB JSON;`n_probs=8192` 只要 108 ms,而实测字母的最坏排名是 531(全词表 248320 名中的第 531 名)。本引擎的阶梯是 256 → 8192 → 全词表。
5. **`--jinja` 必须加。** 否则拿不到 `enable_thinking=False` 的对话模板,提示词结尾缺少 `<think>\n\n</think>\n\n`(本引擎会直接报错)。

## 正确性说明

- 分词器:llama-server 对 `A`..`Z` 的 token id 是 32..57,与 `decision_config.json` 的 `letter_token_ids` **完全一致**;字母在模板前缀后仍是独立单 token。
- 模板:渲染结果以 `<think>\n\n</think>\n\n` 结尾,和 `jevfmt.THINK_OFF_CHECK` 要求的一致。
- 读出头:取末位选项字母的 log-prob,除以 `temperature_by_type`,再**在选项上**归一化。因为最终只在这几个选项间归一化,绝对量级无关,所以用 top-k 而不是全词表 softmax 是精确等价的(不是近似)。
- **并发路径的浮点非确定性(已知,已量化)。** 串行路径逐位可复现(两次跑的差 = 0);并发路径因为批大小不同、归约顺序不同,会有一个字段漂移:实测最大值 **6.2e-04**(`urgent.noul`:0.5210414846 → 0.5216601861,相对 0.12%)。**每个问题的最终标签(choice / 判定 / score)100% 一致**,修复前后也一样(修复前官方脚本给的 `urgent = 0.5210414846327736`,修复后完全一致)。要逐位可复现就用 `--concurrency 1` 换延迟。
- 参考实现对 CUDA 有硬门槛(缺 `flash-linear-attention` 会拒绝启动),这台机器没有 CUDA 环境,所以没能和官方 CUDA 路径做同题对数。上面是本地两条独立路径(GPU / CPU)互为对照的结果。

## 目录

```
local-run/
  llama/                     llama.cpp b11417 Vulkan 构建
  models/
    StartLux-Decision-0.8B-Q8_0-GGUF/    774 MiB   ← 已下载、已校验(SHA256 与远端一致)
    StartLux-Decision-2B-Q8_0-GGUF/      1919 MiB  (CPU 用)
    StartLux-Decision-2B-Q4_K_M-GGUF/    1215 MiB  ← GPU 可用
    StartLux-Decision-4B-Q4_K_M-GGUF/    2583 MiB  (边缘,不稳)
  gguf-local/
    startlux_local.py       决策引擎 + /v1/systemone 服务(零依赖)
    selftest.py             四类问题自测
    bench_local.py          延迟基准
  logs/                     一键脚本的日志与进程状态(按档位分文件)
  fetch_model.py            下载别的 GGUF(分段直写,无残留分片,实测 25-40 MB/s)
  demo_call.py              最小调用示例(标准库,零依赖)
  demo-request.json         示例请求(choice + noul + score + choice 四个问题)
  1*.cmd / 2*.cmd           手动分步启动脚本
```

一键启动脚本在**仓库根目录**：`../run.cmd`(启动)、`../stop.cmd`(停止)、
`../run_local.py`(编排逻辑,可带 `--model/--mario/--status/--stop/--follow` 等参数)。

想换模型只需两步:

```bash
python fetch_model.py StartLux-Decision-9B-Q4_K_M-GGUF StartLux-Decision-9B-Q4_K_M.gguf   # 9B 装不进 3 GB,但 CPU 能跑
python gguf-local/selftest.py --model-dir models/StartLux-Decision-9B-Q4_K_M-GGUF
```

