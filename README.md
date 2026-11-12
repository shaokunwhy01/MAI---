# M2 0.2.1

## 这是什么

M2 0.2.1 是一个**带外置事件记忆的调制式语言模型**。是一个开放架构，底座可以换，记忆层可以接入agent实时调制。

它的组成是两层：

- **底座**：`Qwen2.5-0.5B-Instruct`，**494M 参数**（约 0.5B），24 层，hidden 896，词表 151936。  
  使用完整、未量化未剪枝的 bfloat16 权重；落盘约 **988MB**（942MiB），  
  这是**文件体积**（494M 参数 × 2 字节），不是参数量，两者不要混淆。  
  权重**冻结**，不做任何微调，也不在推理时修改。
- **记忆层（M2）**：一个事件账本（随使用累积，拆包时含 15 条），加上一个把记忆折算成  
  **logit 偏置**的调制器。它不改权重，只在每一步解码时往 logits 上加一个  
  有界的小量。

换句话说：**M2 0.2.1 的能力来自 Qwen 底座，个性与记忆来自事件账本**。  
两者之间没有训练过程，只有一次前向计算里的加法。

底座落在 `qwen2.5-0.5b-instruct` 这一档，本地 CPU 就能跑，  
生成速度约 90 ms/token，上下文 4096 token。

## 和主流大模型有什么不同

主流大模型的"记忆"通常有两条路：把内容拼进上下文窗口（用完即忘），  
或者微调/继续训练（把记忆烧进权重）。M2 0.2.1 走的是第三条：

| 维度     | 主流大模型          | M2 0.2.1                                           |
| ------ | -------------- | -------------------------------------------------- |
| 权重是否变化 | 训练/微调会变        | **冻结**，只读                                          |
| 记忆存在哪  | 上下文窗口，会话结束即丢   | **持久事件账本**，跨会话累积                                   |
| 记忆怎么生效 | 拼进 prompt      | **加到 logits 上的偏置**，不改文本                            |
| 偏置有多大  | ——             | **有硬上界**：`μ·(1+ρ+ν)` ≈ **1.8 logit**，实测通常 0.05–0.3 |
| 记忆怎么来的 | 隐式（预训练/RAG 检索） | **显式事件**：模型说过的话，逐条入账                               |
| 谁来决定好坏 | 模型自己判断         | **人给分**。80 是中性线，高于进好评堆，低于进差评堆                      |
| 可解释性   | 黑箱             | 偏置可观测、可测量；`μ=0` 时必须与纯底座**逐位相同**                    |
| 规模     | 7B ~ 万亿级       | 0.5B                                               |
| 上下文    | 128K 以上        | 4096                                               |
| 工具调用   | 支持             | **不支持**（工具定义被忽略）                                   |
| 训练成本   | 巨大             | **零训练**，只维护账本                                      |

2条差异值得记住：

1. **记忆的影响是被限幅的**。不管账本积累多少，单步偏置永远不会超过 1.8 logit——  
   这意味着记忆只能"推一把"，不能"改写"模型的表达能力。
2. **分数是人的判断，不是模型的自我评价**。模型只负责生成和入库，  
   评分由人来做；未评分的事件停在 80 分，权重为零，不参与好坏任何一侧。

## 这是测试版本

**M2 0.2.1 是内部测试版本（beta0.2.1），用途只有一个：验证路线可行性。**

具体说，它要回答的问题是：*在不训练、不微调、不扩充参数的前提下，  
仅靠一个冻结的小底座加一层有界的外部记忆调制，能否让模型表现出  
可累积、可审计、可回滚的"记忆行为"？*

因此：

- 它**不是产品**，不面向生产环境、不对外提供任何服务承诺。
- 它的**通用能力等同于底座**（Qwen2.5-0.5B-Instruct），不应拿它和主流模型比能力。
- 它的**指标不可横向比较**。当前处于 S1/S2 阶段：每侧事件数不足约 100 条时，  
  趋势通道是关闭的，只有相量通道在工作，此时输出 ≈ 底座 + 弱偏置。  
  这个阶段评价"记忆有没有用"没有意义。
- 接口、账本格式、调制参数（`μ`、`ν`、`ρ`、`α`、`β`）都可能在后续版本中变更，  
  不保证兼容。
- 已知限制：不支持工具调用；上下文超过 4096 token 会被截断；  
  事件规模上千后每次写入需要全量重建，性能尚未优化。

结论的正确读法是：**它证明了这条路能走，而不是它已经走到了。**

---

## 平台说明：这套脚本原本是 macOS 的

本包最初在 **macOS（Intel x86_64）+ Python 3.12** 上打包，以下文件有平台限制：

| 文件 / 目录 | 适用系统 | 说明 |
| --- | --- | --- |
| `m2ctl` | macOS / Linux（bash） | 设置离线环境变量后调用 `python -m m2.cli` |
| `m2serve` | **仅 macOS** | 依赖 `launchd`、`~/Library/LaunchAgents`、`lsof`；`install/uninstall/status` 在 Windows 上完全不可用 |
| `wheels/` | **仅 macOS x86_64 + Python 3.12** | 离线依赖包，全是 `macosx_10_9_x86_64` / `cp312` wheel，Windows **装不上** |
| `.venv/bin/python` | macOS / Linux 目录结构 | Windows 对应 `.venv\Scripts\python.exe` |
| `m2ctl.bat` | **Windows** | 本次补充的 Windows 启动器，等价于 `./m2ctl` |
| `.hf-cache/` | 跨平台（本地文件） | 底座权重快照，**不随仓库分发**，见下一节 |
| `m2_state/` | 跨平台（本地状态） | 事件账本与运行快照，本地状态，不随仓库分发 |

Windows 用户请按文末「Windows 安装与启动」操作；不要尝试执行 `./m2ctl`、`./m2serve`，也不要 `pip install wheels\*`。

## 关于 `.hf-cache/`：权重去哪里下载

**仓库里不含底座权重。** `.hf-cache/` 是每台机器各自放置的**本地离线快照**，已被 `.gitignore` 排除，原因有两个：

- `model.safetensors` 约 **988MB**（942MiB；即 494M 参数 × 2 字节 bfloat16 的完整权重文件，此处指**文件体积而非参数量**），远超 GitHub 单文件 **100MB** 上限，push 会被服务端直接拒绝；Git LFS 的免费额度也几乎被这一个文件占满。
- 权重属于 Qwen 官方发布物（Apache-2.0），标准做法是「代码进仓库、权重单独下载」。

**这不影响你本机的模型完整性**：只要文件还在 `.hf-cache/local/Qwen2.5-0.5B-Instruct/`，`m2ctl` 就会离线加载它，模型完整可跑。只有「别人 clone 这个仓库」时才需要自己补权重。

### 下载源

底座模型：**`Qwen/Qwen2.5-0.5B-Instruct`**（494M 参数，Apache-2.0）

- Hugging Face：https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct
- 魔搭 ModelScope（国内推荐）：https://www.modelscope.cn/models/Qwen/Qwen2.5-0.5B-Instruct

方式 A · Hugging Face CLI（项目锁定 `huggingface_hub==0.22.2`，命令为 `huggingface-cli`；新版 CLI 是 `hf download`）

```bash
huggingface-cli download Qwen/Qwen2.5-0.5B-Instruct \
  --local-dir .hf-cache/local/Qwen2.5-0.5B-Instruct
```

国内若连不上，先设镜像再执行：`export HF_ENDPOINT=https://hf-mirror.com`（Windows PowerShell：`$env:HF_ENDPOINT="https://hf-mirror.com"`）。

方式 B · 魔搭 ModelScope（国内推荐，支持断点续传）

```bash
pip install modelscope
modelscope download --model Qwen/Qwen2.5-0.5B-Instruct \
  --local_dir .hf-cache/local/Qwen2.5-0.5B-Instruct
```

若模型 ID 变更，在魔搭站内搜索 `Qwen2.5-0.5B-Instruct` 的 Qwen 官方模型即可。上面两条命令用的是 bash 续行符 `\`，**Windows PowerShell / CMD 请写成一行**（去掉 `\` 和换行）；路径里的正斜杠在 Windows 上同样有效。

### 目录结构必须长这样

```
.hf-cache/
└── local/
    └── Qwen2.5-0.5B-Instruct/        ← 目录名必须完全一致
        ├── config.json               ← 关键：resolve_model_path() 靠它判断
        ├── generation_config.json
        ├── model.safetensors         ← 约 988MB
        ├── tokenizer.json
        ├── tokenizer_config.json
        ├── vocab.json
        └── merges.txt
```

`m2/readout.py` 的 `resolve_model_path()` 只认 `.hf-cache/local/<模型名>/config.json`：**目录名写错或缺 `config.json`，就会退回联网加载**，而 `m2ctl` 又设了 `HF_HUB_OFFLINE=1`，结果会直接报离线错误。多下载几个文件（如 `README.md`、`added_tokens.json`）不影响使用。

下载完用自检确认：

```bash
./m2ctl health          # macOS / Linux
m2ctl.bat health        # Windows
```

输出里出现 `base_path` 指向本地目录、退出码为 0 即正常；若显示「未找到本地快照，将尝试联网」，说明路径放错了。

## Windows 安装与启动

### 1. 准备 Python

建议 **Python 3.12 64 位**：原包与 `requirements.txt` 基于 3.12，Python 3.13 上 `torch==2.2.2` 等固定版本可能没有对应 wheel。若用 3.13，需要自行放宽 `requirements.txt` 中的版本约束。

### 2. 建虚拟环境、装 Windows 版依赖

先 `cd` 到项目根目录（下例为 `D:\M2-0.2.1`，换成你自己的解压 / clone 路径）：

```bat
cd /d D:\M2-0.2.1
py -3.12 -m venv .venv
.venv\Scripts\python -m pip install -U pip
.venv\Scripts\python -m pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

> **不要安装 `wheels/` 里的包**，那些是 macOS wheel。上面直接从 PyPI 拉 Windows 版依赖；`torch` 的 CPU 包较大，请耐心等待。`.venv/` 已在 `.gitignore` 中，不会误提交。

### 3. 下载底座权重

按上一节把 7 个文件放到 `.hf-cache\local\Qwen2.5-0.5B-Instruct\`。例如 PowerShell：

```powershell
pip install modelscope
modelscope download --model Qwen/Qwen2.5-0.5B-Instruct --local_dir .hf-cache/local/Qwen2.5-0.5B-Instruct
```

### 4. 启动

```bat
m2ctl.bat --help
m2ctl.bat health
m2ctl.bat serve --port 8765
```

`m2ctl.bat` 会设置 `HF_HOME`、`HF_HUB_OFFLINE=1`、`TRANSFORMERS_OFFLINE=1`，并把所有参数转发给 `.venv\Scripts\python.exe -m m2.cli`，与 macOS 的 `./m2ctl` 等价；找不到 `.venv` 时回退到系统 `python`。

后台运行（不占用当前终端）：

```bat
start "M2" /min cmd /c m2ctl.bat serve --port 8765
```

停止服务：先 `netstat -ano | findstr :8765` 找到 LISTENING 行的 PID，再 `taskkill /PID <PID> /F`；前台运行则直接 Ctrl+C。健康检查：`curl http://127.0.0.1:8765/health`。

> Windows 没有 `launchd`，`m2serve` 的 `install / uninstall / status` 不可用。需要开机自启时，用「任务计划程序」新建任务，程序指向 `m2ctl.bat`，参数填 `serve --port 8765`，起始位置填项目根目录。

