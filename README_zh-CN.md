# Qwen-Image-2.1 在 8GB 显存上运行 — 安装、对外提供接口

在 8GB 显卡 **或** 8GB 内存的 Apple Silicon Mac 上运行 Qwen-Image-2.1（7B，Apache-2.0），
并对外暴露一个 **公网 HTTP 接口**：上传图片 + 一句文本指令，返回改好的图片。

一条命令装好全部内容：ComfyUI（以 git 子模块固定版本）、对应硬件的 torch、GGUF 模型栈、
FastAPI 服务、Cloudflare 隧道。不需要手动下载任何文件。

```bash
git clone --recurse-submodules https://github.com/toyhank/qwen-image-2.1-8gb
cd qwen-image-2.1-8gb
./install.sh                 # Windows: .\install.ps1
qwen21 start                 # ComfyUI + API + 公网隧道
```

```
ComfyUI   http://127.0.0.1:8188
API       http://127.0.0.1:8000  （交互式文档在 /docs）
公网      https://<随机串>.trycloudflare.com/v1/edit
```

在任何地方调用：

```bash
curl -X POST https://<隧道>/v1/edit \
  -H "Authorization: Bearer $API_KEY" \
  -F "image=@before.png" \
  -F "prompt=把杯子上的字改成 'ROASTED IN TOKYO'，保持原来的墨迹质感"
```

返回 JSON，里面是 base64 的 PNG 和一个下载地址。完整接口说明见
[docs/API.md](docs/API.md)。

## 一条命令做了什么

`qwen21 install`（`install.sh` / `install.ps1` 负责引导）：

1. 初始化 **ComfyUI 子模块**，固定在 `v0.37.0`——第一个自带 `TextEncodeQwenImage21`
   和 `QwenImage21Cache` 的稳定版本；
2. 建立虚拟环境并安装适合你硬件的 torch；
3. 安装 `city96/ComfyUI-GGUF` **以及它的 `gguf` 依赖**（只装节点不装依赖，
   加载模型时照样报错）；
4. 按配置档下载模型（支持断点续传与体积校验）；
5. 把内置工作流复制进 ComfyUI；
6. 建立 API 虚拟环境并安装 `api/requirements.txt`；
7. 根据 `.env.example` 生成 `.env`。

最后自动执行 `doctor`，告诉你还差什么。

## 硬件配置档

| 配置档 | 机器 | 扩散模型 | 文本编码器 | 默认 |
|---|---|---|---|---|
| `nvidia-8gb` | RTX 4060 8GB、2080 SUPER、3060 | `qwen_image_2.1-Q4_K.gguf`（4.2 GB） | `qwen3vl_8b_w4a8`（6.3 GB，固定在 CPU） | 1024px，25 步 |
| `mac-8gb` | MacBook Air M3 8GB | `qwen_image_2.1-Q3_K.gguf`（3.3 GB） | `qwen3vl_8b_w4a8` | 768px，20 步 |
| `mac-8gb-lite` | 内存吃紧的 8GB | `Q2_K`（2.6 GB） | `qwen3vl_8b_w4a8` | 640px，16 步 |
| `full` | 12GB+ 显存 / 16GB+ 统一内存 | `qwen_image_2.1_int8_convrot.safetensors` | `qwen3vl_8b_int8_convrot` | 1328px，25 步 |

默认 `auto` 会根据 `nvidia-smi` / `sysctl hw.memsize` 自动选择，也可以用
`--profile` 指定。

为什么用 GGUF 而不是官方 int8：官方模板需要 7.3 GB 的 transformer **和** 9.4 GB 的
编码器同时在内存里，8GB 机器都做不到。Q4 GGUF（4.2 GB）加 4bit 编码器（6.3 GB，
跑在 CPU 上）可以装下，ComfyUI 自己的动态显存管理负责在采样前后搬进搬出。
详见 [docs/HARDWARE.md](docs/HARDWARE.md)。

## 命令一览

| 命令 | 作用 |
|---|---|
| `qwen21 install` | 完整安装（`--skip-models`、`--force`、`--recreate-env`、`--ref`） |
| `qwen21 start` | 后台依次启动 ComfyUI、API、隧道 |
| `qwen21 status` | 现在跑着什么、可以从哪里访问 |
| `qwen21 doctor` | 检查硬件、文件、节点、密钥；有问题时返回非 0 |
| `qwen21 stop` / `restart` | 全部停止 / 重启 |
| `qwen21 logs api\|comfy\|tunnel` | 查看日志 |
| `qwen21 tunnel --mode quick\|named\|off` | 单独管理公网入口 |
| `qwen21 models` | 下载或校验模型 |
| `qwen21 smoke` | 端到端测试：上传图片、执行编辑、报告体积与耗时 |
| `qwen21 smoke --direct` | 用 ComfyUI 的 `/prompt` 校验生成的图结构 |
| `qwen21 export` | 把 API 格式的工作流 JSON 写到 `workflows/api/` |
| `qwen21 keygen [--write]` | 生成本地 API key |
| `qwen21 keys` | 哪些密钥还是占位符、从哪里获取 |
| `qwen21 serve` / `qwen21 comfy` | 只在前台运行 API / ComfyUI |

## 公网接口

默认是 Cloudflare **quick tunnel**：不需要账号、不需要 DNS，地址是随机的
`*.trycloudflare.com`，每次重启都会变。

需要固定域名时，把 `TUNNEL_MODE` 设为 `named`，并把 Cloudflare
Zero Trust → Networks → Tunnels → Create 里的 token 粘进 `.env`。
完整步骤与安全检查清单见 [docs/DEPLOY.md](docs/DEPLOY.md)。

两个值得知道的设计细节：

* **只有 API 会被暴露**，ComfyUI 始终监听 `127.0.0.1:8188`。
* **请求不会一直阻塞到隧道超时**。Cloudflare 对超过约 100 秒没有数据返回的源站
  会直接断开，而 25 步的编辑远不止 100 秒。所以请求最多等待 `SYNC_MAX_WAIT`
  （默认 90 秒），没做完就返回 `202` 加一个 job id 供轮询。一个接口，不需要
  客户端写重试逻辑。

## 密钥

只有一个是必需的。

```bash
qwen21 keygen --write     # 把 API_KEY 写进 .env
```

`TUNNEL_TOKEN`（Cloudflare，只有固定域名时需要）和 `HF_TOKEN`（Hugging Face，
只在限流或需要授权仓库时才需要）都是可选的，`.env.example` 里写清了获取页面。
`qwen21 doctor` 会列出所有还是占位符的配置项。

## 目录结构

```
qwen21/          安装器、配置档表、进程管理、隧道管理
qwen_api/        FastAPI 服务（独立虚拟环境，通过 HTTP 与 ComfyUI 通信）
ComfyUI/         git 子模块，固定在 v0.37.0
workflows/       ComfyUI 界面工作流 + 服务用的 API 格式图
api/             API 服务依赖
scripts/         从官方模板重新生成 workflows/
docs/            API.md、DEPLOY.md、HARDWARE.md
runtime/         虚拟环境、日志、pid、下载的工具、任务数据库（不进 git）
tests/           只依赖标准库的单元测试：python -m unittest discover -s tests
```

服务是独立进程，通过 ComfyUI 官方 HTTP 接口通信。正因如此，同一套代码可以跑在
Windows 便携版（自带 Python）、macOS 虚拟环境、以及远程 GPU 机器上，而不必让
任何一端去 import 另一端的依赖。

## 常见问题

| 现象 | 原因 / 解决 |
|---|---|
| `doctor` 说 ComfyUI 版本太老 | 子模块被固定了版本：`git -C ComfyUI fetch --depth 1 origin tag v0.37.0 && git -C ComfyUI checkout v0.37.0` |
| 加载时提示缺 `gguf` | `qwen21 install` 会安装节点依赖；如果你是手动装的节点，执行 `runtime/venvs/comfy/bin/pip install gguf` |
| Cloudflare 报 `524` | 长任务的正常现象：改用 `wait=false` 加轮询，或换 named 隧道 |
| NVIDIA 显存不足 | 关掉占显存的程序，降低 `resolution` / `steps`，编码器保持 `device=cpu` |
| Mac 内存吃紧 / 很慢 | 8GB 机器的正常现象：换 `--profile mac-8gb-lite`，降低 `steps`，或把 `QwenImage21Cache` 的 `device` 设为 `off` |
| `invalid API key` | 执行 `qwen21 keygen --write`，然后 `qwen21 stop && qwen21 start`（服务只在启动时读 `.env`） |

## 致谢与许可

* **Qwen-Image-2.1** — <https://github.com/QwenLM/Qwen-Image>（Apache-2.0）。
  本仓库只负责本地部署，模型版权归 Qwen 团队。
* **ComfyUI** — <https://github.com/Comfy-Org/ComfyUI>（GPL-3.0），以子模块引入。
* **ComfyUI-GGUF** — <https://github.com/city96/ComfyUI-GGUF>（Apache-2.0）。
* GGUF 权重 — <https://huggingface.co/leejet/Qwen-Image-2.1-GGUF>；
  safetensors — <https://huggingface.co/Comfy-Org/Qwen-Image-2.1>。

本仓库的辅助代码采用 MIT 许可（[LICENSE](LICENSE)）。**模型权重不在此分发**，
遵循各自上游许可；把生成结果对外提供之前请先确认这些条款。

[English](README.md)
