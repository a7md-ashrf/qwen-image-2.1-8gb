# Qwen-Image-2.1 8GB 显存运行方案

**让 Qwen-Image-2.1 在 8GB NVIDIA 显卡上跑起来：ComfyUI + GGUF + 一键安装。**

这个项目专门解决一个很实际的问题：

> **只有 8GB 显存，也想本地跑 Qwen-Image-2.1。**

项目提供：

- 一键安装低显存依赖
- 自动检测显卡与模型文件
- 自动安装 ComfyUI-GGUF
- 自动下载 Q4 GGUF、W4A8 文本编码器和 Qwen-Image-2.1 专用 VAE
- 文生图工作流
- 图片编辑工作流
- 低显存启动模式
- Windows / Linux 安装脚本

> 当前开发环境已在 **RTX 2080 SUPER 8GB** 上验证模型文件与工作流配置。

---

## 为什么做这个项目

Qwen-Image-2.1 是一个非常新的开源图像模型，生成、编辑、文字渲染、多图参考等能力都很强。

问题是：

**原始模型对显存要求并不友好。**

很多 8GB 显卡用户看到模型体积后，会直接认为：

> “这玩意 8GB 肯定跑不了。”

实际上，通过：

- GGUF 量化
- Q4 扩散模型
- W4A8 文本编码器
- ComfyUI CPU offload
- \`--lowvram\`

可以把显存压力显著降低。

需要注意：

> **“支持 8GB 显存”不等于所有模型权重加起来只有 8GB。**

部分权重会在内存和显存之间动态加载，因此建议系统内存至少 **32GB**。

---

# 30 秒安装

## Windows

首先准备：

- NVIDIA 显卡
- ComfyUI
- Git
- Python

然后：

\`\`\`powershell
git clone https://github.com/toyhank/qwen-image-2.1-8gb
cd qwen-image-2.1-8gb

.\install.ps1 -ComfyUI "D:\ComfyUI_windows_portable\ComfyUI"
\`\`\`

例如你的 ComfyUI 在：

\`\`\`text
G:\AI\image\ComfyUI_windows_portable\ComfyUI
\`\`\`

那么执行：

\`\`\`powershell
.\install.ps1 -ComfyUI "G:\AI\image\ComfyUI_windows_portable\ComfyUI"
\`\`\`

---

## Linux

\`\`\`bash
git clone https://github.com/toyhank/qwen-image-2.1-8gb
cd qwen-image-2.1-8gb

./install.sh /path/to/ComfyUI
\`\`\`

---

# 安装器会做什么

安装脚本会自动完成：

1. 安装或更新 \`ComfyUI-GGUF\`
2. 下载 Qwen-Image-2.1 Q4 GGUF
3. 下载 Qwen3-VL W4A8 文本编码器
4. 下载 Qwen-Image-2.1 专用 VAE
5. 将工作流复制到 ComfyUI

不会把模型直接打包进本项目。

所有模型仍然从 Hugging Face 上游仓库下载。

---

# 先检测环境

如果你已经装过一部分模型，建议先运行：

\`\`\`bash
python qwen21.py doctor --comfy /path/to/ComfyUI
\`\`\`

Windows 示例：

\`\`\`powershell
python qwen21.py doctor --comfy "G:\AI\image\ComfyUI_windows_portable\ComfyUI"
\`\`\`

正常情况下会看到类似输出：

\`\`\`text
Qwen-Image-2.1 8GB doctor
=================================
GPU: NVIDIA GeForce RTX 2080 SUPER — 8.0 GiB VRAM

OK: ComfyUI-GGUF
OK: compatible GGUF found: qwen_image_2.1_Q4_K_M.gguf
OK: qwen3vl_8b_w4a8.safetensors
OK: qwen_image_2.1_vae_bf16.safetensors

Ready.
\`\`\`

---

# 模型文件

默认低显存方案：

| 类型 | 文件 |
|---|---|
| 扩散模型 | \`qwen_image_2.1-Q4_K.gguf\` |
| 文本编码器 | \`qwen3vl_8b_w4a8.safetensors\` |
| VAE | \`qwen_image_2.1_vae_bf16.safetensors\` |

其中最重要的一点：

## Qwen-Image-2.1 必须使用 2.1 专用 VAE

也就是：

\`\`\`text
qwen_image_2.1_vae_bf16.safetensors
\`\`\`

不要直接拿旧版 Qwen-Image VAE 替代。

---

# ComfyUI 工作流

项目内已经包含两个工作流：

### 文生图

\`\`\`text
workflows/qwen-image-2.1-8gb-t2i.json
\`\`\`

### 图片编辑

\`\`\`text
workflows/qwen-image-2.1-8gb-edit.json
\`\`\`

可以直接把 JSON 文件拖进 ComfyUI。

也可以自动复制：

\`\`\`bash
python qwen21.py workflows --comfy /path/to/ComfyUI
\`\`\`

---

# 低显存启动

推荐通过项目启动：

\`\`\`bash
python qwen21.py launch --comfy /path/to/ComfyUI
\`\`\`

它会自动带上：

\`\`\`text
--lowvram
\`\`\`

如果你还需要局域网访问：

\`\`\`bash
python qwen21.py launch --comfy /path/to/ComfyUI -- --listen 0.0.0.0
\`\`\`

---

# 8GB 显卡建议

建议：

- 先从 **1024×1024**
- 单张生成
- 不要同时加载多个大模型
- 关闭占显存的软件
- 浏览器别开几十个标签页
- 保持 \`--lowvram\`
- 系统内存建议 32GB 或以上

如果 OOM：

1. 降低分辨率
2. 重启 ComfyUI
3. 关闭其他占显存程序
4. 检查是否加载了其他 checkpoint
5. 确认使用的是 Q4 GGUF

---

# 适合哪些显卡

这个仓库主要面向：

- RTX 3060 8GB
- RTX 4060 8GB
- RTX 4060 Ti 8GB
- RTX 3070 8GB
- RTX 2080 / 2080 SUPER 8GB
- RTX 5060 8GB
- 其他 NVIDIA 8GB 显卡

不同显卡速度会有明显差异。

**8GB 主要解决的是“能不能跑”，不是保证速度一定快。**

---

# 推荐测试提示词

可以用这个 Prompt 测试文字生成能力：

\`\`\`text
Cinematic macro street photography of a vintage wooden coffee cart in Tokyo at dusk.
Resting on the weathered counter is a clear paper cup with crisp, bold black handwritten lettering that reads:

"QWEN 2.1 / 8GB VRAM"

Soft golden bokeh lights, steam rising from an espresso machine,
shallow depth of field, 85mm lens, realistic water droplets.
\`\`\`

如果模型能比较准确地生成：

\`\`\`text
QWEN 2.1 / 8GB VRAM
\`\`\`

就很适合作为项目 README 的实际运行截图。

---

# 项目目标

这个仓库不准备变成另一个巨型 ComfyUI 整合包。

目标很简单：

> **让只有 8GB 显存的人，尽可能少折腾地跑起来 Qwen-Image-2.1。**

以后如果有更好的：

- GGUF
- FP8
- NVFP4
- 更低显存方案
- 更快工作流
- SageAttention / FlashAttention 优化
- 6GB 显存方案

也可以继续加入。

---

# 上游项目

Qwen-Image-2.1：

https://github.com/QwenLM/Qwen-Image-2.1

ComfyUI：

https://github.com/comfyanonymous/ComfyUI

ComfyUI-GGUF：

https://github.com/leejet/ComfyUI-GGUF

Qwen-Image-2.1 GGUF：

https://huggingface.co/leejet/Qwen-Image-2.1-GGUF

ComfyUI 模型文件：

https://huggingface.co/Comfy-Org/Qwen-Image-2.1

---

# License

本仓库中的安装脚本和辅助代码采用 MIT License。

模型权重不会重新打包或分发，模型本身遵循各自上游仓库的 License。

---

如果这个项目帮你省下了折腾 CUDA、显存、ComfyUI 节点和模型路径的时间，可以点一个 ⭐。

这样其他 **8GB 显卡用户**也更容易搜到这个方案。
