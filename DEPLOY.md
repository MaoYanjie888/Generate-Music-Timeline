# Generate-Music-Timeline 从零部署教程

> 本仓库仅包含**代码与配置**。模型权重、整合包、虚拟环境等依赖体积巨大（约 200GB），**未随仓库分发**。
> 从 GitHub 下载仓库后，请按下文顺序完成依赖获取与放置，再启动运行。

## 0. 部署总览

```
准备环境 → 下载仓库 → 获取并放置依赖 → 修复虚拟环境 → 双击 启动工具.bat
（使用 Tab④ 时另开窗口）→ 运行 start_acestep_api.bat
```

预计耗时主要取决于依赖下载速度；依赖放置正确后，启动约 1~3 分钟即可进入主界面。

主界面共四个 Tab：**①歌词时间戳 / ②情绪语音合成 / ③母版音色培育 / ④ACE-Step 音色跟随**。

## 1. 环境要求

- **Windows 10/11**（64 位），仅支持 **NVIDIA 显卡**（CUDA）
- 显存建议 **16GB**（母版音色培育、ACE-Step 音色跟随加载模型较吃显存）
- 磁盘剩余空间 **≥ 200GB**（完整依赖）
- **路径不能含中文**：部分工具链在含中文路径下异常，请放在纯 ASCII 路径，例如 `D:\Generate-Music-Timeline`
- **无需系统级 Python**：仓库自带 `runtime_py310`（Python 3.10 运行时）与虚拟环境结构

## 2. 目录结构（部署完成后的完整形态）

```
Generate-Music-Timeline/
├── 启动工具.bat          # 一键启动：定位依赖 → 检查服务 → 打开主界面
├── setup_win.bat         # 移动项目 / 换电脑后运行一次，修复虚拟环境路径绑定
├── start_acestep_api.bat # 启动 Tab④ 的 ACE-Step 独立 REST API（127.0.0.1:8001）
├── webui.py              # 主界面入口（Gradio，默认 http://127.0.0.1:7860）
├── requirements.txt      # venv_timeline 主环境依赖清单
├── DEPLOY.md             # 本教程
├── deps/
│   └── YuE2-T8-Local-*/  # 启动器历史定位项（四个 Tab 核心链路已不依赖其服务）
├── models/
│   ├── large-v3.pt       # Whisper 大模型（歌词识别，Tab①）
│   ├── Qwen3-ASR-1.7B-hf/          # 歌词识别模型（Tab①）
│   └── Qwen3-ForcedAligner-0.6B-hf/  # 歌词时间戳对齐模型（Tab①）
├── runtime_py310/        # Python 3.10 运行时（python.exe / python310.dll）
├── venv_timeline/        # 主环境（webui.py 运行于此）
├── singing/              # 情绪语音合成 / 母版音色培育 / ACE-Step 音色跟随
├── src/                  # 工具库与 MVGR 歌词时间戳
├── configs/              # 配置与预设（presets/）
└── logs/  outputs/       # 运行时自动生成（日志 / 产物）
```

## 3. 组件依赖获取渠道

下表给出各组件**用途、获取渠道与放置位置**。除标注的官方链接外，不提供未经核实的下载地址；整合包/权重请自行从可信任渠道获取。

| 组件 | 用途 | 获取渠道 | 放置位置 |
| --- | --- | --- | --- |
| **Qwen3-TTS** | Tab② 情绪语音合成、Tab③ 母版音色培育 | 官方：[https://github.com/QwenLM/Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS)。本仓库不打包其运行环境，模型权重与 `qwen_tts` 包位于**外部 ComfyUI 环境**（代码按 `F:\ComfyUI` 布局引用：`models\Qwen3-TTS\Qwen3-TTS-12Hz-1.7B-Base` 与 `...-VoiceDesign`）。未配置时 Tab②③ 不可用 | 外部 ComfyUI 环境 |
| **ACE-Step 1.5** | Tab④ 音色跟随 Cover 翻唱（独立 REST API） | 官方：[https://github.com/ace-step/ACE-Step](https://github.com/ace-step/ACE-Step)。需独立环境及其自带 venv，由 `start_acestep_api.bat` 在 `127.0.0.1:8001` 启动；未配置时 Tab④ 不可用 | 外部独立环境（默认 `F:\ACE-Step-1.5`） |
| **Whisper large-v3** | 歌词识别（Tab①） | 模型权重文件 `large-v3.pt` | `models\large-v3.pt` |
| **Qwen3-ASR-1.7B** | 歌词识别（Tab①） | 权重目录 | `models\Qwen3-ASR-1.7B-hf` |
| **Qwen3-ForcedAligner-0.6B** | 歌词时间戳对齐（Tab①） | 权重目录 | `models\Qwen3-ForcedAligner-0.6B-hf` |
| **YuE2-T8 一键包** | 启动器历史定位项（四个 Tab 核心链路已不依赖其服务） | 从可信任的整合包渠道获取，目录名保持 `YuE2-T8-Local-*` 前缀 | `deps\YuE2-T8-Local-*` |
| **runtime_py310** | Python 3.10 运行时 | 随依赖包分发（也可自行安装 Python 3.10 后按结构放置） | 仓库根 |
| **venv_timeline** | 主虚拟环境（webui.py 运行于此） | 随依赖包分发；缺失时按 §4.2 重建 | 仓库根 |

> 路径含中文会导致部分工具链异常，所有依赖放置路径均建议纯 ASCII。

## 4. 部署步骤

### 4.1 下载仓库

```bash
git clone https://github.com/MaoYanjie888/Generate-Music-Timeline.git
```

将仓库放到**纯 ASCII 路径**（如 `D:\Generate-Music-Timeline`），勿放桌面含中文用户目录等位置。

### 4.2 修复 / 重建虚拟环境

**情况 A：依赖包已自带 venv（推荐）**

双击运行 `setup_win.bat`，它会把 `venv_timeline` 的 `pyvenv.cfg` 重写为绑定仓库自带 `runtime_py310`（移动项目 / 换电脑后必须执行一次）。

**情况 B：需要重建 venv_timeline（主环境）**

```bat
:: 1) 基于自带运行时创建虚拟环境
runtime_py310\python.exe -m venv venv_timeline

:: 2) 先装 PyTorch（CUDA 12.1 构建，必须走 PyTorch 官方 wheel 源）
venv_timeline\Scripts\pip install torch==2.5.1+cu121 torchaudio==2.5.1+cu121 --extra-index-url https://download.pytorch.org/whl/cu121

:: 3) 再装其余依赖
venv_timeline\Scripts\pip install -r requirements.txt
```

### 4.3 启动主界面

双击仓库根 **`启动工具.bat`**，脚本自动执行：

1. 在 `deps\` 下**自动定位** `YuE2-T8-Local-*` 一键包（目录名不写死，随版本可变）
2. 检查 YuE2 服务 `http://127.0.0.1:8189/api/health`，未运行时尝试自动启动（历史定位项）
3. 用 `venv_timeline\Scripts\python.exe webui.py` 启动主界面，浏览器自动打开 `http://127.0.0.1:7860`（端口被占时自动递增）

### 4.4 启动 Tab④ 独立 API（按需）

使用「④ ACE-Step 音色跟随」前，另开一个窗口运行根目录 **`start_acestep_api.bat`**：

- 使用 ACE-Step 自带 venv，不污染 `venv_timeline`
- 只在本机回环 `127.0.0.1:8001` 提供服务，数据不出本机
- 默认环境路径为 `F:\ACE-Step-1.5`，与脚本内 `ACE_ROOT` 保持一致；路径不同请相应修改

## 5. 首次启动等待与验证

- 主界面打开后即见四个 Tab：①歌词时间戳 / ②情绪语音合成 / ③母版音色培育 / ④ACE-Step 音色跟随。
- **Tab「①歌词时间戳」**：上传歌曲音频与歌词文件 → 输出每句起止时间 → 导出 `.lrc / .csv`。
- **Tab「②情绪语音合成」**：选音色、填文字与情绪 → 合成带情绪朗读（需 §3 的 Qwen3-TTS 外部环境）。
- **Tab「③母版音色培育」**：声音设计或参考音色克隆 → 批量合成、页内试听下载（需 Qwen3-TTS 外部环境）。
- **Tab「④ACE-Step 音色跟随」**：先按 §4.4 启动 API，再上传参考音频 → Cover 翻唱；产物在 `outputs\` 下。

## 6. 常见问题（FAQ）

| 现象 | 原因与处理 |
| --- | --- |
| 启动后界面打不开 / 服务报错 | 查看日志：`logs\webui_restart.log`。首次启动需 1~2 分钟加载 |
| 提示未找到 `deps\YuE2-T8-Local-*` | 启动器的历史定位项未放置或目录名前缀不对，检查 §3 放置位置 |
| 提示未找到 `runtime_py310\python.exe` | 运行 `setup_win.bat` 前先确认 `runtime_py310` 已放置 |
| Tab② ③ 不可用 | 未配置 Qwen3-TTS（外部 ComfyUI 环境），见 §3 |
| Tab④ 提示 API 未连接 | 先运行 `start_acestep_api.bat`，确认 `127.0.0.1:8001` 就绪；检查 `ACE_ROOT` 路径 |
| 显存不足 | 关闭其他占用显存的程序；避免同时跑母版音色培育与 ACE-Step |
| 端口被占用 | 主界面端口 7860 被占时自动递增；ACE-Step API 端口 8001 需保持空闲 |
| 路径含中文报错 | 将整个项目移至纯 ASCII 路径后重跑 `setup_win.bat` |
| 非 NVIDIA 环境 | 不支持（依赖 CUDA 运行时） |

### 日志位置汇总

- 主界面 / 启动器：`logs\webui_restart.log`
- 生成产物：`outputs\`

## 7. 已知限制

- **Tab②③ 依赖外部 ComfyUI 环境的 Qwen3-TTS**，本仓库不打包。
- **Tab④ 依赖独立的 ACE-Step 环境**，需单独启动 API，未配置时该 Tab 不可用。
- 全部推理在本机完成，无云端调用；生成内容含 AI 合成成分，请遵守当地法律法规与平台规则。

## 8. 免责声明

本项目为个人本地工具，代码按现状提供，不承诺任何效果与稳定性。第三方组件（Qwen3-TTS、ACE-Step、Whisper、demucs 等）版权归其各自作者所有，分发时请遵守对应开源协议。
