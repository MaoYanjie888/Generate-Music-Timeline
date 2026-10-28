# Generate-Music-Timeline 从零部署教程

> 本仓库仅包含**代码与配置**。模型权重、整合包、虚拟环境等依赖体积巨大（约 200GB），**未随仓库分发**。
> 从 GitHub 下载仓库后，请按下文顺序完成依赖获取与放置，再启动运行。

## 0. 部署总览

```
准备环境 → 下载仓库 → 获取并放置依赖 → 修复虚拟环境 → 安装一键包运行环境 → 双击 启动工具.bat
```

预计耗时主要取决于依赖下载速度；依赖放置正确后，启动约 1~3 分钟即可进入主界面。

## 1. 环境要求

- **Windows 10/11**（64 位），仅支持 **NVIDIA 显卡**（CUDA）
- 显存建议 **16GB**（母版长曲生成 + 换声同时加载较吃显存）
- 磁盘剩余空间 **≥ 200GB**（完整依赖）
- **路径不能含中文**：部分工具链在含中文路径下异常，请放在纯 ASCII 路径，例如 `D:\Generate-Music-Timeline`
- **无需系统级 Python**：仓库自带 `runtime_py310`（Python 3.10 运行时）与虚拟环境结构

## 2. 目录结构（部署完成后的完整形态）

```
Generate-Music-Timeline/
├── 启动工具.bat          # 一键启动：定位 YuE2 一键包 → 启动服务 → 打开主界面
├── setup_win.bat         # 移动项目 / 换电脑后运行一次，修复虚拟环境路径绑定
├── webui.py              # 主界面入口（Gradio，默认 http://127.0.0.1:7860）
├── requirements.txt      # venv_timeline 主环境依赖清单
├── DEPLOY.md             # 本教程
├── deps/
│   ├── YuE2-T8-Local-*/  # YuE2 歌曲生成一键包（含自带 Python 运行时、模型与 htdemucs）
│   ├── SongBloom/        # SongBloom 代码仓库 + 权重（cache/model）
│   ├── SoulX-Singer/     # SoulX-Singer（含自带 venv）
│   └── ACE-Step-1.5/     # （可选组件，未配置时相关功能自动不可用）
├── models/
│   ├── large-v3.pt       # Whisper 大模型（歌词识别）
│   ├── Qwen3-ASR-1.7B-hf/          # 歌词识别模型
│   ├── Qwen3-ForcedAligner-0.6B-hf/  # 歌词时间戳对齐模型
│   └── VoxCPM2/          # VoxCPM 权重目录
├── runtime_py310/        # Python 3.10 运行时（python.exe / python310.dll）
├── venv_timeline/        # 主环境（webui.py 运行于此）
├── venv_voxcpm/          # VoxCPM 合成独立环境
├── venv_songbloom/       # SongBloom 母版生成独立环境
├── singing/              # 唱歌 / 时间戳流水线核心代码
├── src/                  # 工具库与 MVGR 时间戳
├── configs/              # 配置与预设（presets/）
└── logs/  outputs/       # 运行时自动生成（日志 / 产物）
```

## 3. 组件依赖获取渠道

下表给出各组件**用途、获取渠道与放置位置**。除标注的官方链接外，不提供未经核实的下载地址；整合包/权重请自行从可信任渠道获取。

| 组件 | 用途 | 获取渠道 | 放置位置 |
| --- | --- | --- | --- |
| **YuE2-T8 一键包** | 母版歌曲生成（作曲/编曲）、Seed-VC 换声后端、htdemucs 人声/伴奏分离 | YuE2 开源项目：[https://github.com/multimodal-art-projection/YuE](https://github.com/multimodal-art-projection/YuE)。开发环境使用的 `YuE2-T8-Local` **一键整合包**（含自带 Python 运行时与模型）为本地整合形态，**非官方仓库直接分发**，请从可信任的整合包渠道获取，目录名保持 `YuE2-T8-Local-*` 前缀 | `deps\YuE2-T8-Local-*` |
| **Seed-VC** | 参考音色换声（母版人声 → 参考音色） | 由 YuE2-T8 一键包内置 `voice_convert` 后端提供，无需单独放置；官方项目：[https://github.com/Plachtaa/seed-vc](https://github.com/Plachtaa/seed-vc) | 随一键包 |
| **VoxCPM2** | 文本合成备用通道 / 母版音色培育（voxcpm 引擎） | 官方：[https://github.com/OpenBMB/VoxCPM](https://github.com/OpenBMB/VoxCPM)；权重目录 `VoxCPM2` 需自行获取 | `models\VoxCPM2` |
| **SongBloom** | 参考音色生成母版（备用引擎，最长 150 秒） | 无统一官方发布页，**不提供链接**。需自行获取代码仓库与权重，按 `deps\SongBloom` 布局放置（代码含 `SongBloom/models` 包；权重在 `cache/model`，共 5 个文件：`autoencoder_music_dsp1920.ckpt`、`songbloom_full_150s.pt`、`songbloom_full_150s.yaml`、`stable_audio_1920_vae.json`、`vocab_g2p.yaml`） | `deps\SongBloom` |
| **Qwen3-TTS** | 情绪语音合成、母版音色培育（qwen3 引擎） | 官方：[https://github.com/QwenLM/Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS)。**注意**：本仓库不打包 Qwen3-TTS 运行环境，模型权重与 `qwen_tts` 包位于**外部 ComfyUI 环境**（代码按 `F:\ComfyUI` 布局引用：`models\Qwen3-TTS\Qwen3-TTS-12Hz-1.7B-Base` 与 `...-VoiceDesign`）。未配置时相关 Tab 不可用，不影响主通道唱歌 | 外部 ComfyUI 环境 |
| **htdemucs** | 母版人声/伴奏分离 | 随 YuE2 一键包内置（`runtime` + `models\Demucs\` 下的 `htdemucs.yaml`、`955717e8.safetensors` 等） | 随一键包 |
| **Whisper large-v3** | 歌词识别（MVGR） | 模型权重文件 `large-v3.pt` | `models\large-v3.pt` |
| **Qwen3-ASR-1.7B** | 歌词识别 | 权重目录 | `models\Qwen3-ASR-1.7B-hf` |
| **Qwen3-ForcedAligner-0.6B** | 歌词时间戳对齐 | 权重目录 | `models\Qwen3-ForcedAligner-0.6B-hf` |
| **runtime_py310** | Python 3.10 运行时 | 随依赖包分发（也可自行安装 Python 3.10 后按结构放置） | 仓库根 |
| **venv_timeline / venv_voxcpm / venv_songbloom** | 三个虚拟环境 | 随依赖包分发；缺失时可按 §4.3 重建 | 仓库根 |

> 路径含中文会导致部分工具链异常，所有依赖放置路径均建议纯 ASCII。

## 4. 部署步骤

### 4.1 下载仓库

```bash
git clone https://github.com/MaoYanjie888/Generate-Music-Timeline.git
```

将仓库放到**纯 ASCII 路径**（如 `D:\Generate-Music-Timeline`），勿放桌面含中文用户目录等位置。

### 4.2 获取并放置依赖

按 §3 表格将各组件放到对应目录。完成后核对：

- `deps\` 下存在 `YuE2-T8-Local-*`、`SongBloom` 等
- `models\` 下存在 `large-v3.pt`、`VoxCPM2`、`Qwen3-ASR-1.7B-hf`、`Qwen3-ForcedAligner-0.6B-hf`
- 仓库根存在 `runtime_py310\python.exe` 与 `venv_timeline\Scripts\python.exe`（缺失时见 §4.3）

### 4.3 修复 / 重建虚拟环境

**情况 A：依赖包已自带三个 venv（推荐）**

双击运行 `setup_win.bat`，它会把 `venv_timeline` 与 `deps\SoulX-Singer\venv` 的 `pyvenv.cfg` 重写为绑定仓库自带 `runtime_py310`（移动项目 / 换电脑后必须执行一次）。

**情况 B：需要重建 venv_timeline（主环境）**

```bat
:: 1) 基于自带运行时创建虚拟环境
runtime_py310\python.exe -m venv venv_timeline

:: 2) 先装 PyTorch（CUDA 12.1 构建，必须走 PyTorch 官方 wheel 源）
venv_timeline\Scripts\pip install torch==2.5.1+cu121 torchaudio==2.5.1+cu121 --extra-index-url https://download.pytorch.org/whl/cu121

:: 3) 再装其余依赖
venv_timeline\Scripts\pip install -r requirements.txt
```

**venv_voxcpm / venv_songbloom** 缺失时：创建虚拟环境后，按 VoxCPM 官方仓库说明安装 `voxcpm` 包及依赖（[https://github.com/OpenBMB/VoxCPM](https://github.com/OpenBMB/VoxCPM)），按 SongBloom 代码仓库附带的依赖清单安装 SongBloom 依赖。

### 4.4 安装 YuE2 一键包运行环境

进入 `deps\YuE2-T8-Local-*\`，若整合包未预装运行环境，双击 **`安装运行环境.bat`** 完成安装（一键包自带 `runtime\python.exe` 后即完成）。之后**不要**手动改动一键包目录名。

### 4.5 启动

双击仓库根 **`启动工具.bat`**，脚本自动执行：

1. 在 `deps\` 下**自动定位** `YuE2-T8-Local-*` 一键包（目录名不写死，随版本可变）
2. 检查 YuE2 服务 `http://127.0.0.1:8189/api/health`
3. 未运行时自动调用一键包 `启动本地整合包.ps1` 启动 YuE2 服务（首次加载模型约 1~3 分钟）
4. 用 `venv_timeline\Scripts\python.exe webui.py` 启动主界面，浏览器自动打开 `http://127.0.0.1:7860`（端口被占时自动递增）

主界面共 4 个 Tab：**唱歌创作 / 情绪语音合成 / 歌词时间戳 / 母版音色培育**。

## 5. 首次启动等待与验证

- 启动器输出 `[OK] YuE2 服务已运行 (127.0.0.1:8189)` 表示母版生成链路就绪；`[OK] 定位到 YuE2 一键包` 表示依赖定位正确。
- 主界面打开后，进入 **Tab「唱歌创作」**：填写 style / lyrics，选择合成通道（`voice_convert`=母版换参考音色，`voxcpm`=文本合成备用），点击生成。
- 生成产物在 `outputs\` 下（母版 + 分离 vocal/acc + 最终 `final_song.wav`）。
- **Tab「歌词时间戳」**：上传歌曲音频与歌词文件 → 输出每句起止时间 → 导出 `.lrc / .csv` 或一键复制。
- **Tab「母版音色培育」**：从描述捏音色或参考音色克隆（VoxCPM 引擎可用；Qwen3 引擎需 §3 的 ComfyUI 环境）。

## 6. 常见问题（FAQ）

| 现象 | 原因与处理 |
| --- | --- |
| 启动后界面打不开 / 服务报错 | 查看日志：`logs\webui_restart.log`、`logs\yue2_service.log`，以及一键包内 `deps\YuE2-T8-Local-*\logs\server.stderr.log`。YuE2 服务首次启动需 1~2 分钟加载 |
| 提示未找到 `deps\YuE2-T8-Local-*` | 一键包未放置或目录名前缀不对，检查 §3 放置位置 |
| 提示未找到 `runtime_py310\python.exe` | 运行 `setup_win.bat` 前先确认 `runtime_py310` 已放置 |
| 歌曲生成显示错误但产物已存在 | 多为轮询超时误报：母版长曲可达 50~80 分钟，优先查看 `outputs\` 下是否有 `final_song.wav` |
| 生成中途崩溃（0xC0000005 / 卡死） | 多为 GPU 上下文未释放或 pyworld 遇 NaN；pipeline 已加网络兜底、轮询看门狗与音准校正子进程化，仍失败时关闭其他占显存程序重试 |
| 显存不足 | 关闭其他占用显存的程序；缩短歌词 / 降低音频分辨率重试 |
| 端口被占用 | 主界面端口 7860 被占时自动递增；YuE2 服务端口 8189 被占时启动器会尝试安全切换 |
| 路径含中文报错 | 将整个项目移至纯 ASCII 路径后重跑 `setup_win.bat` |
| 非 NVIDIA 环境 | 不支持（依赖 CUDA 运行时） |
| 母版音色培育 / 情绪语音合成不可用 | 大概率未配置 Qwen3-TTS（外部 ComfyUI 环境），见 §3；VoxCPM 引擎不受影响 |

### 日志位置汇总

- 主界面 / 启动器：`logs\webui_restart.log`、`logs\yue2_service.log`
- YuE2 一键包服务：`deps\YuE2-T8-Local-*\logs\server.stderr.log`
- 唱歌流水线：`outputs\logs\` 下（songbloom / 分离 / 换声等）
- 生成产物：`outputs\`

## 7. 已知限制

- **Seed-VC 换声效果一般**（残留原唱腔 / 咬字 / 气声，融合度有限），仅作技术验证，推荐改走 VoxCPM 通道，详见 README「功能状态」。
- **Qwen3-TTS 相关 Tab 依赖外部 ComfyUI 环境**，本仓库不打包。
- **SongBloom 无官方统一发布渠道**，请自行评估来源可靠性。
- 全部推理在本机完成，无云端调用；生成内容含 AI 合成成分，请遵守当地法律法规与平台规则。

## 8. 免责声明

本项目为个人本地工具，代码按现状提供，不承诺任何效果与稳定性。第三方组件（YuE2、Seed-VC、VoxCPM、Qwen3-TTS、demucs 等）版权归其各自作者所有，分发时请遵守对应开源协议。
