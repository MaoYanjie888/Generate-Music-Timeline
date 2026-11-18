---
AIGC:
    Label: "1"
    ContentProducer: 001191440300708461136T1XGW3
    ProduceID: d80f7d114115af5f21aec47fe9775375_fed96b52d2b211f1992e525400638852
    ReservedCode1: mVpWXHxN8g1P6pXxh59wKsmpfumVXUE9ny2hwCf06fW1aMIcpMOYmRCeqaPGDkDZyvvF/8lz4i/4jPTNHARzOT0RbqckpeGVvF4NfLrkgFHJC2qAdpoCFKcROMCR1+JpiGH7rubWiN2iZbfUZF8sEPKKuw4ZleCbVI0ezmlpLouLNgeTAF38VP5t4wg=
    ContentPropagator: 001191440300708461136T1XGW3
    PropagateID: d80f7d114115af5f21aec47fe9775375_fed96b52d2b211f1992e525400638852
    ReservedCode2: mVpWXHxN8g1P6pXxh59wKsmpfumVXUE9ny2hwCf06fW1aMIcpMOYmRCeqaPGDkDZyvvF/8lz4i/4jPTNHARzOT0RbqckpeGVvF4NfLrkgFHJC2qAdpoCFKcROMCR1+JpiGH7rubWiN2iZbfUZF8sEPKKuw4ZleCbVI0ezmlpLouLNgeTAF38VP5t4wg=
---

# Generate-Music-Timeline

纯本地部署的一站式音乐创作工作台（Windows / NVIDIA）。提供 **歌词时间戳精对齐、情绪语音合成、母版音色培育、ACE-Step 音色跟随** 四个功能 Tab，全部本地运行，数据不出本机。

> 本仓库仅包含代码与配置。模型权重、依赖一键包体积巨大（约 200GB），不在仓库内，需按文末「依赖准备」自行获取。

## 功能状态（重要）

| Tab | 功能 | 状态 | 说明 |
| --- | --- | --- | --- |
| **① 歌词时间戳** | 音频 + 歌词 → 逐句起止时间 | ✅ 可用 | MVGR ForcedAligner 多级窗口 + 线性补偿，导出 lrc / csv |
| **② 情绪语音合成** | 音色 + 文字 + 情绪 → 带情绪朗读 | ✅ 可用 | 基于 Qwen3-TTS，需外部 ComfyUI 环境 |
| **③ 母版音色培育** | 声音设计 / 参考音色克隆 | ✅ 可用 | 基于 Qwen3-TTS，语料逐条合成 + 页内试听下载 |
| **④ ACE-Step 音色跟随** | 参考音频 → Cover 翻唱 | ✅ 可用 | 端到端无机械音，走独立 REST API（推荐） |

> 旧版「一键成曲 / 唱歌创作 / SoulX Score 按谱合成」链路及其 VoxCPM / SongBloom / Seed-VC 换声通道已从本版本移除，相关代码与入口不再保留。当前版本聚焦上表四个 Tab。

## 功能特性

- **① 歌词时间戳精对齐（MVGR）**：Whisper + Qwen3-ASR 识别，ForcedAligner 多级窗口对齐 + 全局线性补偿 + 单调性整理，输出 lrc / csv，适配换歌后的精度退化
- **② 情绪语音合成**：音色 + 文字 + 情绪标签合成带情绪的朗读语音（Qwen3-TTS）
- **③ 母版音色培育**：声音设计或参考音色克隆，语料按「文本|情绪」逐条合成，页内试听 + 下载
- **④ ACE-Step 音色跟随**：参考音频经独立 ACE-Step REST API 做 Cover 翻唱，端到端生成、无机械音；API 服务独立运行、环境隔离，数据只在 127.0.0.1 回环传输
- **纯本地**：所有推理在本机完成，无云端调用

## 目录结构

```
Generate-Music-Timeline/
├── 启动工具.bat          # 一键启动（自动定位依赖 → 检查服务 → 打开主界面）
├── setup_win.bat         # 移动项目后运行一次，修复虚拟环境路径
├── start_acestep_api.bat # 启动 Tab④ 所需的 ACE-Step 独立 REST API（127.0.0.1:8001）
├── webui.py              # 主界面入口（Gradio），注册下列四个 Tab
├── requirements.txt      # venv_timeline 主环境依赖清单
├── DEPLOY.md             # 从零部署教程
├── MVGR.spec             # MVGR 时间戳工具打包配置
├── singing/              # 语音合成 / 音色培育 / Cover 模块
│   ├── emotion_tts_webui.py / emotion_tts_worker.py  # Tab② 情绪语音合成
│   ├── master_voice_webui.py / master_voice_worker.py  # Tab③ 母版音色培育
│   └── acestep_cover_ui.py   # Tab④ ACE-Step 音色跟随（HTTP 调用独立 API）
├── src/                  # 工具库与 MVGR 歌词时间戳
│   ├── webui/app.py      # Tab① 歌词时间戳对齐界面
│   ├── model_manager.py  # 模型管理
│   └── utils/            # 配置 / 日志 / 路径 / 更新检测
└── configs/              # 配置与预设（presets/）
```

## 环境要求

- **Windows 10/11**，仅支持 NVIDIA 显卡（CUDA）
- 显存建议 **16GB**（母版音色培育、ACE-Step 音色跟随加载模型较吃显存）
- 磁盘剩余空间 ≥ 200GB（完整依赖）
- Python 3.10（本仓库自带虚拟环境结构，依赖均装于 venv 内）

## 部署教程

> **仓库不含依赖**：本仓库仅包含代码与配置，模型权重、一键包、虚拟环境等依赖（约 200GB）未随仓库分发。从 GitHub 下载后必须先按 [DEPLOY.md](DEPLOY.md) 完成部署，否则无法运行。

1. 阅读 [DEPLOY.md](DEPLOY.md)，按「环境要求」确认本机条件
2. 按 DEPLOY.md「组件依赖获取渠道」获取并放置各组件依赖
3. 运行 `setup_win.bat` 修复虚拟环境路径（移动项目 / 换电脑后必须执行）
4. 双击 `启动工具.bat` 启动主界面
5. 使用 Tab④ 前，另开窗口运行 `start_acestep_api.bat` 启动独立 API

## 快速开始

1. 准备依赖（见下节），按原始目录结构放置到对应位置
2. 双击 `启动工具.bat`，等待浏览器自动打开主界面
3. **Tab「① 歌词时间戳」**：上传歌曲音频与歌词 → 输出逐句起止时间 → 导出 lrc / csv
4. **Tab「② 情绪语音合成」**：选音色、填文字与情绪 → 合成带情绪朗读
5. **Tab「③ 母版音色培育」**：声音设计或参考音色克隆 → 批量合成并试听下载
6. **Tab「④ ACE-Step 音色跟随」**：先运行 `start_acestep_api.bat`，再上传参考音频 → Cover 翻唱

## 依赖准备

以下内容体积巨大，未随仓库分发，需自行下载/部署：

- `models/`：`large-v3.pt`（Whisper）、`Qwen3-ASR-1.7B-hf`、`Qwen3-ForcedAligner-0.6B-hf`（Tab① 歌词时间戳）
- **Qwen3-TTS（Tab②③）**：模型权重与 `qwen_tts` 包位于外部 ComfyUI 环境（按 `F:\ComfyUI` 布局引用：`models\Qwen3-TTS\Qwen3-TTS-12Hz-1.7B-Base` 与 `...-VoiceDesign`）
- **ACE-Step 1.5（Tab④）**：独立的 `F:\ACE-Step-1.5` 环境（自带 venv），由 `start_acestep_api.bat` 在 `127.0.0.1:8001` 启动 REST API
- `runtime_py310/`：Python 3.10 运行时
- `venv_timeline/`：主界面虚拟环境

> `deps/YuE2-T8-Local-*` 一键包仅为启动器的历史定位项；四个 Tab 的核心链路已不依赖 YuE2 服务。

路径含中文会导致部分工具链异常，建议放在纯 ASCII 路径下。

## 常见问题

| 现象 | 原因与处理 |
| --- | --- |
| 启动后界面打不开 / 服务报错 | 查看 `logs\webui_restart.log`；首次加载模型约 1~2 分钟 |
| Tab② ③ 不可用 | 大概率未配置 Qwen3-TTS（外部 ComfyUI 环境），见「依赖准备」 |
| Tab④ 提示 API 未连接 | 先运行 `start_acestep_api.bat` 启动独立 API，确认 `127.0.0.1:8001` 就绪 |
| 时间戳对齐偏差 | 用多级窗口 + 线性补偿重跑；换歌后建议重新对齐 |

## 免责声明

- 本项目为个人本地工具，代码按现状提供，不承诺任何效果与稳定性
- 生成内容含 AI 合成成分，请遵守当地法律法规与平台规则，勿用于侵权用途
- 第三方组件（Qwen3-TTS、ACE-Step、Whisper、demucs 等）版权归其各自作者所有，分发时请遵守对应开源协议
*（内容由AI生成，仅供参考）*
