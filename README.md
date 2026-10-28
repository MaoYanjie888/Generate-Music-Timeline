# Generate-Music-Timeline

纯本地部署的歌曲生成 + 歌词时间戳工具套件（Windows / NVIDIA）。整合 YuE2 歌曲母版生成、Seed-VC 参考音色换声、VoxCPM / SongBloom 备用合成通道、母版音色培育、MVGR 歌词时间戳精对齐，全部本地运行，无需联网。

> 本仓库仅包含代码与配置。模型权重、依赖一键包体积巨大（约 200GB），不在仓库内，需按文末「依赖准备」自行获取。

## 功能状态（重要）

| 功能 | 状态 | 说明 |
| --- | --- | --- |
| 歌曲母版生成（YuE2） | ✅ 可用 | style + lyrics 生成母版 vocal / 伴奏，长曲支持 |
| 歌词时间戳精对齐（MVGR） | ✅ 可用 | ForcedAligner 多级窗口 + 线性补偿，输出 LRC / JSON |
| 母版音色培育 | ✅ 可用 | 基于参考音色 + 语料批量克隆，产出可试听样本 |
| 备用合成通道（VoxCPM / SongBloom） | ✅ 可用 | 零样本歌声克隆 / 参考音色直接唱歌 |
| **参考音色唱歌（Seed-VC 换声）** | ⚠️ **效果不佳，约等于用不了** | 母版经 Seed-VC 换声后仍残留原唱腔 / 咬字 / 气声，融合度有限，与参考音色差距明显；仅作技术验证，不建议依赖此功能产出成品。推荐改用 VoxCPM 通道 |

## 功能特性

- **歌曲母版生成**：YuE2（T8 本地版）按 style / lyrics 生成母版人声与伴奏，支持 seed、cfg_scale、ABC 乐谱条件
- **歌词发音友好预处理**：提交前用 pypinyin 检测多音字 / 生僻字并替换为常用同音字，降低认错字概率（可开关）
- **音准校正**：换声后以母版 vocal 旋律为基准拉回逐音符跑调，仅在跑调段落局部重合成（可开关）
- **母版音色培育**：Qwen3-TTS 批量克隆，语料按「文本|情绪」逐条合成，页内试听 + 下载
- **MVGR 时间戳精对齐**：多级窗口 ForcedAligner 对齐 + 全局线性补偿 + 单调性整理，适配换歌后的精度退化
- **双通道合成**：Seed-VC 换声通道 / VoxCPM 零样本克隆通道，自动故障回退
- **纯本地**：所有推理在本机完成，无云端调用

## 目录结构

```
Generate-Music-Timeline/
├── 启动工具.bat          # 一键启动（自动定位依赖）
├── setup_win.bat         # 移动项目后运行一次，修复虚拟环境路径
├── webui.py              # 主界面入口（Gradio）
├── MVGR.spec             # MVGR 时间戳工具打包配置
├── singing/              # 唱歌 / 时间戳流水线核心代码
│   ├── pipeline.py       # 主流水线（母版生成 → 换声/克隆 → 音准校正 → 混音）
│   ├── webui.py          # 唱歌页面（Gradio）
│   ├── pitch_correct.py  # 音准校正（WORLD 重合成）
│   ├── mix_master.py     # 母版混音
│   ├── master_voice_webui.py / master_voice_worker.py  # 母版音色培育
│   ├── emotion_tts_worker.py  # 情绪 TTS 引擎
│   ├── voxcpm_worker.py  # VoxCPM 合成 worker
│   ├── songbloom_worker.py   # SongBloom 合成 worker
│   └── demucs_worker.py  # 伴奏分离 worker
├── src/                  # 工具库与 MVGR 时间戳
│   ├── webui/app.py      # MVGR 时间戳对齐界面
│   ├── model_manager.py  # 模型管理
│   └── utils/            # 配置 / 日志 / 路径 / 更新检测
└── configs/              # 配置与预设
    └── presets/          # anime / cinematic / fantasy / realistic 预设
```

## 环境要求

- **Windows 10/11**，仅支持 NVIDIA 显卡（CUDA）
- 显存建议 **16GB**（母版长曲生成 + 换声同时加载较吃显存）
- 磁盘剩余空间 ≥ 200GB（完整依赖）
- Python 3.10（本仓库自带虚拟环境结构，依赖均装于 venv 内）

## 快速开始

1. 准备依赖（见下节），按原始目录结构放置到对应位置
2. 双击 `启动工具.bat`，等待浏览器自动打开主界面
3. Tab「唱歌」生成歌曲：填 style / lyrics，选择合成通道
4. Tab「母版音色培育」批量克隆音色；Tab「MVGR」做歌词时间戳对齐

## 依赖准备

以下内容体积巨大，未随仓库分发，需自行下载/部署：

- `deps/YuE2-T8-Local-*/`：YuE2 歌曲生成一键包（含自带 Python 运行时与模型）
- `deps/` 下 Seed-VC、VoxCPM、SongBloom、Qwen3-TTS 等模型
- `models/`：各模型权重
- `runtime_py310/`：Python 3.10 运行时
- `venv_timeline/`、`venv_voxcpm/`、`venv_songbloom/`：虚拟环境

路径含中文会导致部分工具链异常，建议放在纯 ASCII 路径下。

## 常见问题

| 现象 | 原因与处理 |
| --- | --- |
| 启动后界面打不开 / 服务报错 | 看 `logs\webui_restart.log` 与 `logs\yue2_service.log`；YuE2 服务首次启动需 1~2 分钟加载 |
| 歌曲生成显示错误但产物已存在 | 多为轮询超时误报：母版长曲可达 50~80 分钟，优先查看 `outputs\` 下是否有 `final_song.wav` |
| 生成中途崩溃（0xC0000005 / 卡死） | 多为 GPU 上下文未释放或 pyworld 遇 NaN；pipeline 已加网络兜底、轮询看门狗与音准校正子进程化，仍失败时关闭其他占显存程序重试 |

## 免责声明

- 本项目为个人本地工具，代码按现状提供，不承诺任何效果与稳定性
- 生成内容含 AI 合成成分，请遵守当地法律法规与平台规则，勿用于侵权用途
- 第三方组件（YuE2、Seed-VC、VoxCPM、Qwen3-TTS、demucs 等）版权归其各自作者所有，分发时请遵守对应开源协议
