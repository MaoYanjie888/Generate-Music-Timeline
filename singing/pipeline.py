# -*- coding: utf-8 -*-
"""
YuE2 参考音色歌曲创作管线 (双合成通道)
=============================================
一键流程: 参考音色 + 歌词 + 风格提示词 -> 歌曲

合成通道 (--channel):
  - voice_convert (默认, 推荐): 母版人声 -> 音色转换 (恢复主通道)
      YuE2 生成母版 -> htdemucs 分离 vocal+acc ->
      T8 内置 Seed-VC 将「母版 vocal」换成参考音色(保留母版旋律唱腔) ->
      mix_audio 混音(换声 vocal + 母版伴奏)
  - voxcpm (备用): VoxCPM2 零样本歌声克隆 (纯文本合成)
      YuE2 生成母版 -> htdemucs 分离伴奏 ->
      voxcpm_worker(venv_voxcpm) 用歌词+参考音色直接合成干声(48kHz) ->
      mix_audio 混音
      说明: VoxCPM 为纯 TTS 模型, generate() 无旋律/音高条件输入,
            合成结果偏「朗读腔」; 仅作为无 Seed-VC 场景的备用通道。

运行环境: 主流程由项目主 python(venv_timeline) 运行;
          VoxCPM 合成在独立 venv F:\Generate-Music-Timeline\venv_voxcpm 中执行。
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

# 音准校正模块(主通道换声后以母版 vocal 旋律为基准拉回音高)
try:
    from . import pitch_correct
except ImportError:  # 以脚本方式直接运行 pipeline.py 时
    import pitch_correct

# 专业母带混音模块(主通道: 人声低切+压缩+混响+响度对齐母版)
try:
    from . import mix_master
except ImportError:
    import mix_master

# ---------------- 常量 ----------------
YUE2_SERVICE = "http://127.0.0.1:8189"
# 项目根 = singing/ 的上一级; 所有路径均相对推导, 整个项目文件夹可整体移动/拷贝到任意位置(开源可移植)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# VoxCPM 独立环境(主 Agent 已部署): venv + 本地权重
VOXCPM_VENV_PY = PROJECT_ROOT / "venv_voxcpm" / "Scripts" / "python.exe"
VOXCPM_WORKER = Path(__file__).resolve().parent / "voxcpm_worker.py"
VOXCPM_MODEL_DIR = PROJECT_ROOT / "models" / "VoxCPM2"

# SongBloom 独立环境(参考音色生成歌曲, 替代 YuE2 母版): venv + 本地权重
# 权重由主 Agent 预先下载到 deps/SongBloom/cache/model (backups/CypressYang-SongBloom 镜像)
SONGBLOOM_VENV_PY = PROJECT_ROOT / "venv_songbloom" / "Scripts" / "python.exe"
SONGBLOOM_WORKER = Path(__file__).resolve().parent / "songbloom_worker.py"
SONGBLOOM_MODEL_DIR = PROJECT_ROOT / "deps" / "SongBloom" / "cache" / "model"
SONGBLOOM_MODEL_NAME = "songbloom_full_150s"
DEFAULT_SONGBLOOM_DTYPE = "bfloat16"
SONGBLOOM_PROMPT_SEC = 10  # SongBloom 参考音色要求前 10s(48kHz)

# VoxCPM 合成默认参数
DEFAULT_CFG_VALUE = 2.0
DEFAULT_INFERENCE_TIMESTEPS = 10

# 合成通道
DEFAULT_CHANNEL = "voice_convert"

# Seed-VC 换声默认参数 (T8 voice_convert, backend=seed-vc)
# 20261022 换声质量优化(电音/机械音 + 音色旋律融合):
#   - diffusion_steps 30->50: 扩散步数越多细节越细腻, 机械/电音伪影更少
#   - cfg_rate 0.7->0.5: 内容 CFG 过高会锁死内容特征产生金属电音感, 降低更自然
#   - auto_f0_adjust True->False: 关闭全局音域平移, 让换声直接保留母版旋律,
#     音准由后续 pitch_correct 兜底, 避免「先移调再拉回」双重处理造成音色劣化
DEFAULT_SEEDVC_DIFFUSION_STEPS = 50
DEFAULT_SEEDVC_CFG_RATE = 0.5
DEFAULT_SEEDVC_AUTO_F0_ADJUST = False
DEFAULT_SEMI_TONE_SHIFT = 0

# 音准校正默认参数 (主通道换声后, 以母版 vocal 旋律为基准拉回跑调音高)
# 20261022: Seed-VC 换声后逐音符跑调根因是 auto_f0_adjust 全局中位音高平移
# + 分离残留伴奏污染 RMVPE f0 提取; 该模块在换声轨上做慢变音高曲线校正。
DEFAULT_PITCH_CORRECT = True
DEFAULT_PITCH_CORRECT_SMOOTH_SEC = 0.5
DEFAULT_PITCH_CORRECT_THRESH_ST = 0.15
DEFAULT_PITCH_CORRECT_MAX_ST = 2.0

# 参考音色扩增目标时长(extend_reference 可复用函数默认参数)
REF_PROMPT_TARGET_SEC = 15.0


def _find_yue2_kit() -> Path:
    """在 deps/ 下自动定位 YuE2 一键包目录(目录名随版本可能变化, 不写死)。"""
    deps = PROJECT_ROOT / "deps"
    if deps.is_dir():
        for p in deps.iterdir():
            if p.is_dir() and p.name.startswith("YuE2-T8-Local"):
                return p
    return deps / "YuE2-T8-Local-v1.6.0-Windows-NVIDIA-20260919"


YUE2_KIT = _find_yue2_kit()
# YuE2 内置 htdemucs 分离环境(runtime 自带 demucs + 模型, 用于从母版分离人声/伴奏)
YUE2_RUNTIME_PY = YUE2_KIT / "runtime" / "python.exe"
DEMUCS_WORKER = Path(__file__).resolve().parent / "demucs_worker.py"
# 新项目统一输出目录(原 F:\YuE2-SoulX-Studio\outputs 已迁移到本项目下)
PROJECT_OUTPUTS = PROJECT_ROOT / "outputs"
# 离线依赖: HF 模型缓存随项目走, 不再依赖 C 盘用户缓存
os.environ["HF_HOME"] = str(PROJECT_ROOT / "hf_cache")
os.environ.setdefault("HF_HUB_CACHE", str(PROJECT_ROOT / "hf_cache" / "hub"))
YUE2_POLL_INTERVAL = 3.0
YUE2_POLL_TIMEOUT = 7200  # 生成一首歌最长等待 120 分钟(GPU 被占时 YuE2 可能超 60 分钟)


# ---------------- YuE2 HTTP 客户端 ----------------
def yue2_request(path: str, method="GET", data=None, timeout=30):
    body = None if data is None else json.dumps(data, ensure_ascii=False).encode("utf-8")
    headers = {"Content-Type": "application/json; charset=utf-8"} if body is not None else {}
    req = urllib.request.Request(YUE2_SERVICE + path, data=body, method=method, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        try:
            detail = json.loads(exc.read().decode("utf-8")).get("error")
        except Exception:
            detail = str(exc)
        raise RuntimeError(f"YuE2 接口错误: {detail}")
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        # 20261028: 瞬时网络挂起(服务端 GC/连接半开)直接冒泡会让上层误判为
        # 生成失败; 统一转成 "YuE2 接口错误" 前缀, 由 yue2_wait 的
        # transient_retries 自动短重试, 而不是中断整次生成。
        raise RuntimeError(f"YuE2 接口错误: 网络请求失败 {exc}")


def yue2_health():
    return yue2_request("/api/health", timeout=5)


def yue2_submit(payload: dict) -> str:
    resp = yue2_request("/api/jobs", method="POST", data={"kind": "generate", "request": payload, "source": "soulx-studio"})
    job_id = resp.get("id")
    if not job_id:
        raise RuntimeError(f"YuE2 提交失败: {resp}")
    return job_id


def yue2_submit_kind(kind: str, payload: dict) -> str:
    """按任务类型提交 YuE2 服务(如 voice_convert 参考音色换声)。"""
    resp = yue2_request("/api/jobs", method="POST",
                        data={"kind": kind, "request": payload, "source": "soulx-studio"})
    job_id = resp.get("id")
    if not job_id:
        raise RuntimeError(f"YuE2 提交失败({kind}): {resp}")
    return job_id


def yue2_wait(job_id: str, progress_cb=None):
    """轮询任务直到完成, 返回完整 job 结果"""
    start = time.monotonic()
    # 20261013: 服务端写 status.json 用 os.replace, Windows 下文件句柄被短暂占用时
    # 抛 PermissionError 转 400; 此处对瞬时接口错误增加短重试, 避免一次 400 即放弃后处理
    transient_retries = 10
    while time.monotonic() - start < YUE2_POLL_TIMEOUT:
        poll_t0 = time.monotonic()
        try:
            status = yue2_request(f"/api/jobs/{job_id}")
        except RuntimeError as exc:
            if transient_retries > 0 and str(exc).startswith("YuE2 接口错误"):
                transient_retries -= 1
                time.sleep(YUE2_POLL_INTERVAL)
                continue
            raise
        poll_cost = time.monotonic() - poll_t0
        if poll_cost > 60:
            # 20261028 看门狗: 单次轮询异常缓慢说明服务端挂起/网络半开,
            # 打印告警便于事后定位, 但不打断流程(超时由 yue2_request 兜底重试)
            print(f"[warn] YuE2 轮询异常缓慢: {poll_cost:.1f}s (job {job_id})", flush=True)
        st = status.get("status")
        if progress_cb:
            progress_cb(status)
        if st == "complete":
            return status
        if st in ("failed", "cancelled"):
            raise RuntimeError(status.get("error") or f"YuE2 任务{st}: {job_id}")
        time.sleep(YUE2_POLL_INTERVAL)
    raise RuntimeError("YuE2 生成超时")


# ---------------- 歌词发音友好预处理 ----------------
# 20261028: YuE2 无音素化前置, 汉字->发音由模型隐式习得, 生僻字/多音字易唱错。
# 提交前做发音友好预处理: 高置信同音替换 + 无法注音字检测 + 多音字提示。
# 替换策略保守: 命中表且 pypinyin 验证替换前后读音完全一致才替换, 绝不引入新错音。

# 高置信同音/异体字替换表(异体字/生僻写法 -> 常用字, 读音一致)
_LYRIC_REPLACE_TABLE = {
    "峯": "峰", "崑": "昆", "羣": "群", "裏": "里", "谿": "溪",
    "僊": "仙", "啟": "启", "喫": "吃", "勳": "勋", "巖": "岩",
    "徧": "遍", "慾": "欲", "攬": "揽", "猶": "犹", "萇": "苌",
    "贄": "贽", "釵": "钗", "閨": "闺", "鳳": "凤", "鸞": "鸾",
    "瓊": "琼", "絳": "绛", "紈": "纨", "緗": "缃", "驪": "骊",
}


def _sanitize_lyrics_for_pronunciation(lyrics: str, replace_map: dict = None) -> tuple[str, list[str]]:
    """歌词发音友好预处理。

    做三件事:
      1. 高置信替换: 内置异体字表(或调用方 replace_map)命中且读音一致的字,
         替换为常用字; 替换前用 pypinyin 验证前后读音完全相同, 避免引入新错音;
      2. 无法注音字检测: pypinyin 无法注音的中文字(超词表/非常用字)逐字列出提示;
      3. 多音字提示: 常见高频多音字出现时提示人工确认读音。

    返回 (清洗后歌词, 提示列表)。默认不改原歌词结构, 仅保守替换与提示。
    """
    notes: list[str] = []
    try:
        from pypinyin import lazy_pinyin
    except Exception:
        return lyrics, ["[warn] pypinyin 未安装, 跳过歌词发音预处理"]

    # 1) 高置信替换(读音一致性校验)
    table: dict[str, str] = dict(_LYRIC_REPLACE_TABLE)
    if replace_map:
        table.update(replace_map)
    out_chars: list[str] = []
    for ch in lyrics:
        rep = table.get(ch)
        if rep:
            try:
                old_py = lazy_pinyin(ch)[0]
                new_py = lazy_pinyin(rep)[0]
                if old_py == new_py and old_py != ch:
                    out_chars.append(rep)
                    notes.append(f"替换 {ch!r}({old_py}) -> {rep!r}")
                    continue
            except Exception:
                pass
        out_chars.append(ch)
    cleaned = "".join(out_chars)

    # 2) 无法注音字检测(仅统计 CJK 范围, 排除标点/emoji)
    unknown: list[str] = []
    for ch in sorted(set(cleaned)):
        if "\u4e00" <= ch <= "\u9fff":
            try:
                if lazy_pinyin(ch)[0] == ch:
                    unknown.append(ch)
            except Exception:
                unknown.append(ch)
    if unknown:
        notes.append("无法注音字(模型大概率不认, 建议改常用同音字): " + "、".join(unknown))

    # 3) 高频多音字提示
    _common_poly = ("行长乐重还朝卷盛藏和了着地得曲调干数强便切觉角供散弹圈宿鲜燕华"
                    "传转分更都发单折累漂铺塞舍石提相兴应正种为似系纤血吓好背辟薄称乘"
                    "冲处担当倒钉斗肚度恶范缝服给龟柜过号荷喝横哄划会几夹假间监将降教尽"
                    "劲据卡看壳空露落埋脉闷蒙秘模难宁跑片浅抢悄茄亲任丧扫色杉扇少识熟刷"
                    "衰说宿遂踏汤通同吐拓委尾削压咽叶遗与晕载脏扎粘涨只中轴钻")
    hit_poly = sorted({ch for ch in cleaned if ch in _common_poly})
    if hit_poly:
        notes.append("含常见多音字(建议确认读音): " + "、".join(hit_poly))

    return cleaned, notes


# ---------------- 音频分离 / 处理 ----------------
def htdemucs_separate(audio_path: Path, vocal_out: Path, acc_out: Path, log_tag="sep"):
    """用 YuE2 内置 htdemucs 分离人声/伴奏。

    htdemucs(来自 YuE2 runtime)对母版实测 vocal 覆盖 >95%, 唱段无残留。
    返回 (vocal_out, acc_out), 均为 44100Hz float32 wav。
    """
    cmd = [str(YUE2_RUNTIME_PY), str(DEMUCS_WORKER), str(audio_path), str(vocal_out), str(acc_out)]
    log_dir = PROJECT_OUTPUTS / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    out_log = log_dir / f"{log_tag}_{stamp}.log"
    err_log = log_dir / f"{log_tag}_{stamp}_err.log"
    env = os.environ.copy()
    env["YUE2_KIT"] = str(YUE2_KIT)  # 告诉 worker 一键包位置(避免其自身推导失败)
    with open(out_log, "w", encoding="utf-8") as fout, open(err_log, "w", encoding="utf-8") as ferr:
        proc = subprocess.run(cmd, stdout=fout, stderr=ferr, env=env)
    if proc.returncode != 0:
        tail = ""
        if err_log.exists():
            tail = "\n".join(err_log.read_text(encoding="utf-8", errors="replace").splitlines()[-30:])
        raise RuntimeError(f"htdemucs 分离[{log_tag}]失败, 退出码 {proc.returncode}\n{tail}")
    if not Path(vocal_out).is_file() or not Path(acc_out).is_file():
        raise RuntimeError(f"htdemucs 分离[{log_tag}]未生成 vocal/acc")
    return Path(vocal_out), Path(acc_out)


def t8_voice_convert(source_audio: Path, reference_audio: Path, save_dir: Path,
                     backend: str = "seed-vc",
                     diffusion_steps: int = DEFAULT_SEEDVC_DIFFUSION_STEPS,
                     cfg_rate: float = DEFAULT_SEEDVC_CFG_RATE,
                     auto_f0_adjust: bool = DEFAULT_SEEDVC_AUTO_F0_ADJUST,
                     semi_tone_shift: int = DEFAULT_SEMI_TONE_SHIFT,
                     vocal_gain_db: float = 0.0, accompaniment_gain_db: float = 0.0,
                     log_tag: str = "t8vc") -> dict:
    """调用 YuE2 服务内置 voice_convert(Seed-VC)完成参考音色换声。

    T8 整合包自带 Seed-VC(DiT-seed-v2 uvit whisper-base 44k), 是当前社区公认
    本地音色融合质量最高的 zero-shot 歌声转换: 音色贴合度高且保留源 vocal 的
    旋律/情感/气声/唱腔 —— 即「母版人声换参考音色」主通道。

    服务端要求 source/reference 必须位于 T8 包 uploads/ 或 outputs/jobs 下,
    函数自动复制、调用后清理临时上传文件。
    返回 T8 job result(含 audio/converted_vocal/accompaniment 等路径)。
    """
    uploads = YUE2_KIT / "uploads"
    uploads.mkdir(parents=True, exist_ok=True)
    # 20261027: 母版生成/htdemucs 分离刚结束后 GPU CUDA 上下文未完全释放,
    # Seed-VC worker 启动即原生崩溃(0xC0000005/python312.dll, 日志0字节),
    # 静止态重跑换声可成功; 此处对原生崩溃失败自动重试一次, 避免整次生成白跑。
    crash_retries = 1
    while True:
        stamp = time.strftime("%Y%m%d-%H%M%S")
        src_up = uploads / f"src_{log_tag}_{stamp}{source_audio.suffix}"
        ref_up = uploads / f"ref_{log_tag}_{stamp}{reference_audio.suffix}"
        shutil.copy2(source_audio, src_up)
        shutil.copy2(reference_audio, ref_up)
        payload = {
            "backend": backend,
            "source_path": str(src_up),
            "reference_path": str(ref_up),
            "diffusion_steps": int(diffusion_steps),
            "cfg_rate": float(cfg_rate),
            "auto_f0_adjust": bool(auto_f0_adjust),
            "semi_tone_shift": int(semi_tone_shift),
            "vocal_gain_db": float(vocal_gain_db),
            "accompaniment_gain_db": float(accompaniment_gain_db),
        }
        try:
            job_id = yue2_submit_kind("voice_convert", payload)
            job = yue2_wait(job_id)
            break
        except RuntimeError as exc:
            msg = str(exc)
            if crash_retries > 0 and ("3221225477" in msg or "0xC0000005" in msg or "ACCESS_VIOLATION" in msg):
                crash_retries -= 1
                time.sleep(20)  # 等待 GPU CUDA 上下文彻底释放
                continue
            raise
        finally:
            # 清理临时上传(服务端任务已把文件物化到 job 目录)
            for p in (src_up, ref_up):
                try:
                    p.unlink(missing_ok=True)
                except OSError:
                    pass
    result = job.get("result") or {}
    if not result.get("audio"):
        raise RuntimeError(f"voice_convert({backend}) 未返回结果: {json.dumps(result, ensure_ascii=False)[:500]}")
    return result


def extend_reference(src: Path, out: Path, target_sec: float = REF_PROMPT_TARGET_SEC,
                     crossfade_ms: float = 300.0):
    """参考音色扩增: 循环拼接 + 交叉淡化到 target_sec。

    通用可复用工具: 当下游模型需要更长参考片段时循环拼接, 交叉淡化避免
    拼接点咔哒声。输入可能带伴奏, 扩增后可再交给 htdemucs 分离。
    """
    import numpy as np
    import soundfile as sf

    y, sr = sf.read(str(src), dtype="float32")
    if y.ndim > 1:
        y = y.mean(axis=1)
    total = int(sr * target_sec)
    if len(y) >= total:
        sf.write(str(out), y[:total], sr, subtype="FLOAT")
        return out
    cf = min(int(sr * crossfade_ms / 1000.0), len(y) // 2, total // 2)
    out_ = np.zeros(total, dtype=np.float32)
    pos = 0
    while pos < total:
        seg = y[: min(len(y), total - pos + cf)]
        if pos == 0:
            out_[pos:pos + len(seg)] = seg
            pos += len(seg)
            continue
        # 交叉淡化: 前 cf 采样与已写区重叠渐变
        ov = min(cf, len(seg))
        fade = 0.5 - 0.5 * np.cos(np.linspace(0.0, np.pi, ov))
        out_[pos - ov:pos] = out_[pos - ov:pos] * (1.0 - fade) + seg[:ov] * fade
        out_[pos:pos + len(seg) - ov] = seg[ov:]
        pos += len(seg) - ov
    out.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out), out_, sr, subtype="FLOAT")
    return out


# ---------------- VoxCPM 子进程调用 ----------------
def voxcpm_synthesize(lyrics: str, reference: Path, out: Path,
                      reference_text: str = None, auto_transcribe: bool = True,
                      cfg_value: float = DEFAULT_CFG_VALUE,
                      inference_timesteps: int = DEFAULT_INFERENCE_TIMESTEPS,
                      denoise: bool = False, log_tag: str = "voxcpm") -> Path:
    """在 venv_voxcpm 中调用 voxcpm_worker.py, 用歌词+参考音色直接合成干声。

    返回输出 wav 路径(48kHz, model.tts_model.sample_rate)。
    """
    cmd = [
        str(VOXCPM_VENV_PY), str(VOXCPM_WORKER),
        "--lyrics", lyrics,
        "--reference", str(reference),
        "--output", str(out),
        "--cfg_value", str(float(cfg_value)),
        "--inference_timesteps", str(int(inference_timesteps)),
    ]
    if reference_text and str(reference_text).strip():
        cmd += ["--reference_text", str(reference_text).strip()]
    elif auto_transcribe:
        cmd += ["--auto_transcribe"]
    if denoise:
        cmd.append("--denoise")
    log_dir = PROJECT_OUTPUTS / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    out_log = log_dir / f"{log_tag}_{stamp}.log"
    err_log = log_dir / f"{log_tag}_{stamp}_err.log"
    env = os.environ.copy()
    with open(out_log, "w", encoding="utf-8") as fout, open(err_log, "w", encoding="utf-8") as ferr:
        proc = subprocess.run(cmd, stdout=fout, stderr=ferr, env=env)
    if proc.returncode != 0:
        tail = ""
        if err_log.exists():
            tail = "\n".join(err_log.read_text(encoding="utf-8", errors="replace").splitlines()[-30:])
        raise RuntimeError(f"VoxCPM 合成[{log_tag}]失败, 退出码 {proc.returncode}\n{tail}")
    out = Path(out)
    if not out.is_file():
        raise RuntimeError(f"VoxCPM 合成[{log_tag}]未生成输出: {out}")
    return out


# ---------------- SongBloom 母版生成 ----------------
def lyrics_to_songbloom(lyrics: str) -> str:
    """将普通歌词文本转换为 SongBloom token 格式。

    SongBloom 歌词规范(docs/lyric_format.md):
      - [intro]/[inst]/[outro] 为纯器乐段(仅结构标记, 1 标记约 1 秒);
      - [verse]/[chorus]/[bridge] 为演唱段标记, 每句以句号 "." 结尾;
      - 句号分隔句子, 逗号 "," 仅出现在演唱/器乐段末尾(最后一段除外)。
    转换规则: 按空行分段 -> 段落轮流标记 verse/chorus/bridge -> 句内补句号 ->
    开头加 2 个 [intro], 结尾加 4 个 [outro]。
    """
    sections = [s.strip() for s in re.split(r"\n\s*\n|\r\n\s*\r\n", lyrics.strip()) if s.strip()]
    if not sections:
        return ""
    marks = ["verse", "chorus", "bridge"]
    parts = ["[intro]", "[intro]"]
    for i, sec in enumerate(sections):
        # 清洗: 去掉用户可能手写的结构标记与残留方括号
        sec = re.sub(r"^\[(verse|chorus|bridge|intro|inst|outro)\]\s*", "", sec, flags=re.I)
        sec = re.sub(r"[\[\]]", "", sec).strip()
        sec = re.sub(r"\s+", " ", sec).strip()
        if not sec:
            continue
        sentences = [s.strip() for s in re.split(r"[。.!?；;]+", sec) if s.strip()]
        sentences = [s + "." for s in sentences]
        mark = marks[i % len(marks)]
        parts.append(f"[{mark}] " + " ".join(sentences))
        if i != len(sections) - 1:
            parts.append(",")
    parts += [",", "[outro]", "[outro]", "[outro]", "[outro]"]
    return " ".join(parts)


def songbloom_synthesize(lyrics_songbloom: str, prompt_wav: Path, out: Path,
                         dtype: str = DEFAULT_SONGBLOOM_DTYPE,
                         log_tag: str = "songbloom") -> Path:
    """在 venv_songbloom 中调用 songbloom_worker.py, 用参考音色+歌词生成母版歌曲。

    返回输出 flac 路径(48kHz)。权重走 deps/SongBloom/cache/model 本地缓存, 不联网。
    """
    if not SONGBLOOM_VENV_PY.is_file():
        raise RuntimeError(f"SongBloom 环境缺失: {SONGBLOOM_VENV_PY}")
    if not SONGBLOOM_MODEL_DIR.is_dir():
        raise RuntimeError(f"SongBloom 权重缺失: {SONGBLOOM_MODEL_DIR} (需 songbloom_full_150s.pt 等)")
    cmd = [
        str(SONGBLOOM_VENV_PY), str(SONGBLOOM_WORKER),
        "--lyrics", lyrics_songbloom,
        "--prompt", str(prompt_wav),
        "--output", str(out),
        "--local-dir", str(SONGBLOOM_MODEL_DIR),
        "--model-name", SONGBLOOM_MODEL_NAME,
        "--dtype", str(dtype),
    ]
    log_dir = PROJECT_OUTPUTS / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    out_log = log_dir / f"{log_tag}_{stamp}.log"
    err_log = log_dir / f"{log_tag}_{stamp}_err.log"
    with open(out_log, "w", encoding="utf-8") as fout, open(err_log, "w", encoding="utf-8") as ferr:
        proc = subprocess.run(cmd, stdout=fout, stderr=ferr)
    if proc.returncode != 0:
        tail = ""
        if err_log.exists():
            tail = "\n".join(err_log.read_text(encoding="utf-8", errors="replace").splitlines()[-30:])
        raise RuntimeError(f"SongBloom 母版生成[{log_tag}]失败, 退出码 {proc.returncode}\n{tail}")
    out = Path(out)
    if not out.is_file():
        raise RuntimeError(f"SongBloom 母版生成[{log_tag}]未生成输出: {out}")
    return out


# ---------------- 混音 ----------------
def mix_audio(vocal: Path, acc: Path, out: Path, vocal_gain_db: float = 0.0, acc_gain_db: float = 0.0):
    """用 numpy 混音: vocal + acc -> out (wav); 采样率不一致时用 librosa 重采样对齐"""
    import numpy as np
    import soundfile as sf

    v, sr_v = sf.read(str(vocal), dtype="float32")
    a, sr_a = sf.read(str(acc), dtype="float32")

    # 单声道归一(必须放在重采样之前):
    # librosa.resample 会把 (n, 2) 立体声误当作「n 个通道 × 2 个样本」,
    # 逐样本独立重采样导致极慢(soxr_hq 下整曲需 15~20 分钟)且输出长度错乱。
    if v.ndim > 1:
        v = v.mean(axis=1)
    if a.ndim > 1:
        a = a.mean(axis=1)

    if sr_v != sr_a:
        # 重采样到统一采样率(取较高者, 用 librosa)
        # 20261013 修复: 默认 res_type=soxr_hq 在立体声 44.1k->48k 重采样极慢
        #   (5 秒音频耗时约 30s, 整曲需 15~20 分钟导致混音假死);
        #   改用 polyphase(基于 numpy/scipy 的多相滤波器, 音质与速度均衡, 秒级完成)。
        import librosa
        target_sr = max(sr_v, sr_a)
        if sr_v != target_sr:
            v = librosa.resample(v, orig_sr=sr_v, target_sr=target_sr, res_type="polyphase")
            sr_v = target_sr
        if sr_a != target_sr:
            a = librosa.resample(a, orig_sr=sr_a, target_sr=target_sr, res_type="polyphase")
            sr_a = target_sr

    length = max(len(v), len(a))
    mix = np.zeros(length, dtype=np.float32)
    v_gain = 10 ** (vocal_gain_db / 20)
    a_gain = 10 ** (acc_gain_db / 20)
    mix[: len(v)] += v * v_gain
    mix[: len(a)] += a * a_gain

    # 简单限幅
    peak = np.abs(mix).max()
    if peak > 0.99:
        mix = mix / peak * 0.97

    out.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out), mix, sr_v)
    return out


# ---------------- 主流程 ----------------
def run_creation(reference_audio: str, lyrics: str, style: str,
                 seed: int = 831001, candidates: int = 1, cfg_scale: float = 1.0,
                 cot: str = "full",
                 prompt_max_sec: int = 25, vocal_gain_db: float = 0.0, acc_gain_db: float = 0.0,
                 channel: str = DEFAULT_CHANNEL,
                 engine: str = "yue2",
                 diffusion_steps: int = DEFAULT_SEEDVC_DIFFUSION_STEPS,
                 cfg_rate: float = DEFAULT_SEEDVC_CFG_RATE,
                 auto_f0_adjust: bool = DEFAULT_SEEDVC_AUTO_F0_ADJUST,
                 semi_tone_shift: int = DEFAULT_SEMI_TONE_SHIFT,
                 apply_pitch_correct: bool = DEFAULT_PITCH_CORRECT,
                 pitch_smooth_sec: float = DEFAULT_PITCH_CORRECT_SMOOTH_SEC,
                 pitch_thresh_st: float = DEFAULT_PITCH_CORRECT_THRESH_ST,
                 pitch_max_st: float = DEFAULT_PITCH_CORRECT_MAX_ST,
                 use_master_mix: bool = True,
                 reference_text: str = None, auto_transcribe: bool = True,
                 cfg_value: float = DEFAULT_CFG_VALUE,
                 inference_timesteps: int = DEFAULT_INFERENCE_TIMESTEPS,
                 denoise: bool = False,
                 # 20261012 回退后补回: 母版阶段显存安全参数(默认 14GiB + 关闭 AR offload,
                 # 避免服务端默认 23.5GiB 在 16GB 卡上卡死/失败; 调用方可不传)
                 yue2_memory_budget_gib: float = 14.0, yue2_offload_ar: bool = False,
                 work_dir: Path = None, progress_cb=None,
                 clean_lyrics: bool = True, lyrics_replace_map: dict = None) -> dict:
    """执行完整创作流程, 返回结果文件路径字典

    合成通道(channel):
      - "voice_convert"(默认, 推荐): 恢复「母版人声 -> 音色转换」主通道
          YuE2 生成母版 -> htdemucs 分离 vocal+acc ->
          T8 内置 Seed-VC 将母版 vocal 换成参考音色(保留母版旋律唱腔) ->
          mix_audio 混音(换声 vocal + 母版伴奏)
      - "voxcpm"(备用): VoxCPM2 零样本歌声克隆
          YuE2 生成母版 -> htdemucs 分离伴奏 ->
          voxcpm_worker(venv_voxcpm) 用歌词+参考音色直接合成干声(48kHz) ->
          mix_audio 混音
        注意: VoxCPM 为纯 TTS 模型, 无旋律条件输入, 合成结果偏朗读腔。
    """
    channel = str(channel or DEFAULT_CHANNEL).lower()
    if channel not in ("voice_convert", "voxcpm"):
        raise RuntimeError(f"未知合成通道: {channel} (可选 voice_convert/voxcpm)")
    engine = str(engine or "yue2").lower()
    if engine not in ("yue2", "songbloom"):
        raise RuntimeError(f"未知母版引擎: {engine} (可选 yue2/songbloom)")

    # 20261028 歌词发音友好预处理(可开关): 高置信同音替换 + 无法注音字/多音字提示
    if clean_lyrics and lyrics:
        lyrics, _lyrics_notes = _sanitize_lyrics_for_pronunciation(lyrics, lyrics_replace_map)
        for _note in _lyrics_notes:
            print(f"[lyrics] {_note}", flush=True)

    if progress_cb:
        progress_cb("初始化", 0, "准备创作环境")
    work_dir = Path(work_dir) if work_dir else PROJECT_OUTPUTS / time.strftime("%Y%m%d-%H%M%S")
    session = work_dir / "session"
    session.mkdir(parents=True, exist_ok=True)
    ref_copy = session / "reference.wav"

    # 参考音色准备: 复制/转码并限制长度
    import librosa
    import soundfile as sf
    y, sr = librosa.load(reference_audio, sr=None, mono=True)
    max_len = prompt_max_sec * sr
    if len(y) > max_len:
        y = y[:max_len]
    sf.write(str(ref_copy), y, sr)

    # 1) 母版生成 (engine: yue2=YuE2 服务 / songbloom=SongBloom 本地推理)
    if engine == "songbloom":
        if progress_cb:
            progress_cb("SongBloom 生成母版", 10, "正在用参考音色+歌词生成母版歌曲(最长150s)")
        lyrics_sb = lyrics_to_songbloom(lyrics)
        if not lyrics_sb:
            raise RuntimeError("歌词为空, 无法转换为 SongBloom 格式")
        if progress_cb:
            progress_cb("SongBloom 生成母版", 12, f"歌词结构: {lyrics_sb[:80]}...")
        master_audio = session / "master_songbloom.flac"
        songbloom_synthesize(lyrics_sb, ref_copy, master_audio)
        if progress_cb:
            progress_cb("SongBloom 生成母版", 35, f"母版生成完成: {master_audio.name}")
    else:
        # YuE2 生成母版
        if progress_cb:
            progress_cb("YuE2 生成母版", 10, "正在按歌词与风格生成母版歌曲(含旋律/伴奏)")
        payload = {
            "style": style,
            "lyrics": lyrics,
            "cot": cot,
            "seed": int(seed),
            "candidates": int(candidates),
            "cfg_scale": float(cfg_scale),
            # 20261012 回退后补回: 显存安全参数随母版任务提交
            "memory_budget_gib": float(yue2_memory_budget_gib),
            "offload_ar": bool(yue2_offload_ar),
        }
        job_id = yue2_submit(payload)
        job = yue2_wait(job_id, progress_cb=lambda st: progress_cb("YuE2 生成母版", 15, st.get("stage") or st.get("status") or "生成中"))
        result = job.get("result") or {}
        candidates_list = result.get("candidates") or []
        if not candidates_list:
            raise RuntimeError(f"YuE2 生成未返回候选: {json.dumps(result, ensure_ascii=False)[:500]}")
        cand = candidates_list[0]
        master_audio = Path(cand.get("audio") or cand.get("result"))
        if not master_audio.is_file():
            # 尝试从 job 目录找 audio.flac
            job_dir = Path(job.get("job_dir") or "").resolve() if job.get("job_dir") else None
            if job_dir and (job_dir / "artifacts" / "song" / "audio.flac").is_file():
                master_audio = job_dir / "artifacts" / "song" / "audio.flac"
            else:
                raise RuntimeError(f"找不到 YuE2 生成音频: {cand}")
        if progress_cb:
            progress_cb("YuE2 生成母版", 35, f"母版生成完成: {master_audio.name}")

    # 2) htdemucs 分离母版 -> vocal + 伴奏(两种通道共用)
    if progress_cb:
        progress_cb("分离母版人声/伴奏", 40, "正在从母版分离人声与伴奏")
    t_sep = session / "sep"
    t_vocal, t_acc = htdemucs_separate(master_audio, t_sep / "vocal_clean.wav", t_sep / "acc_clean.wav", log_tag="master")
    acc_wav = session / "acc.wav"
    shutil.copy2(t_acc, acc_wav)
    if progress_cb:
        progress_cb("分离母版人声/伴奏", 50, "分离完成")

    # 3) 干声合成/换声(按通道)
    if channel == "voice_convert":
        # 主通道: 母版 vocal + 参考音色 -> T8 Seed-VC 换声(保留母版旋律唱腔)
        if progress_cb:
            progress_cb("参考音色换声", 60, "正在用 Seed-VC 将母版人声换成参考音色(保留母版旋律唱腔)")
        # 参考音色先循环扩增再交 Seed-VC(短参考特征更稳定)
        ref_ext = extend_reference(ref_copy, session / "ref_extended.wav")
        vc_dir = session / "vc_output"
        t8_result = t8_voice_convert(
            t_vocal, ref_ext, vc_dir,
            backend="seed-vc",
            diffusion_steps=diffusion_steps, cfg_rate=cfg_rate,
            auto_f0_adjust=auto_f0_adjust, semi_tone_shift=semi_tone_shift,
            vocal_gain_db=vocal_gain_db, accompaniment_gain_db=acc_gain_db,
            log_tag="seedvc",
        )
        if progress_cb:
            progress_cb("参考音色换声", 90, "换声完成(保留母版旋律)")
        converted_vocal = Path(t8_result.get("converted_vocal") or "")
        if not converted_vocal.is_file():
            raise RuntimeError(f"Seed-VC 换声未生成转换人声: {json.dumps(t8_result, ensure_ascii=False)[:500]}")
        vocal_out = session / "voice_convert_vocal.wav"
        shutil.copy2(converted_vocal, vocal_out)
        # 音准校正: 以母版 vocal(t_vocal)旋律为基准, 拉回换声轨逐音符跑调
        # (Seed-VC auto_f0_adjust 仅做全局中位平移, 且分离残留伴奏会污染 f0)
        if apply_pitch_correct:
            if progress_cb:
                progress_cb("音准校正", 92, "以母版人声旋律为基准校正换声轨音高")
            try:
                # 20261028 根治: pyworld(Cython 原生层)遇异常信号会卡死不返回/崩溃,
                # Python 异常无法捕获, 会占死整个 WebUI 事件线程导致"第二次生成报错/无产物"。
                # 改为子进程执行 + 超时看门狗(900s), 任何崩溃/超时/未产出都回退未校正
                # 换声轨并继续, 主流程与 WebUI 永不被音准校正拖死。
                pitch_correct_py = Path(__file__).resolve().parent / "pitch_correct.py"
                subprocess.run(
                    [
                        str(sys.executable), str(pitch_correct_py),
                        "--source", str(vocal_out),
                        "--reference", str(t_vocal),
                        "--output", str(vocal_out),
                        "--smooth-sec", str(pitch_smooth_sec),
                        "--apply-thresh-st", str(pitch_thresh_st),
                        "--max-corr-st", str(pitch_max_st),
                    ],
                    timeout=900, check=True,
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                )
                if not Path(vocal_out).is_file() or Path(vocal_out).stat().st_size <= 0:
                    raise RuntimeError("音准校正未生成输出文件")
            except Exception as exc:
                # 校正失败不中断主流程: 回退为未校正换声轨
                shutil.copy2(converted_vocal, vocal_out)
                print(f"[warn] 音准校正失败/超时, 已回退未校正换声轨: {exc}")
    else:
        # 备用通道: VoxCPM 文本合成(歌词 + 参考音色, 无旋律条件, 偏朗读腔)
        if progress_cb:
            progress_cb("VoxCPM 歌声合成", 60, "正在用歌词+参考音色合成干声(VoxCPM2)")
        vocal_out = session / "voxcpm_vocal.wav"
        voxcpm_synthesize(
            lyrics, ref_copy, vocal_out,
            reference_text=reference_text,
            auto_transcribe=auto_transcribe,
            cfg_value=cfg_value,
            inference_timesteps=inference_timesteps,
            denoise=denoise,
        )
        if progress_cb:
            progress_cb("VoxCPM 歌声合成", 90, "干声合成完成")

    # 4) 混音: 换声/合成干声 + 母版伴奏(采样率不一致自动重采样对齐)
    if progress_cb:
        progress_cb("混音", 95, "正在混合人声与伴奏")
    if not acc_wav.is_file():
        raise RuntimeError(f"母版伴奏缺失: {acc_wav}")
    final_out = work_dir / "final_song.wav"
    if channel == "voice_convert" and use_master_mix:
        # 主通道: 专业母带混音(人声低切+压缩+卷积混响匹配伴奏空间感+响度对齐母版)
        # 解决「干声+伴奏直接相加」的生硬拼接感, 整体听感贴近母版歌曲
        if progress_cb:
            progress_cb("混音", 95, "母带混音(压缩+混响+响度对齐母版)")
        try:
            mix_master.mix_master(
                vocal_out, acc_wav, final_out,
                vocal_gain_db=vocal_gain_db, acc_gain_db=acc_gain_db,
                master_ref=master_audio,
                enable_compressor=True, reverb_wet=0.12, loudness_match=True,
            )
        except Exception as exc:
            # 母带混音失败不中断主流程: 回退简单混音
            print(f"[warn] 母带混音失败, 回退简单混音: {exc}")
            mix_audio(vocal_out, acc_wav, final_out, vocal_gain_db, acc_gain_db)
    else:
        mix_audio(vocal_out, acc_wav, final_out, vocal_gain_db, acc_gain_db)

    # 复制母版/干声/伴奏, 方便对比
    shutil.copy2(master_audio, work_dir / "master_song.flac")
    shutil.copy2(vocal_out, work_dir / f"{channel}_vocal.wav")
    shutil.copy2(acc_wav, work_dir / "accompaniment.wav")

    result_files = {
        "final": str(final_out),
        "master": str(work_dir / "master_song.flac"),
        "vocal": str(work_dir / f"{channel}_vocal.wav"),
        "accompaniment": str(work_dir / "accompaniment.wav"),
        "work_dir": str(work_dir),
        "channel": channel,
        "engine": engine,
        "pitch_corrected": bool(apply_pitch_correct and channel == "voice_convert"),
    }
    if progress_cb:
        progress_cb("完成", 100, "创作完成")
    return result_files


def main():
    parser = argparse.ArgumentParser(description="YuE2 参考音色歌曲创作 (换声主通道 + VoxCPM 备用)")
    parser.add_argument("--reference", required=True, help="参考音色音频路径")
    parser.add_argument("--lyrics", required=True, help="歌词文本")
    parser.add_argument("--style", required=True, help="风格提示词")
    parser.add_argument("--seed", type=int, default=831001)
    parser.add_argument("--candidates", type=int, default=1)
    parser.add_argument("--cfg_scale", type=float, default=1.0)
    parser.add_argument("--cot", default="full")
    parser.add_argument("--prompt_max_sec", type=int, default=25, help="参考音色最大时长(秒)")
    parser.add_argument("--vocal_gain_db", type=float, default=0.0)
    parser.add_argument("--acc_gain_db", type=float, default=0.0)
    parser.add_argument("--channel", default=DEFAULT_CHANNEL, choices=["voice_convert", "voxcpm"],
                        help="合成通道: voice_convert=母版人声->Seed-VC换声(默认,推荐) / voxcpm=VoxCPM文本合成(备用,朗读腔)")
    parser.add_argument("--engine", default="yue2", choices=["yue2", "songbloom"],
                        help="母版引擎: yue2=YuE2服务(默认) / songbloom=SongBloom本地推理(参考音色生成歌曲, 最长150s)")
    parser.add_argument("--diffusion_steps", type=int, default=DEFAULT_SEEDVC_DIFFUSION_STEPS,
                        help="Seed-VC 扩散步数(默认30)")
    parser.add_argument("--cfg_rate", type=float, default=DEFAULT_SEEDVC_CFG_RATE, help="Seed-VC CFG(默认0.7)")
    parser.add_argument("--no_auto_f0_adjust", dest="auto_f0_adjust", action="store_false",
                        help="关闭 Seed-VC 自动音域适配(默认开)")
    parser.add_argument("--auto_f0_adjust", action="store_true", default=True)
    parser.add_argument("--semi_tone_shift", type=int, default=DEFAULT_SEMI_TONE_SHIFT,
                        help="Seed-VC 半音移调(默认0)")
    parser.add_argument("--reference_text", default=None, help="参考音频转录文本(可选, VoxCPM 通道用)")
    parser.add_argument("--no_auto_transcribe", dest="auto_transcribe", action="store_false",
                        help="关闭参考音频自动转录(SenseVoice, 默认开, VoxCPM 通道用)")
    parser.add_argument("--auto_transcribe", action="store_true", default=True)
    parser.add_argument("--cfg_value", type=float, default=DEFAULT_CFG_VALUE, help="VoxCPM CFG(默认2.0)")
    parser.add_argument("--inference_timesteps", type=int, default=DEFAULT_INFERENCE_TIMESTEPS,
                        help="VoxCPM 扩散步数(默认10)")
    parser.add_argument("--denoise", action="store_true", default=False, help="启用外部降噪增强(默认关)")
    parser.add_argument("--work_dir", default=None)
    args = parser.parse_args()

    files = run_creation(
        args.reference, args.lyrics, args.style,
        seed=args.seed, candidates=args.candidates, cfg_scale=args.cfg_scale, cot=args.cot,
        prompt_max_sec=args.prompt_max_sec,
        vocal_gain_db=args.vocal_gain_db, acc_gain_db=args.acc_gain_db,
        channel=args.channel,
        engine=args.engine,
        diffusion_steps=args.diffusion_steps, cfg_rate=args.cfg_rate,
        auto_f0_adjust=args.auto_f0_adjust, semi_tone_shift=args.semi_tone_shift,
        reference_text=args.reference_text,
        auto_transcribe=args.auto_transcribe,
        cfg_value=args.cfg_value,
        inference_timesteps=args.inference_timesteps,
        denoise=args.denoise,
        work_dir=Path(args.work_dir) if args.work_dir else None,
        progress_cb=lambda stage, pct, msg: print(f"[{pct:3.0f}%] {stage}: {msg}", flush=True),
    )
    print("\n=== 创作完成 ===")
    for k, v in files.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
