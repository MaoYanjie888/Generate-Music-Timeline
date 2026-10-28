"""MVGR 歌词时间戳工具 — 精简版 WebUI。

功能：上传歌曲音频 + 歌词文件（.lrc/.txt）→ Qwen3-ASR + ForcedAligner 识别 word 级时间戳
→ DTW 字符相似度对齐到歌词行 → 输出每句歌词的起止时间，
可导出 .lrc / .csv 或一键复制。

歌词文本 100% 来自用户上传的歌词文件；Qwen3-ASR 仅提供时间轴。
"""

from __future__ import annotations

import argparse
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any, Optional

import gradio as gr

from src.utils.logger import get_logger
from src.utils.paths import resource_path
from src.webui import session as session_store

logger = get_logger(__name__)

CONFIG_PATH = resource_path("configs/default.yaml")


# ================================================================
# 歌词解析
# ================================================================

def _parse_lyrics_file(lyrics_path: str) -> list[str]:
    """解析歌词文件（.lrc / .txt）为纯文本歌词行列表。

    - .lrc：去掉 [mm:ss.xx] 时间戳标签，提取纯文本行；
    - .txt：按行拆分纯文本。

    Args:
        lyrics_path: 歌词文件路径。

    Returns:
        歌词行列表（已去除空行与元数据标签）。

    Raises:
        FileNotFoundError: 文件不存在。
        ValueError: 解析后无有效歌词行。
    """
    path = Path(lyrics_path)
    if not path.exists():
        raise FileNotFoundError(f"歌词文件不存在: {path}")

    raw = path.read_text(encoding="utf-8", errors="ignore")
    if not raw.strip():
        try:
            raw = path.read_text(encoding="gbk", errors="ignore")
        except Exception:
            pass

    lines: list[str] = []
    for line in raw.splitlines():
        text = line.strip().lstrip("\ufeff")
        if not text:
            continue
        # 去掉 LRC 时间戳标签 [mm:ss.xx]（可多个）
        text = re.sub(r"\[\d{1,2}:\d{2}(?:[.:]\d{1,3})?\]", "", text).strip()
        if not text:
            continue
        # 跳过元数据标签行（如 [ti:xxx] [ar:xxx] [al:xxx] [by:xxx] [offset:xxx]）
        if text.startswith("[") and text.endswith("]"):
            continue
        lines.append(text)

    if not lines:
        raise ValueError("歌词文件解析后无有效歌词行")
    return lines


# ================================================================
# DTW 歌词对齐
# ================================================================

def _char_similarity(a: str, b: str) -> float:
    """基于字符重叠的相似度（0-1），用于 DTW 距离度量。

    用公共字符数 / 较短串长度，短词与包含它的长句相似度更高，
    使歌词行边界对齐更准确。
    """
    if not a or not b:
        return 0.0
    ca, cb = Counter(a), Counter(b)
    overlap = sum((ca & cb).values())
    return overlap / min(len(a), len(b))


def _is_cjk_char(char: str) -> bool:
    """判断字符是否为 CJK 汉字（与 Qwen3ASRProcessor._is_cjk_char 一致）。

    Qwen3-ForcedAligner 对中文按"逐字切分"（每个汉字一个 word），
    词密度等防护需要据此区分中文逐字段与英文整词段。
    """
    codepoint = ord(char)
    return (
        (0x4E00 <= codepoint <= 0x9FFF)
        or (0x3400 <= codepoint <= 0x4DBF)
        or (0x20000 <= codepoint <= 0x2A6DF)
        or (0x2A700 <= codepoint <= 0x2B73F)
        or (0x2B740 <= codepoint <= 0x2B81F)
        or (0x2B820 <= codepoint <= 0x2CEAF)
        or (0xF900 <= codepoint <= 0xFAFF)
        or (0x2F800 <= codepoint <= 0x2FA1F)
    )


def _pick_line_span(
    idxs: list[int],
    dist_row: list[float],
    words: list[dict[str, Any]],
    min_sim: float,
) -> tuple[int, int]:
    """从某行 DTW 路径的 word 索引中选择行 start / end 索引。

    优先使用"命中歌词"的词（words[j]["hit"]，即该词与用户歌词有字符重叠）：
    - start 取第一个命中词：跳过行首气声、衬词、间奏残留幻觉词；
    - end 取最后一个命中词：避免行尾把间奏/音乐幻觉词吸收进本行。
    行内没有任何命中词（Whisper 转写错字较多）时退回相似度达标词：
    - start 取第一个相似度达标词，end 取最后一个相似度达标词。
    """
    hit_idxs = [j for j in idxs if words[j].get("hit")]
    if hit_idxs:
        return hit_idxs[0], hit_idxs[-1]
    start = idxs[0]
    for j in idxs:
        if 1.0 - dist_row[j] >= min_sim:
            start = j
            break
    end = idxs[-1]
    for j in reversed(idxs):
        if 1.0 - dist_row[j] >= min_sim:
            end = j
            break
    return start, end


def _dtw_align_lines(
    lyrics_lines: list[str],
    words: list[dict[str, Any]],
    min_sim: float = 0.3,
    axis_start: float = 0.0,
) -> list[dict[str, Any]]:
    """用动态时间规整（DTW）把歌词行对齐到 word 时间戳序列。

    距离 = 1 - 字符相似度。回溯得到每行歌词对应的 word 区间，
    取区间首 word 的 start 与末 word 的 end 作为该行时间。

    对齐策略（保证任何歌曲都不漏行）：
    - 匹配质量达标（avg_sim >= min_sim）的行直接采用 DTW 时间；
    - 匹配质量差的行不丢弃，改用相邻达标行的边界做线性插值；
    - 没有任何达标行时按行号均分整个时间轴兜底；
    - 首行 start 优先取第一个"命中歌词"的词（words[j]["hit"]），
      避免前奏噪声词/衬词被纳入，保证开场时间戳落在真实开口处；
    - 行 end 优先取最后一个命中歌词的词，避免吸收间奏/尾奏幻觉词；
    - 开头无达标锚时，插值起点用 axis_start（第一个真实语音段的起点），
      避免第一行时间戳从 0 顺推导致开场严重失真。
    """
    m, n = len(lyrics_lines), len(words)
    if m == 0 or n == 0:
        return []

    dist = [
        [
            1.0
            - _char_similarity(lyrics_lines[i], words[j]["word"])
            + (0.8 if not words[j].get("hit") else 0.0)
            for j in range(n)
        ]
        for i in range(m)
    ]

    INF = float("inf")
    dp = [[INF] * n for _ in range(m)]
    dp[0][0] = dist[0][0]
    for j in range(1, n):
        dp[0][j] = dp[0][j - 1] + dist[0][j]
    for i in range(1, m):
        dp[i][0] = dp[i - 1][0] + dist[i][0]
    for i in range(1, m):
        for j in range(1, n):
            dp[i][j] = dist[i][j] + min(
                dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1]
            )

    line_ranges: dict[int, list[int]] = {k: [] for k in range(m)}
    i, j = m - 1, n - 1
    while i > 0 or j > 0:
        line_ranges[i].append(j)
        if i == 0:
            j -= 1
        elif j == 0:
            i -= 1
        else:
            prev = min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])
            if prev == dp[i - 1][j - 1]:
                i -= 1
                j -= 1
            elif prev == dp[i - 1][j]:
                i -= 1
            else:
                j -= 1
    line_ranges[0].append(0)

    row_info: list[dict[str, Any]] = []
    for k in range(m):
        idxs = sorted(set(line_ranges[k]))
        if not idxs:
            row_info.append({"k": k, "idxs": [], "sim": 0.0, "hit_cnt": 0, "good": False})
            continue
        # 行级达标判定：行内存在至少 1 个"命中歌词"的词即视为达标。
        # 相比"行均相似度"，该判定对 Whisper 转写错字更鲁棒——只要转写中
        # 保留了歌词的任意字，该行就能成为锚点，避免整首歌退化为均分兜底。
        hit_cnt = sum(1 for j in idxs if words[j].get("hit"))
        row_info.append({
            "k": k,
            "idxs": idxs,
            "sim": 0.0,
            "hit_cnt": hit_cnt,
            "good": hit_cnt >= 1,
        })

    anchors: list[dict[str, Any]] = []
    for info in row_info:
        if not info["good"] or not info["idxs"]:
            continue
        idxs = info["idxs"]
        # start/end 优先落在"命中歌词"的词上，跳过衬词/间奏幻觉词
        start_idx, end_idx = _pick_line_span(idxs, dist[info["k"]], words, min_sim)
        anchors.append({
            "k": info["k"],
            "start": float(words[start_idx]["start"]),
            "end": float(words[end_idx]["end"]),
        })

    total_end = float(words[-1]["end"]) if words else 0.0

    # 全局健康行平均时长（good 行），用于插值段的合理行时长
    good_durs = [float(a["end"]) - float(a["start"]) for a in anchors]
    avg_line_dur = (sum(good_durs) / len(good_durs)) if good_durs else 4.0

    if not anchors:
        # 极端兜底：无任何达标行时，均分范围锚定"命中歌词"词的跨度，
        # 避免从 0 秒开局导致前奏被歌词占满；前奏/尾奏空白由此让出。
        span_start = axis_start if axis_start > 0.0 else 0.0
        span_end = total_end
        hit_words = [w for w in words if w.get("hit")]
        if hit_words:
            span_start = float(hit_words[0]["start"])
            span_end = float(hit_words[-1]["end"])
        span = max(span_end - span_start, 1.0)
        return [
            {
                "start": round(span_start + span * k / m, 2),
                "end": round(span_start + span * (k + 1) / m, 2),
                "text": lyrics_lines[k],
            }
            for k in range(m)
        ]

    aligned: list[dict[str, Any]] = []
    k = 0
    while k < m:
        info = row_info[k]
        if info["good"] and info["idxs"]:
            idxs = info["idxs"]
            start_idx, end_idx = _pick_line_span(idxs, dist[k], words, min_sim)
            aligned.append({
                "start": round(float(words[start_idx]["start"]), 2),
                "end": round(float(words[end_idx]["end"]), 2),
                "text": lyrics_lines[k],
            })
            k += 1
            continue

        k2 = k
        while k2 < m and not (row_info[k2]["good"] and row_info[k2]["idxs"]):
            k2 += 1
        # 开头无达标锚时，插值起点取第一个真实语音段起点，避免从 0 顺推
        prev_end = axis_start
        for kk in range(k - 1, -1, -1):
            if row_info[kk]["good"] and row_info[kk]["idxs"]:
                prev_end = float(aligned[kk]["end"])
                break
        next_start = total_end
        if k2 < m:
            idxs2 = row_info[k2]["idxs"]
            s2, _e2 = _pick_line_span(idxs2, dist[k2], words, min_sim)
            next_start = float(words[s2]["start"])
        cnt = k2 - k
        seg_linear = (next_start - prev_end) / cnt
        seg = seg_linear
        if k2 >= m:
            # 尾部无后锚段：按全局平均行时长顺推，避免压缩成 0.6~1s/行
            seg = avg_line_dur
        elif prev_end > 0.0:
            # 中间插值段：均分时长异常（过密或过稀）时改用平均行时长顺推，
            # 但不得越过后锚点 next_start
            if seg_linear < 0.7 * avg_line_dur or seg_linear > 1.4 * avg_line_dur:
                seg = avg_line_dur
            if prev_end + seg * cnt > next_start:
                seg = seg_linear
        for kk in range(k, k2):
            start = prev_end + seg * (kk - k)
            end = prev_end + seg * (kk - k + 1)
            aligned.append({
                "start": round(start, 2),
                "end": round(end, 2),
                "text": lyrics_lines[kk],
            })
        k = k2
    return aligned


# ================================================================
# Qwen3-ASR 转写与对齐
# ================================================================

_LANG_MAP = {
    "zh": "Chinese",
    "yue": "Cantonese",
    "en": "English",
    "ja": "Japanese",
    "ko": "Korean",
    "fr": "French",
    "de": "German",
    "it": "Italian",
    "es": "Spanish",
    "pt": "Portuguese",
    "ru": "Russian",
}


def _to_wav_16k(audio_path: str) -> str:
    """将任意输入音频转为 16kHz 单声道 PCM WAV（Qwen3-ASR 仅接受 WAV/PCM）。

    已是 .wav 时直接复用（模型内部会按 16kHz 重采样）；
    其余格式（mp3/m4a/flac/ogg 等）用 ffmpeg 转换到系统临时目录。
    """
    from tempfile import gettempdir

    src = Path(audio_path)
    if not src.exists():
        raise FileNotFoundError(f"音频文件不存在: {src}")
    if src.suffix.lower() == ".wav":
        return str(src)
    out = Path(gettempdir()) / f"mvgrtool_qwen_{src.stem}_{int(time.time())}.wav"
    cmd = [
        "ffmpeg", "-y", "-i", str(src),
        "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le",
        str(out),
    ]
    import subprocess
    proc = subprocess.run(cmd, capture_output=True, timeout=600)
    if proc.returncode != 0 or not out.exists():
        err_txt = (proc.stderr or b"").decode("utf-8", errors="replace")
        raise RuntimeError(f"ffmpeg 转 WAV 失败: {err_txt[-500:]}")
    return str(out)


def _audio_duration_sec(wav_path: str) -> float:
    """用 ffprobe 获取音频时长（秒）。"""
    import json
    import subprocess

    proc = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "json", wav_path],
        capture_output=True, timeout=120,
    )
    try:
        return float(json.loads(proc.stdout.decode("utf-8", errors="replace"))["format"]["duration"])
    except Exception:
        return 0.0


def _split_audio_segments(wav_path: str, max_dur: float = 280.0) -> list[tuple[float, str]]:
    """长音频按静音段切片，返回 [(offset_seconds, segment_wav_path)]。

    Qwen3-ForcedAligner 单次对齐上限约 5 分钟（~300s 音频），
    超过安全线（默认 280s）时先按静音段切分，各段分别对齐后合并时间轴偏移。
    """
    from tempfile import gettempdir

    duration = _audio_duration_sec(wav_path)
    if duration <= max_dur or duration <= 0.0:
        return [(0.0, wav_path)]

    # 用 silencedetect 找出静音区间
    import re
    import subprocess

    proc = subprocess.run(
        ["ffmpeg", "-i", wav_path, "-af",
         "silencedetect=noise=-35dB:d=0.6", "-f", "null", "-"],
        capture_output=True, timeout=600,
    )
    stderr_txt = (proc.stderr or b"").decode("utf-8", errors="replace")
    silence_ends = [float(m) for m in re.findall(r"silence_end:\s*([\d.]+)", stderr_txt)]
    silence_starts = [float(m) for m in re.findall(r"silence_start:\s*([\d.]+)", stderr_txt)]
    cuts: list[float] = []
    for s, e in zip(silence_starts, silence_ends):
        if s < 1.0 or e > duration - 1.0:
            continue
        mid = (s + e) / 2.0
        if mid > 0.0:
            cuts.append(mid)
    cuts = sorted(cuts)

    bounds = [0.0] + cuts + [duration]
    segs: list[tuple[float, float]] = []
    for a, b in zip(bounds[:-1], bounds[1:]):
        if b - a < 1.0:
            continue
        segs.append((a, b))
    # 静音点不足时均分兜底，保证每段 <= max_dur
    if not cuts or max(b - a for a, b in segs) > max_dur:
        segs = []
        t = 0.0
        while t < duration:
            segs.append((t, min(t + max_dur * 0.9, duration)))
            t = segs[-1][1]

    tmpdir = Path(gettempdir())
    out: list[tuple[float, str]] = []
    for i, (a, b) in enumerate(segs):
        seg_wav = tmpdir / f"mvgrtool_qwen_seg_{int(time.time())}_{i}.wav"
        subprocess.run(
            ["ffmpeg", "-y", "-ss", f"{a:.3f}", "-to", f"{b:.3f}",
             "-i", wav_path, "-c:a", "pcm_s16le", str(seg_wav)],
            capture_output=True, timeout=600,
        )
        if seg_wav.exists():
            out.append((a, str(seg_wav)))
    return out or [(0.0, wav_path)]


def _qwen_transcribe_align(
    audio_path: str,
    model_manager,
    language: str = "",
) -> list[dict[str, Any]]:
    """Qwen3-ASR 转写整句 + ForcedAligner 词级时间戳。

    返回词级时间戳列表：[{"start": s, "end": e, "word": text}, ...]，
    时间单位为秒（绝对时间轴，长音频切片后已叠加偏移）。
    """
    import torch

    asr = model_manager.get_model("qwen-asr")
    aligner = model_manager.get_model("qwen-aligner")
    if asr is None or aligner is None:
        raise RuntimeError("Qwen3-ASR / ForcedAligner 模型未加载")

    asr_processor, asr_model = asr["processor"], asr["model"]
    aligner_processor, aligner_model = aligner["processor"], aligner["model"]

    wav_path = _to_wav_16k(audio_path)
    segs = _split_audio_segments(wav_path)

    # 语言映射：config 中的代码（zh/en/ja/ko...）→ Qwen 官方语言名
    lang_name = _LANG_MAP.get(language.strip().lower(), None) or "auto"
    if lang_name == "auto":
        lang_param = None
    else:
        lang_param = lang_name

    all_words: list[dict[str, Any]] = []
    for offset, seg_wav in segs:
        # Step 1: ASR 转写整句
        try:
            inputs = asr_processor.apply_transcription_request(
                audio=seg_wav,
                language=lang_param,
            )
        except Exception:
            # 语言参数不被接受时回退自动检测
            inputs = asr_processor.apply_transcription_request(audio=seg_wav)
        inputs = inputs.to(asr_model.device, asr_model.dtype)
        output_ids = asr_model.generate(**inputs, max_new_tokens=1024)
        generated_ids = output_ids[:, inputs["input_ids"].shape[1]:]
        parsed = asr_processor.decode(generated_ids, return_format="parsed")[0]
        transcript = str(parsed.get("transcription") or "").strip()
        seg_lang = parsed.get("language") or lang_name or "English"
        if not transcript:
            continue

        # Step 2: ForcedAligner 词级对齐
        aligner_inputs, word_lists = aligner_processor.prepare_forced_aligner_inputs(
            audio=seg_wav,
            transcript=transcript,
            language=seg_lang,
        )
        aligner_inputs = aligner_inputs.to(aligner_model.device, aligner_model.dtype)
        with torch.inference_mode():
            outputs = aligner_model(**aligner_inputs)
        timestamps = aligner_processor.decode_forced_alignment(
            logits=outputs.logits,
            input_ids=aligner_inputs["input_ids"],
            word_lists=word_lists,
            timestamp_token_id=aligner_model.config.timestamp_token_id,
        )[0]

        for item in timestamps:
            text = str(item.get("text", "")).strip()
            if not text:
                continue
            all_words.append({
                "start": float(offset) + float(item.get("start_time", 0.0)),
                "end": float(offset) + float(item.get("end_time", 0.0)),
                "word": text,
            })
    return all_words

def _load_language() -> str:
    """读取默认识别语言（configs/default.yaml -> audio.language）。"""
    try:
        import yaml
        cfg = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
        lang = str(cfg.get("audio", {}).get("language", "zh") or "zh")
        return lang
    except Exception:
        return "zh"


def _align_lyrics_to_audio(
    audio_path: str,
    lyrics_lines: list[str],
) -> list[dict[str, Any]]:
    """将用户歌词行对齐到 Qwen3-ASR word 级时间戳（DTW 字符相似度）。

    Args:
        audio_path: 音频文件路径。
        lyrics_lines: 用户歌词行列表。

    Returns:
        对齐后的歌词列表，每项含 start / end / text。

    Raises:
        RuntimeError: 模型加载或转写失败。
        ValueError: 未获得有效 word 时间戳或对齐失败。
    """
    from src.model_manager import ModelManager

    lang_name = _load_language()
    model_manager = ModelManager()
    model_manager.load_model("qwen-asr")
    model_manager.load_model("qwen-aligner")
    try:
        raw_words = _qwen_transcribe_align(
            str(audio_path),
            model_manager,
            language=lang_name,
        )
    except Exception as e:
        raise RuntimeError(f"Qwen3-ASR 转写失败: {e}")

    if not raw_words:
        raise ValueError("Qwen3-ASR 未返回词级时间戳，无法对齐")

    # Qwen3-ForcedAligner 无 Whisper 式 segment 概念，
    # 按词间隙聚类构造虚拟段，复用下方词密度/命中率/跨度防护过滤。
    def _words_to_segments(word_list, gap_threshold: float = 1.0):
        segs = []
        cur: list[dict[str, Any]] = []
        for w in word_list:
            if cur and float(w["start"]) - float(cur[-1]["end"]) > gap_threshold:
                segs.append({
                    "start": float(cur[0]["start"]),
                    "end": float(cur[-1]["end"]),
                    "words": cur,
                    "no_speech_prob": 0.0,
                })
                cur = []
            cur.append(w)
        if cur:
            segs.append({
                "start": float(cur[0]["start"]),
                "end": float(cur[-1]["end"]),
                "words": cur,
                "no_speech_prob": 0.0,
            })
        return segs

    segments = _words_to_segments(raw_words)

    # 词级"命中歌词"判定：Whisper 词与任意歌词行的最大字符相似度达阈值，
    # 即视为真实歌词词。前奏/间奏/尾奏上的音乐幻觉词、衬词哼唱（"啊/啦"）
    # 通常与歌词文本无关，命中率天然偏低，据此可将"没开口唱歌"的段落剥离。
    def _word_hit(word_text: str) -> bool:
        return (
            max(
                (_char_similarity(word_text, ln) for ln in lyrics_lines),
                default=0.0,
            )
            >= 0.5
        )

    # 第一轮收集：仅剔除明确的无语音段与词密度异常段
    candidate_segments: list[tuple[dict[str, Any], list[dict[str, Any]], float]] = []
    for segment in segments:
        seg_start = float(segment.get("start", 0.0))
        if segment.get("no_speech_prob", 0.0) > 0.6:
            continue
        seg_words = [w for w in segment.get("words", []) if str(w.get("word", "")).strip()]
        if not seg_words:
            continue
        seg_end = float(segment.get("end", seg_start))
        seg_dur = max(seg_end - seg_start, 0.1)

        # 防护一：词密度异常段剔除（音乐/伴奏上的密集幻觉词）。
        # 注意 Qwen3-ForcedAligner 对中文是"逐字切分"（每个汉字一个 word，
        # 见 processor.split_words_for_alignment），正常中文唱歌语速约 4~8 字/秒，
        # 若按 Whisper 时代的"词数/秒>3"一刀切，真实歌词段会被大量误杀，
        # 导致 DTW 退化走插值/均分兜底、时间戳被压成 0.4~0.8s/行。
        # 这里区分 CJK 逐字段与英文整词段：
        # - CJK 主导段：按"字符数/秒"判异常，>10 字/秒才认为是密集幻觉；
        # - 非 CJK 段：保持整词密度 >3 词/秒判异常。
        cjk_chars = sum(
            1 for w in seg_words
            for ch in str(w.get("word", ""))
            if _is_cjk_char(ch)
        )
        total_chars = sum(len(str(w.get("word", ""))) for w in seg_words)
        is_cjk_dom = total_chars > 0 and cjk_chars / total_chars > 0.5
        char_density = total_chars / seg_dur
        word_density = len(seg_words) / seg_dur
        if (is_cjk_dom and char_density > 10.0) or (not is_cjk_dom and word_density > 3.0):
            continue
        candidate_segments.append((segment, seg_words, seg_dur))

    # 无歌词段过滤：段内"命中歌词"的词占比过低（< 25%）→ 判定为前奏/间奏/
    # 尾奏/衬词/音乐幻觉段，整段不进入歌词时间轴。用户歌词就是 ground truth，
    # 该过滤比纯 no_speech_prob 更可靠，能直接解决"间奏被分到歌词时间戳"问题。
    keep_segments: list[tuple[dict[str, Any], list[dict[str, Any]], float]] = []
    for segment, seg_words, seg_dur in candidate_segments:
        hits = sum(1 for w in seg_words if _word_hit(str(w.get("word", "")).strip()))
        if hits / len(seg_words) < 0.25:
            continue
        keep_segments.append((segment, seg_words, seg_dur))

    # 兜底：全部段都被过滤（歌词与转写差异极大或歌词极短）时，
    # 退化为仅靠基础过滤，避免整首歌时间轴为空。
    if not keep_segments:
        keep_segments = candidate_segments

    # 第一个真实歌词段的起点：作为时间轴起点下限，保证开场时间戳不早于开口
    first_voice_start: float | None = (
        float(keep_segments[0][0].get("start", 0.0)) if keep_segments else None
    )

    words: list[dict[str, Any]] = []
    for segment, seg_words, seg_dur in keep_segments:
        seg_start = float(segment.get("start", 0.0))
        seg_end = float(segment.get("end", seg_start))

        # 防护二：word 时间戳跨度异常重建。若整段 word 时间被压缩到 segment
        # 开头（跨度 < 段时长 40%），说明词级对齐退化，直接采用会把多行歌词
        # 压成 0.4~0.8s/行密集堆叠。此处保留词级相对位置做等比缩放铺满整段：
        # 词间先后顺序与相对间隔比例不变，比旧版"按文本长度均匀铺开"更接近真实。
        first_ws = float(seg_words[0].get("start", seg_start))
        last_we = float(seg_words[-1].get("end", seg_end))
        span = max(last_we - first_ws, 0.0)
        if span < seg_dur * 0.4:
            if span > 0.001:
                scale = seg_dur / span
                for w in seg_words:
                    text = str(w.get("word", "")).strip()
                    s = seg_start + (float(w.get("start", first_ws)) - first_ws) * scale
                    e = seg_start + (float(w.get("end", last_we)) - first_ws) * scale
                    words.append({
                        "start": min(max(s, seg_start), seg_end),
                        "end": min(max(e, seg_start), seg_end),
                        "word": text,
                        "hit": _word_hit(text),
                    })
            else:
                # span 为 0（所有词同一时刻）：退化为按文本长度均匀铺开兜底
                total_len = sum(max(len(str(w.get("word", "")).strip()), 1) for w in seg_words)
                cur = seg_start
                for w in seg_words:
                    text = str(w.get("word", "")).strip()
                    dur = seg_dur * max(len(text), 1) / max(total_len, 1)
                    words.append({
                        "start": cur,
                        "end": min(cur + dur, seg_end),
                        "word": text,
                        "hit": _word_hit(text),
                    })
                    cur += dur
            continue

        # 正常段：直接采用 Whisper 提供的 word 时间戳
        for w in seg_words:
            text = str(w.get("word", "")).strip()
            words.append({
                "start": float(w.get("start", 0.0)),
                "end": float(w.get("end", 0.0)),
                "word": text,
                "hit": _word_hit(text),
            })
    if not words:
        raise ValueError("Qwen3-ASR 未返回 word 级时间戳，无法对齐")

    aligned = _dtw_align_lines(lyrics_lines, words, axis_start=first_voice_start or 0.0)
    if not aligned:
        raise ValueError("歌词对齐失败：未能将歌词行匹配到音频时间轴")

    # 精对齐阶段: 歌词直驱 ForcedAligner 逐行修正行边界(消除两层映射误差, 提升到字级精度)
    try:
        aligned = _refine_align_by_lyrics(aligned, str(audio_path), model_manager, lang_name)
    except Exception as e:
        logger.warning(f"歌词精对齐失败, 使用粗对齐结果: {e}")
    finally:
        model_manager.unload_model("qwen-asr")
        model_manager.unload_model("qwen-aligner")
    return aligned


def _cut_wav_segment(wav_path: str, start_sec: float, end_sec: float) -> str | None:
    """从 16k WAV 切出 [start, end] 时间段到临时文件, 返回临时路径(失败返回 None)。"""
    import subprocess
    from tempfile import gettempdir

    duration = _audio_duration_sec(wav_path)
    start_sec = max(0.0, float(start_sec))
    end_sec = min(duration, float(end_sec))
    if end_sec - start_sec < 0.05:
        return None
    out = Path(gettempdir()) / f"mvgrtool_line_{int(time.time() * 1000)}_{int(start_sec * 1000)}.wav"
    proc = subprocess.run(
        ["ffmpeg", "-y", "-ss", f"{start_sec:.3f}", "-to", f"{end_sec:.3f}",
         "-i", wav_path, "-c:a", "pcm_s16le", str(out)],
        capture_output=True, timeout=120,
    )
    if proc.returncode != 0 or not out.exists():
        return None
    return str(out)


def _refine_align_by_lyrics(
    aligned: list[dict[str, Any]],
    audio_path: str,
    model_manager,
    lang_name: str,
) -> list[dict[str, Any]]:
    """歌词直驱 ForcedAligner 逐行精对齐, 消除「ASR转写 -> DTW映射」两层误差。

    升级版(2026-10-28):
    1. 多级窗口(0.8/1.5/2.5s): 小窗口优先保证局部精度, 失败逐级放大兜底, 提升换歌后命中率;
    2. 全局线性补偿: 用命中行最小二乘拟合整轨偏移 y=k*x+b, 修正 DTW 语速估计的系统性误差;
    3. 单调性整理: 逐行钳制保证时间戳不重叠、跨度>=0.1s。

    失败(无时间戳/跨度异常/对齐异常)时回退粗对齐时间, 保证任何歌曲都不崩。
    """
    import torch

    aligner = model_manager.get_model("qwen-aligner")
    if aligner is None:
        return aligned
    aligner_processor, aligner_model = aligner["processor"], aligner["model"]

    wav_path = _to_wav_16k(audio_path)
    raw_lang = lang_name.strip().lower()
    qwen_lang = "Chinese" if raw_lang in ("zh", "yue") else (_LANG_MAP.get(raw_lang) or "English")

    # 多级窗口: 小窗口优先(局部最准), 失败逐级放大
    WINDOWS = (0.8, 1.5, 2.5)

    rows: list[dict[str, Any]] = []
    for row in aligned:
        text = str(row.get("text", "")).strip()
        coarse_s = float(row.get("start", 0.0))
        coarse_e = float(row.get("end", coarse_s + 1.0))
        rows.append({
            "text": text,
            "coarse_s": coarse_s,
            "coarse_e": coarse_e,
            "coarse_span": max(coarse_e - coarse_s, 0.1),
            "fine_s": None,
            "fine_e": None,
        })
    # 按粗对齐起点排序, 保证后续单调整理前提
    rows.sort(key=lambda r: r["coarse_s"])

    # 第一遍: 逐行多级窗口精对齐
    for r in rows:
        if not r["text"]:
            continue
        coarse_s, coarse_e = r["coarse_s"], r["coarse_e"]
        for w in WINDOWS:
            seg_wav = _cut_wav_segment(wav_path, coarse_s - w, coarse_e + w)
            if not seg_wav:
                continue
            try:
                inputs, word_lists = aligner_processor.prepare_forced_aligner_inputs(
                    audio=seg_wav,
                    transcript=r["text"],
                    language=qwen_lang,
                )
                inputs = inputs.to(aligner_model.device, aligner_model.dtype)
                with torch.inference_mode():
                    outputs = aligner_model(**inputs)
                timestamps = aligner_processor.decode_forced_alignment(
                    logits=outputs.logits,
                    input_ids=inputs["input_ids"],
                    word_lists=word_lists,
                    timestamp_token_id=aligner_model.config.timestamp_token_id,
                )[0]
                ts = [it for it in timestamps if str(it.get("text", "")).strip()]
                if ts:
                    cand_s = float(ts[0].get("start_time", 0.0))
                    cand_e = float(ts[-1].get("end_time", 0.0))
                    span = max(cand_e - cand_s, 0.0)
                    # 跨度校验: 太短(全落在同一时刻)或太长(窗口内混入下一句)均回退;
                    # 窗口越大越可能是兜底, 上限适度放宽
                    max_span = max(3.0, r["coarse_span"] * (1.6 + 0.6 * w))
                    if 0.25 <= span <= max_span:
                        r["fine_s"] = coarse_s - w + cand_s
                        r["fine_e"] = coarse_s - w + cand_e
                        break
            except Exception:
                pass
            finally:
                if seg_wav:
                    try:
                        Path(seg_wav).unlink(missing_ok=True)
                    except Exception:
                        pass

    # 第二遍: 全局线性补偿(最小二乘拟合整轨偏移 y = k*x + b)
    ok = [r for r in rows if r["fine_s"] is not None]
    k, b = 0.0, 0.0
    if len(ok) >= 3:
        xs = [(r["coarse_s"] + r["coarse_e"]) / 2.0 for r in ok]
        ys = [(r["fine_s"] - r["coarse_s"] + r["fine_e"] - r["coarse_e"]) / 2.0 for r in ok]
        n = len(xs)
        sx, sy = sum(xs), sum(ys)
        sxx = sum(x * x for x in xs)
        sxy = sum(x * y for x, y in zip(xs, ys))
        denom = n * sxx - sx * sx
        if abs(denom) > 1e-9:
            k = (n * sxy - sx * sy) / denom
            b = (sy - k * sx) / n
            # 缩放系数过激(>±20%)说明拟合不可信, 退化为纯平移
            if abs(k) > 0.2:
                k, b = 0.0, sum(ys) / n
        else:
            b = sum(ys) / n
        for r in rows:
            if r["fine_s"] is None:
                r["fine_s"] = (1.0 + k) * r["coarse_s"] + b
                r["fine_e"] = (1.0 + k) * r["coarse_e"] + b
    elif ok:
        # 命中行过少, 仅做整体平移
        b = sum((r["fine_s"] - r["coarse_s"] + r["fine_e"] - r["coarse_e"]) / 2.0 for r in ok) / len(ok)
        for r in rows:
            if r["fine_s"] is None:
                r["fine_s"] = r["coarse_s"] + b
                r["fine_e"] = r["coarse_e"] + b

    # 第三遍: 单调性整理(逐行钳制, 保证不重叠且跨度>=0.1s)
    prev_end = -1.0
    refined: list[dict[str, Any]] = []
    for r in rows:
        s = r["fine_s"] if r["fine_s"] is not None else r["coarse_s"]
        e = r["fine_e"] if r["fine_e"] is not None else r["coarse_e"]
        s = max(s, prev_end + 0.02)
        e = max(e, s + 0.1)
        prev_end = e
        refined.append({"start": round(s, 3), "end": round(e, 3), "text": r["text"]})

    used_fine = len(ok)
    if used_fine:
        detail = f"，全局补偿 y={k:+.3f}*x{b:+.3f}" if (k or abs(b) > 1e-6) else ""
        logger.info(f"歌词精对齐: {used_fine}/{len(aligned)} 行命中字级时间戳{detail}")
    return refined


# ================================================================
# WebUI
# ================================================================

def build_mvgr_tab():
    """构建「歌词时间戳」Tab 内容（供顶层双子页面嵌入；也可独立运行使用）。"""
    gr.Markdown(
        "# MVGR 歌词时间戳工具\n"
        "上传歌曲音频与歌词文件（.lrc/.txt），自动识别生成每句歌词的起止时间。\n\n"
        "歌词文本以您上传的文件为准，Qwen3-ASR 仅用于提供时间轴。"
    )

    audio_input = gr.File(
        label="上传歌曲音频",
        file_types=[".mp3", ".wav", ".m4a", ".flac", ".ogg"],
    )
    lyrics_file = gr.File(
        label="上传歌词文件（.lrc/.txt）",
        file_types=[".lrc", ".txt"],
    )
    recognize_btn = gr.Button("识别歌词", variant="primary")
    status_text = gr.Textbox(label="状态", interactive=False)
    lyrics_timeline = gr.Dataframe(
        headers=["开始时间", "结束时间", "歌词内容"],
        label="歌词时间轴识别结果",
        interactive=False,
    )
    with gr.Row():
        export_lrc_btn = gr.Button("导出 .lrc", size="sm")
        export_csv_btn = gr.Button("导出 .csv", size="sm")
        copy_lyrics_btn = gr.Button("复制歌词时间轴文本", size="sm")
    lyrics_export_file = gr.File(label="导出文件", interactive=False)
    lyrics_copy_text = gr.Textbox(
        label="歌词时间轴文本（可一键复制）",
        interactive=False,
        lines=6,
    )

    # 歌词文件上传即持久化（异常关闭后可恢复）
    # outputs 必须为空：若回填 lyrics_file，恢复会话时 gradio 会把持久化路径
    # 复制到缓存目录再次触发 change，形成无限循环（组件一直"排队"）
    lyrics_file.change(
        fn=on_lyrics_upload,
        inputs=[lyrics_file],
        outputs=[],
    )

    # 歌词识别（主动模式：按用户上传歌词做时间对齐）
    recognize_btn.click(
        fn=on_recognize_lyrics,
        inputs=[audio_input, lyrics_file],
        outputs=[status_text, lyrics_timeline],
    )

    # 歌词时间轴导出
    export_lrc_btn.click(
        fn=lambda d: on_export_lyrics(d, "lrc"),
        inputs=[lyrics_timeline],
        outputs=[lyrics_export_file],
    )
    export_csv_btn.click(
        fn=lambda d: on_export_lyrics(d, "csv"),
        inputs=[lyrics_timeline],
        outputs=[lyrics_export_file],
    )
    copy_lyrics_btn.click(
        fn=on_copy_lyrics,
        inputs=[lyrics_timeline],
        outputs=[lyrics_copy_text],
    )

    return (audio_input, lyrics_file, recognize_btn, status_text, lyrics_timeline,
            lyrics_export_file, lyrics_copy_text)


def create_app() -> gr.Blocks:
    """创建 Gradio Web 应用（仅歌词时间戳功能，独立运行入口）。"""
    with gr.Blocks(
        title="MVGR 歌词时间戳工具",
    ) as app:
        build_mvgr_tab()
    return app
# ================================================================
# 事件处理
# ================================================================

def on_lyrics_upload(lyrics_path):
    """上传歌词文件后立即持久化到 uploads/，异常关闭后可恢复。

    返回值不回填 lyrics_file（outputs 为空），避免恢复会话时 gradio
    把持久化路径复制到缓存目录再次触发 change，形成无限循环。
    内容与 session 中已有歌词相同时跳过复制。
    """
    if not lyrics_path:
        return None
    sess = session_store.load_session() or {}
    existing = (sess.get("lyrics") or {}).get("path", "")
    if existing and Path(existing).exists():
        try:
            if Path(lyrics_path).read_bytes() == Path(existing).read_bytes():
                return None
        except Exception:
            pass
    persisted = session_store.persist_upload(lyrics_path, "lyrics")
    sess["lyrics"] = {
        "path": persisted,
        "rows": (sess.get("lyrics") or {}).get("rows", []),
    }
    session_store.save_session(sess)
    return None


def on_recognize_lyrics(audio_path, lyrics_path, progress=gr.Progress()):
    """识别歌词：按用户上传歌词做时间对齐（主动模式）。"""
    if not audio_path:
        yield "请先上传歌曲音频", []
        return
    if not lyrics_path:
        yield "请先上传歌词文件（.lrc/.txt）", []
        return
    start_time = time.time()
    try:
        yield (
            "正在加载 Qwen3-ASR 模型并识别歌词，预计等待 1-3 分钟"
            "（首次加载模型更久），请勿关闭页面...",
            [],
        )
        progress(0.1, desc="正在解析歌词文件...")
        lyrics_lines = _parse_lyrics_file(lyrics_path)
        progress(0.2, desc="正在加载 Qwen3-ASR 模型转写音频...")
        yield (
            f"歌词解析完成：共 {len(lyrics_lines)} 行。正在加载 Qwen3-ASR 模型"
            "转写音频（Qwen3-ASR-1.7B 较慢，首次加载需加载模型，请耐心等待）...",
            [],
        )
        aligned = _align_lyrics_to_audio(audio_path, lyrics_lines)
        if not aligned:
            yield (
                "歌词对齐失败：未能将歌词行匹配到音频时间轴，"
                "请检查歌词与音频是否匹配",
                [],
            )
            return
        rows = [
            [
                round(float(l.get("start", 0.0)), 2),
                round(float(l.get("end", 0.0)), 2),
                str(l.get("text", "")),
            ]
            for l in aligned
        ]
        # 持久化歌词时间轴并更新会话快照
        lyrics_path = session_store.persist_lyrics(rows)
        sess = session_store.load_session() or {}
        sess["lyrics"] = {"path": lyrics_path, "rows": rows}
        session_store.save_session(sess)
        elapsed = time.time() - start_time
        yield (
            f"识别完成，共 {len(aligned)} 行，耗时 {elapsed:.1f} 秒"
            "（歌词文本来自您上传的文件）。可在下方「歌词时间轴」查看并导出。",
            rows,
        )
    except FileNotFoundError as e:
        yield f"歌词文件读取失败：{e}", []
    except ValueError as e:
        yield f"歌词解析/对齐失败：{e}", []
    except Exception as e:
        yield f"歌词识别失败：{e}", []


def _lyrics_data_to_rows(lyrics_data):
    """把歌词时间轴数据（gradio Dataframe 或 list）统一转为行列表。"""
    if lyrics_data is None:
        return []
    try:
        import pandas as pd
        if isinstance(lyrics_data, pd.DataFrame):
            return lyrics_data.values.tolist()
    except ImportError:
        pass
    if isinstance(lyrics_data, (list, tuple)):
        return [list(r) for r in lyrics_data]
    return []


def on_export_lyrics(lyrics_data, fmt):
    """导出歌词时间轴为 .lrc 或 .csv 文件。"""
    rows = []
    for row in _lyrics_data_to_rows(lyrics_data):
        if len(row) >= 3:
            rows.append([row[0], row[1], str(row[2])])
    if not rows:
        return None
    out_dir = Path("output")
    out_dir.mkdir(parents=True, exist_ok=True)
    if fmt == "lrc":
        path = out_dir / "lyrics_timeline.lrc"
        lines = []
        for start, _end, text in rows:
            start = float(start)
            mm = int(start // 60)
            ss = start - mm * 60
            lines.append(f"[{mm:02d}:{ss:05.2f}] {text}")
        path.write_text("\n".join(lines), encoding="utf-8")
    else:
        path = out_dir / "lyrics_timeline.csv"
        import csv
        with open(path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            writer.writerow(["开始时间", "结束时间", "歌词内容"])
            for start, end, text in rows:
                writer.writerow([start, end, text])
    return str(path)


def on_copy_lyrics(lyrics_data):
    """生成歌词时间轴文本（LRC 格式），供一键复制。"""
    lines = []
    for row in _lyrics_data_to_rows(lyrics_data):
        if len(row) >= 3:
            start = float(row[0])
            text = str(row[2])
            mm = int(start // 60)
            ss = start - mm * 60
            lines.append(f"[{mm:02d}:{ss:05.2f}] {text}")
    return "\n".join(lines) if lines else "（暂无歌词数据，请先上传音频自动识别）"


def main() -> None:
    """启动 MVGR 歌词时间戳 Web UI。"""
    parser = argparse.ArgumentParser(description="MVGR 歌词时间戳工具 Web UI")
    parser.add_argument("--port", type=int, default=7860, help="Web UI 端口")
    parser.add_argument("--share", action="store_true", help="生成公开链接")
    parser.add_argument("--debug", action="store_true", help="调试模式")

    args = parser.parse_args()

    app = create_app()
    port = args.port
    last_err: Exception | None = None
    for _ in range(10):
        try:
            app.queue(max_size=8).launch(
                server_port=port,
                share=args.share,
                debug=args.debug,
                server_name="localhost",
                theme=gr.themes.Soft(),
            )
            return
        except OSError as e:
            last_err = e
            print(f"[MVGR] 端口 {port} 被占用，自动切换 {port + 1} ...")
            port += 1
    print(f"[MVGR] 启动失败：{last_err}")
    raise last_err


if __name__ == "__main__":
    main()
