#!/usr/bin/env python3
"""ACE-Step Cover 音色跟随 Tab（环境隔离整合）

设计要点：
  - 本模块只依赖 requests（venv_timeline 已自带），通过本机 HTTP 调用
    ACE-Step 独立 REST API Server（默认 127.0.0.1:8001）。
  - 不 import 任何 ACE-Step 模块、不向现有 venv 安装依赖、不触碰母版/分离/
    换声/混伴奏/时间戳链路，也不读写 n_steps/cfg 等未完工改动。
  - 所有数据只在本机回环（127.0.0.1）传输，不出本机。
  - API Server 需单独启动：见项目根目录 start_acestep_api.bat。

时长修复（2026-11-18）：
  ACE-Step 的 task_type=cover 属于 DIRECT_CONDITIONING_TASKS，服务端会把
  audio_duration 强制置 None 并锁定成品时长=参考音频长度（见 inference.py /
  generate_music.py / padding_utils.py），因此 cover 天生无法唱完长歌词。
  本模块新增「音色演唱（时长可控）」模式：task_type=text2music + 参考音频作为
  reference_audio(ref_audio) 音色 conditioning + 显式 audio_duration，时长不再
  被锁死，可按歌词长度生成足够长的成品；原 cover 模式保留以兼容旧用法。
"""
import json
import os
import re
import time
from pathlib import Path

import gradio as gr

try:
    import requests
except ImportError:
    requests = None  # 在界面里给出明确提示

API_BASE = os.environ.get("ACESTEP_API_BASE", "http://127.0.0.1:8001").rstrip("/")
COVER_OUT_DIR = Path(__file__).resolve().parent.parent / "outputs" / "acestep_cover"

_STATUS_TEXT = {0: "进行中/排队", 1: "成功", 2: "失败"}
MODE_CONTROLLED = "音色演唱（时长可控·长歌词推荐）"
MODE_COVER = "原曲翻唱 Cover（时长=参考音频）"


def _health():
    """返回 (是否在线, 健康信息文本)。"""
    if requests is None:
        return False, "缺少 requests 库，请在 venv_timeline 中安装。"
    try:
        r = requests.get(API_BASE + "/health", timeout=5)
        d = r.json().get("data", {})
        online = r.status_code == 200 and d.get("status") == "ok"
        model = d.get("loaded_model") or "-"
        llm = d.get("loaded_lm_model") or "未加载"
        return online, f"在线 · DiT={model} · LM={llm}"
    except Exception as e:
        return False, f"未连接：{API_BASE}（{type(e).__name__}）。请先运行 start_acestep_api.bat"


def check_service():
    ok, info = _health()
    return ("✅ API Server " if ok else "⚠️ API Server ") + info


def _parse_result(raw):
    """query_result 的 result 字段可能是 JSON 字符串或对象。"""
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except Exception:
            return None
    if isinstance(raw, list):
        return raw[0] if raw else None
    if isinstance(raw, dict):
        return raw
    return None


def _download(rel_path: str) -> str:
    """把 /v1/audio?path=... 相对地址下载到本地 outputs/acestep_cover/。"""
    from urllib.parse import quote, urlparse, parse_qs

    COVER_OUT_DIR.mkdir(parents=True, exist_ok=True)
    if rel_path.startswith("http://") or rel_path.startswith("https://"):
        url = rel_path
    else:
        url = API_BASE + rel_path
    # 从 path 查询参数里取出原始文件名
    qs = parse_qs(urlparse(url).query)
    src = qs.get("path", [""])[0]
    name = os.path.basename(src.replace("\\", "/")) or "cover_output.wav"
    stem, _ = os.path.splitext(name)
    ts = time.strftime("%Y%m%d-%H%M%S")
    dest = COVER_OUT_DIR / f"{stem}_{ts}.wav"
    r = requests.get(url, timeout=120)
    r.raise_for_status()
    with open(dest, "wb") as fh:
        fh.write(r.content)
    return str(dest)


def _wav_duration_seconds(path: str):
    """从 WAV 头解析时长（秒，float）；兼容 PCM 与 32-bit float。失败返回 None。"""
    import struct
    try:
        with open(path, "rb") as fh:
            header = fh.read(44)
        if header[:4] != b"RIFF" or header[8:12] != b"WAVE":
            return None
        channels = struct.unpack_from("<H", header, 22)[0]
        rate = struct.unpack_from("<I", header, 24)[0]
        bits = struct.unpack_from("<H", header, 34)[0]
        with open(path, "rb") as fh:
            fh.seek(12)
            data_size = None
            while True:
                chunk = fh.read(8)
                if len(chunk) < 8:
                    break
                cid, size = chunk[:4], struct.unpack("<I", chunk[4:])[0]
                if cid == b"data":
                    data_size = size
                    break
                fh.seek(size + (size % 2), 1)
        if not data_size or not rate or not bits:
            return None
        return data_size / (rate * channels * (bits / 8))
    except Exception:
        return None


def _wav_duration(path: str) -> str:
    """给 UI 展示用：时长文本。"""
    d = _wav_duration_seconds(path)
    return f"{d:.1f}s" if d is not None else "未知"


def infer_duration_from_lyrics(lyrics: str, char_per_sec: float = 4.0, tail_buffer: float = 6.0) -> int:
    """按歌词可唱字数估算所需时长（秒）。

    中文流行演唱经验值约 3~5 字/秒，取保守的 4 字/秒并预留尾部 buffer，
    再 clamp 到 API 允许的 [10, 600]。长歌词据此可得到足够时长，避免唱不完。
    仅统计汉字/日文假名/韩文/英文字词，忽略 [verse]/[chorus] 等结构标签与标点。
    """
    if not lyrics:
        return 30
    # 去掉结构标签行，如 [verse] / [chorus]
    lines = [ln for ln in lyrics.splitlines() if not re.match(r"^\s*\[[^\]]*\]\s*$", ln)]
    text = "\n".join(lines)
    han = len(re.findall(r"[\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]", text))
    words = len(re.findall(r"[A-Za-z]+", text))
    units = han + words
    sec = units / char_per_sec + tail_buffer
    return int(max(10, min(600, round(sec))))


# ---------------------------------------------------------------------------
# 参数 JSON 整段粘贴（可选入口）
# 规范字段来自《ACE-Step 1.5 歌词与风格提示词规范》：caption/lyrics/
# vocal_language/audio_duration/bpm/keyscale/timesignature/thinking 等。
# 解析后只更新已有可见控件；其余高级字段存入 adv_json 交给 do_cover 合并。
# ---------------------------------------------------------------------------
_ADV_ALLOWED = {
    "bpm", "key_scale", "time_signature", "thinking", "use_format",
    "guidance_scale", "shift", "audio_format", "model",
}
_ADV_ALIAS = {
    "keyscale": "key_scale", "key_scale": "key_scale",
    "timesignature": "time_signature", "time_signature": "time_signature",
    "bpm": "bpm", "thinking": "thinking", "use_format": "use_format",
    "guidance_scale": "guidance_scale", "shift": "shift",
    "audio_format": "audio_format", "model": "model",
}
_JSON_DIRECT_KEYS = {
    "task_type", "prompt", "caption", "lyrics", "audio_duration", "duration",
    "seed", "inference_steps", "steps", "reference_audio", "ref_audio", "src_audio",
}


def _safe_adv(text):
    """把高级参数 JSON 文本转成服务端可解析的 multipart 字符串字典。"""
    if not text:
        return {}
    try:
        d = json.loads(text)
    except Exception:
        return {}
    if not isinstance(d, dict):
        return {}
    out = {}
    for k, v in d.items():
        if k not in _ADV_ALLOWED or v is None:
            continue
        if isinstance(v, bool):
            out[k] = "true" if v else "false"
        else:
            out[k] = str(v)
    return out


def parse_params_json(json_text):
    """解析整段参数 JSON，返回供多个控件 gr.update 的元组。

    返回顺序：(prompt, lyrics, vocal_language, duration, auto_duration,
    mode, adv_json, status_text)。任一步失败返回全 gr.update() 与错误提示，
    不影响原手动输入。
    """
    empty = tuple(gr.update() for _ in range(7))
    if not json_text or not json_text.strip():
        return empty + ("提示：粘贴参数 JSON 后点「解析填入」。",)
    try:
        d = json.loads(json_text)
    except Exception as e:
        return empty + (f"❌ JSON 解析失败：{e}",)
    if not isinstance(d, dict):
        return empty + ("❌ JSON 顶层必须是对象 {...}。",)

    prompt = d.get("caption", d.get("prompt", ""))
    lyrics = d.get("lyrics", "")
    lang = d.get("vocal_language", "zh")
    raw_dur = d.get("audio_duration", d.get("duration", 30))
    try:
        dur = int(max(10, min(600, round(float(raw_dur)))))
    except Exception:
        dur = 30
    task_type = str(d.get("task_type", "text2music")).lower()
    mode = MODE_COVER if task_type == "cover" else MODE_CONTROLLED

    adv = {}
    for k, v in d.items():
        if k in _JSON_DIRECT_KEYS:
            continue
        mapped = _ADV_ALIAS.get(k)
        if mapped and mapped in _ADV_ALLOWED:
            adv[mapped] = v
    adv_text = json.dumps(adv, ensure_ascii=False) if adv else "{}"

    n_ly = len([ln for ln in str(lyrics).splitlines() if ln.strip()])
    status = (
        f"✅ 已填入：caption {len(str(prompt))} 字符 · 歌词 {n_ly} 行 · "
        f"时长 {dur}s · 语言 {lang} · 模式 {mode}\n"
        f"高级字段（{len(adv)}）：{adv_text}"
    )
    return (
        gr.update(value=str(prompt)),
        gr.update(value=str(lyrics)),
        gr.update(value=str(lang) if lang in ("zh", "en", "ja", "ko", "es", "fr") else "zh"),
        gr.update(value=dur),
        gr.update(value=False),  # JSON 显式时长优先，关闭自动估算
        gr.update(value=mode),
        gr.update(value=adv_text),
        status,
    )


# ---------------------------------------------------------------------------
# 通用 API 封装（LoRA 选择 + 训练入口）
# ---------------------------------------------------------------------------
def _api_post(path, payload=None, timeout=60):
    if requests is None:
        return None, "缺少 requests 库。"
    try:
        r = requests.post(API_BASE + path, json=payload or {}, timeout=timeout)
        jd = r.json()
        if jd.get("code") not in (200, 0):
            return None, f"失败：{jd.get('error') or jd}"
        return jd.get("data"), None
    except Exception as e:
        return None, f"异常：{type(e).__name__}: {e}"


def _api_get(path, timeout=15):
    if requests is None:
        return None, "缺少 requests 库。"
    try:
        r = requests.get(API_BASE + path, timeout=timeout)
        jd = r.json()
        if jd.get("code") not in (200, 0):
            return None, f"失败：{jd.get('error') or jd}"
        return jd.get("data"), None
    except Exception as e:
        return None, f"异常：{type(e).__name__}: {e}"


def lora_status():
    d, err = _api_get("/v1/lora/status")
    if err:
        return "❌ " + err
    return (
        f"LoRA 已加载：{d.get('lora_loaded')} · 启用：{d.get('use_lora')} · "
        f"scale：{d.get('lora_scale')}\n"
        f"适配器：{d.get('adapters') or []} · 当前：{d.get('active_adapter') or '-'}"
    )


def lora_load(lora_path, adapter_name=""):
    path = (lora_path or "").strip()
    if not path:
        return "❌ 请填写或选择 LoRA 权重路径（adapter 目录或 LoKr/LyCORIS safetensors 文件）。"
    payload = {"lora_path": path}
    if adapter_name and str(adapter_name).strip():
        payload["adapter_name"] = str(adapter_name).strip()
    data, err = _api_post("/v1/lora/load", payload)
    if err:
        return "❌ " + err
    return "✅ 已加载：\n" + str(data)[:500]


def lora_unload():
    data, err = _api_post("/v1/lora/unload", {})
    return ("✅ 已卸载。" if not err else "❌ " + err) + (("" if not err else "") )


def lora_toggle(use_lora):
    data, err = _api_post("/v1/lora/toggle", {"use_lora": bool(use_lora)})
    return ("✅ 已" + ("启用" if use_lora else "禁用") + " LoRA。" if not err else "❌ " + err)


def lora_scale(scale):
    data, err = _api_post("/v1/lora/scale", {"scale": float(scale)})
    return ("✅ scale 已设为 " + str(scale) if not err else "❌ " + err)


def list_lora_paths():
    """扫描常见目录，返回可加载的 LoRA 目录/safetensors 路径列表。"""
    import glob
    roots = []
    ace_root = os.environ.get("ACE_ROOT", r"F:\ACE-Step-1.5")
    proj = str(COVER_OUT_DIR.parent.parent)
    for base in (ace_root, proj):
        for sub in ("lora_output", "lokr_output", "loras", "lora", "outputs/lora"):
            roots.append(os.path.join(base, sub))
    found = []
    for root in roots:
        if not os.path.isdir(root):
            continue
        for p in glob.glob(os.path.join(root, "**", "*.safetensors"), recursive=True):
            found.append(p)
        for p in glob.glob(os.path.join(root, "**", "adapter_config.json"), recursive=True):
            found.append(os.path.dirname(p))
    found = sorted(set(found))
    choices = found[:200]
    info = ("可加载权重 " + str(len(found)) + " 个。\n" +
            ("\n".join(found[:20]) if found else "未发现权重；训练完成后再刷新，或手动粘贴路径。"))
    return gr.update(choices=choices), info


def dataset_scan(audio_dir, dataset_name, custom_tag, tag_position, all_instrumental):
    audio_dir = (audio_dir or "").strip()
    if not audio_dir or not os.path.isdir(audio_dir):
        return "❌ 请填写存在的音频数据集目录（内含音频 + 同名 .lyrics.txt / .json 标注）。"
    payload = {
        "audio_dir": audio_dir,
        "dataset_name": (dataset_name or "my_lora_dataset").strip(),
        "custom_tag": (custom_tag or "").strip(),
        "tag_position": tag_position or "replace",
        "all_instrumental": bool(all_instrumental),
    }
    data, err = _api_post("/v1/dataset/scan", payload, timeout=120)
    if err:
        return "❌ " + err
    samples = data.get("samples") if isinstance(data, dict) else None
    if isinstance(samples, list):
        return f"✅ 扫描到 {len(samples)} 条样本。\n" + str(data)[:600]
    return "✅ 扫描完成：\n" + str(data)[:600]


def dataset_preprocess(output_dir, skip_existing=True):
    output_dir = (output_dir or "").strip()
    if not output_dir:
        return "❌ 请填写张量输出目录。"
    data, err = _api_post(
        "/v1/dataset/preprocess",
        {"output_dir": output_dir, "skip_existing": bool(skip_existing)},
        timeout=600,
    )
    if err:
        return "❌ " + err
    return "✅ 预处理完成：\n" + str(data)[:600]


def training_status():
    d, err = _api_get("/v1/training/status")
    if err:
        return "❌ " + err
    return (
        f"状态：{d.get('status')} · 训练中：{d.get('is_training')}\n"
        f"epoch {d.get('current_epoch')} · step {d.get('current_step')} · "
        f"loss {d.get('current_loss')} · 速度 {d.get('steps_per_second')} step/s\n"
        f"预计剩余 {d.get('estimated_time_remaining')}s · tensor_dir：{d.get('tensor_dir') or '-'}"
        + (f"\n错误：{d.get('error')}" if d.get("error") else "")
    )


def training_start(tensor_dir, kind, rank, alpha, epochs, batch, accum, lr, shift, save_every, out_dir, seed, fp8, grad_ckpt):
    tensor_dir = (tensor_dir or "").strip()
    if not tensor_dir or not os.path.isdir(tensor_dir):
        return "❌ 请填写存在的张量目录（先完成数据集预处理）。"
    out_dir = (out_dir or "").strip() or (r"F:\ACE-Step-1.5\lora_output" if kind == "LoRA" else r"F:\ACE-Step-1.5\lokr_output")
    common = {
        "tensor_dir": tensor_dir,
        "learning_rate": float(lr),
        "train_epochs": int(epochs),
        "train_batch_size": int(batch),
        "gradient_accumulation": int(accum),
        "save_every_n_epochs": int(save_every),
        "training_shift": float(shift),
        "training_seed": int(seed),
        "gradient_checkpointing": bool(grad_ckpt),
    }
    if kind == "LoRA":
        common.update({
            "lora_rank": int(rank), "lora_alpha": int(alpha),
            "use_fp8": bool(fp8), "lora_output_dir": out_dir,
        })
        path = "/v1/training/start"
    else:
        common.update({
            "lokr_linear_dim": int(rank), "lokr_linear_alpha": int(alpha),
            "output_dir": out_dir,
        })
        path = "/v1/training/start_lokr"
    data, err = _api_post(path, common, timeout=120)
    if err:
        return "❌ " + err
    return f"✅ 已提交 {kind} 训练任务：\n" + str(data)[:600]


def training_stop():
    data, err = _api_post("/v1/training/stop", {}, timeout=60)
    return ("✅ 已请求停止训练。" if not err else "❌ " + err)


def training_export(export_path, lora_output_dir):
    export_path = (export_path or "").strip()
    lora_output_dir = (lora_output_dir or "").strip()
    if not export_path or not lora_output_dir:
        return "❌ 请填写导出路径与训练输出目录。"
    data, err = _api_post(
        "/v1/training/export",
        {"export_path": export_path, "lora_output_dir": lora_output_dir},
        timeout=180,
    )
    if err:
        return "❌ " + err
    return "✅ 导出完成：\n" + str(data)[:600]


def do_cover(
    src_audio,
    prompt,
    lyrics,
    vocal_language,
    audio_duration,
    cover_strength,
    inference_steps,
    use_random_seed,
    seed,
    mode=MODE_CONTROLLED,
    auto_duration=True,
    adv_json="",
    progress=gr.Progress(track_tqdm=False),
):
    """上传参考音频并提交任务，轮询完成后下载成品。

    可控模式(text2music+ref_audio)：audio_duration 生效；可选按歌词自动估算时长。
    cover 模式：保持旧行为，成品时长=参考音频时长。
    """
    if requests is None:
        return None, None, "错误：缺少 requests 库。"
    ok, info = _health()
    if not ok:
        return None, None, "错误：API Server 未就绪。" + info
    if not src_audio:
        return None, None, "错误：请先上传参考音频。"

    controlled = (mode != MODE_COVER)
    est = infer_duration_from_lyrics(lyrics)
    if controlled:
        final_duration = max(int(audio_duration), est) if auto_duration else int(audio_duration)
        # 显式 audio_duration：text2music 不在 DIRECT_CONDITIONING_TASKS 中，不会被服务端置空
        data = {
            "task_type": "text2music",
            "prompt": prompt or "",
            "lyrics": lyrics or "",
            "vocal_language": vocal_language or "zh",
            "audio_duration": str(final_duration),
            "audio_format": "wav",
            "inference_steps": str(int(inference_steps)),
            "use_random_seed": "true" if use_random_seed else "false",
            "seed": str(int(seed)) if seed is not None else "-1",
            "batch_size": "1",
        }
        file_field = "ref_audio"  # multipart 映射 reference_audio_path（音色 conditioning）
        note = f"音色演唱模式 · 歌词估算 {est}s · 实际时长 {final_duration}s"
    else:
        final_duration = int(audio_duration)
        # cover 属 DIRECT_CONDITIONING_TASKS：服务端忽略 audio_duration，成品时长=参考音频
        data = {
            "task_type": "cover",
            "prompt": prompt or "",
            "lyrics": lyrics or "",
            "vocal_language": vocal_language or "zh",
            "audio_duration": str(final_duration),
            "audio_format": "wav",
            "audio_cover_strength": str(round(float(cover_strength), 2)),
            "inference_steps": str(int(inference_steps)),
            "use_random_seed": "true" if use_random_seed else "false",
            "seed": str(int(seed)) if seed is not None else "-1",
            "batch_size": "1",
        }
        file_field = "src_audio"
        note = f"原曲翻唱模式 · 成品时长=参考音频时长（时长滑块不生效）"

    # 合并 JSON 入口的高级字段（bpm/key_scale/time_signature/thinking 等）。
    # _safe_adv 不含 prompt/lyrics/audio_duration/task_type，故不覆盖主参数与时长。
    adv = _safe_adv(adv_json)
    if adv:
        data.update(adv)
        note += " · 高级字段:" + ",".join(adv.keys())

    try:
        progress(0.05, desc="提交任务…")
        with open(src_audio, "rb") as fh:
            files = {file_field: (os.path.basename(src_audio), fh, "audio/wav")}
            resp = requests.post(API_BASE + "/release_task", data=data, files=files, timeout=120)
        resp.raise_for_status()
        jd = resp.json()
        if jd.get("code") not in (200, 0):
            return None, None, f"提交失败：{jd.get('error') or jd}"
        task_id = jd["data"]["task_id"]
    except Exception as e:
        return None, None, f"提交异常：{type(e).__name__}: {e}"

    t0 = time.time()
    last_stage = ""
    try:
        while True:
            elapsed = int(time.time() - t0)
            if elapsed > 900:
                return None, None, f"超时（task_id={task_id}），服务端仍在运行，请稍后用日志排查。"
            q = requests.post(
                API_BASE + "/query_result",
                json={"task_id_list": [task_id]},
                timeout=30,
            ).json()
            items = q.get("data") or []
            item = None
            if isinstance(items, dict):
                item = items.get(task_id)
            elif isinstance(items, list) and items:
                item = items[0]
            if not item:
                progress(0.3, desc=f"排队中… {elapsed}s")
                time.sleep(3)
                continue

            status = item.get("status")
            stage = (item.get("progress_text") or "").split("|")[-1].strip()
            if stage and stage != last_stage:
                last_stage = stage
            progress(min(0.85, 0.3 + (status or 0) * 0.5), desc=f"{_STATUS_TEXT.get(status, status)} · {elapsed}s")

            if status == 1:  # 成功（实测该 API 用 1 表示完成，非 2）
                res = _parse_result(item.get("result"))
                rel = res.get("file") if res else None
                if not rel:
                    return None, None, f"任务成功但未找到音频地址：{str(item)[:300]}"
                out = _download(rel)
                gen = (res.get("generation_info") or "").replace("\n", " ")
                dur_sec = _wav_duration_seconds(out)
                dur_txt = f"{dur_sec:.1f}s" if dur_sec is not None else "未知"
                # 成品时长反馈：可控模式给出目标值，便于确认长歌词是否唱完
                target_txt = f" · 目标 {final_duration}s" if controlled else "（= 参考音频时长）"
                msg = (
                    f"✅ 完成 · 用时 {elapsed}s · 成品时长 {dur_txt}{target_txt}\n"
                    f"模式：{note}\n文件：{out}\n"
                    f"种子：{res.get('seed_value', '-')} · DiT：{res.get('dit_model', '-')} · 步数：{int(inference_steps)}\n{gen}"
                )
                progress(1.0, desc="完成")
                return out, final_duration, msg
            if status == 2 or (isinstance(status, str) and status.lower() in ("failed", "error")):
                return None, None, f"生成失败：{str(item)[:400]}"
            time.sleep(3)
    except Exception as e:
        return None, None, f"轮询/下载异常：{type(e).__name__}: {e}"


def build_acestep_cover_tab():
    """构建第 ⑦ 个 Tab：ACE-Step 音色跟随 / 时长可控演唱。"""
    gr.Markdown(
        "### ACE-Step 1.5 · 音色跟随 / 时长可控演唱\n"
        "上传一段参考音频作为音色参考，让 ACE-Step 按提示词与歌词端到端出成品，"
        "无多阶段拼接的机械音。所有计算在本机完成，数据不出本机。\n\n"
        "**生成模式**：\n"
        "- **音色演唱（时长可控·推荐）**：`text2music` + 参考音频作音色 conditioning + 显式时长。"
        "成品时长由下方「时长」决定（或按歌词自动估算），**长歌词可完整唱完**，不再恒等于参考音频长度。\n"
        "- **原曲翻唱 Cover**：`cover` 任务，跟随原曲旋律结构，但成品时长被 ACE-Step 锁定为参考音频长度，"
        "时长滑块不生效；仅短歌词/需要贴原曲旋律时使用。\n\n"
        "**推理步数**：后端为 SFT 模型，建议 **40–60**（默认 50）；过低（如 8）会明显损伤音质、易出废歌。\n\n"
        "> 首次使用请先双击运行项目根目录的 **start_acestep_api.bat** 启动独立 API Server，"
        "再点下方「检测服务」。该服务使用 ACE-Step 自己的环境，与现有工作台环境完全隔离。"
    )
    with gr.Row():
        status_box = gr.Textbox(label="服务状态", value=check_service(), interactive=False, scale=4)
        check_btn = gr.Button("检测服务", size="sm", scale=1)
    check_btn.click(fn=check_service, outputs=[status_box])

    with gr.Row():
        with gr.Column(scale=1):
            mode = gr.Radio(
                label="生成模式",
                choices=[MODE_CONTROLLED, MODE_COVER],
                value=MODE_CONTROLLED,
            )
            src = gr.Audio(label="参考音频（音色参考）", type="filepath")
            prompt = gr.Textbox(
                label="风格提示词 prompt",
                value="warm emotional Mandarin pop ballad, female vocal, gentle piano and strings, clean studio recording",
                lines=3,
            )
            lyrics = gr.Textbox(label="歌词（长歌词请完整粘贴，时长可自动匹配）", value="", lines=3)
            vocal_language = gr.Dropdown(
                label="演唱语言", choices=["zh", "en", "ja", "ko", "es", "fr"], value="zh"
            )
        with gr.Column(scale=1):
            duration = gr.Slider(
                label="时长（秒）— 音色演唱模式生效；勾选自动匹配后取「滑块/歌词估算」较大值",
                minimum=10, maximum=600, value=30, step=1,
            )
            auto_duration = gr.Checkbox(label="按歌词长度自动匹配时长（保证长歌词唱得完）", value=True)
            with gr.Row():
                est_btn = gr.Button("估算歌词时长", size="sm")
                est_box = gr.Textbox(label="歌词估算时长", interactive=False, scale=1)
            est_btn.click(fn=lambda ly: f"约 {infer_duration_from_lyrics(ly)} 秒", inputs=[lyrics], outputs=[est_box])
            strength = gr.Slider(
                label="跟随强度 audio_cover_strength（仅 cover 模式生效）",
                minimum=0.0, maximum=1.0, value=0.7, step=0.05,
            )
            steps = gr.Slider(
                label="推理步数 inference_steps（SFT 建议 40–60，过低音质差）",
                minimum=1, maximum=200, value=50, step=1,
            )
            with gr.Row():
                rand = gr.Checkbox(label="随机种子", value=False)
                seed = gr.Number(label="固定 seed", value=20261118, precision=0)
            run_btn = gr.Button("生成", variant="primary")

    # ===== 可选入口①：参数 JSON 整段粘贴 =====
    with gr.Accordion("① 可选：参数 JSON 整段粘贴（解析填入上方控件，手动输入方式不变）", open=False):
        gr.Markdown(
            "粘贴《ACE-Step 1.5 歌词与风格提示词规范》里的整段参数 JSON，点「解析填入」自动填好 "
            "prompt(caption)/lyrics/语言/时长/模式与 bpm、key_scale 等高级字段；也可继续手动逐项填写。"
        )
        json_box = gr.Textbox(label="参数 JSON", lines=8, value="")
        with gr.Row():
            json_parse_btn = gr.Button("解析填入", size="sm", scale=1)
            adv_box = gr.Textbox(label="高级字段（自动生成，通常无需改动）", value="{}", lines=2, scale=3)
        json_status = gr.Textbox(label="解析结果", interactive=False, lines=3)
        json_parse_btn.click(
            fn=parse_params_json,
            inputs=[json_box],
            outputs=[prompt, lyrics, vocal_language, duration, auto_duration, mode, adv_box, json_status],
        )

    out_audio = gr.Audio(label="成品", type="filepath")
    out_dur = gr.Number(label="成品实际时长（秒）", interactive=False, precision=1)
    out_info = gr.Textbox(label="运行信息", lines=5, interactive=False)
    run_btn.click(
        fn=do_cover,
        inputs=[src, prompt, lyrics, vocal_language, duration, strength, steps, rand, seed, mode, auto_duration, adv_box],
        outputs=[out_audio, out_dur, out_info],
    )

    # ===== 可选入口②：LoRA 选择 / 加载 =====
    with gr.Accordion("② 可选：LoRA 权重选择（加载后服务端生成自动生效）", open=False):
        gr.Markdown(
            "前提是独立 API Server 已加载带 LoRA 端点的版本（本环境 /openapi.json 已确认暴露 "
            "`/v1/lora/{status,load,unload,toggle,scale}`）。加载权重后，上方「生成」会自动带上 LoRA，"
            "无需改 do_cover。训练产物在训练区导出后，可在此刷新/选择。"
        )
        with gr.Row():
            lora_status_box = gr.Textbox(label="LoRA 当前状态", value=lora_status(), interactive=False, lines=3, scale=3)
            lora_refresh_btn = gr.Button("刷新状态", size="sm", scale=1)
        lora_refresh_btn.click(fn=lora_status, outputs=[lora_status_box])
        with gr.Row():
            lora_path = gr.Dropdown(
                label="LoRA 权重路径（选择或手动粘贴）", choices=[], allow_custom_value=True, scale=3
            )
            lora_scan_btn = gr.Button("扫描权重", size="sm", scale=1)
        lora_scan_info = gr.Textbox(label="扫描结果", interactive=False, lines=3)
        lora_scan_btn.click(fn=list_lora_paths, outputs=[lora_path, lora_scan_info])
        with gr.Row():
            lora_name = gr.Textbox(label="适配器名（可选）", value="", scale=2)
            lora_scale_num = gr.Slider(label="LoRA scale", minimum=0.0, maximum=2.0, value=1.0, step=0.05, scale=2)
        with gr.Row():
            lora_load_btn = gr.Button("加载 LoRA", size="sm")
            lora_unload_btn = gr.Button("卸载 LoRA", size="sm")
            lora_on_btn = gr.Button("启用", size="sm")
            lora_off_btn = gr.Button("禁用", size="sm")
            lora_scale_btn = gr.Button("设置 scale", size="sm")
        lora_msg = gr.Textbox(label="操作结果", interactive=False, lines=3)
        lora_load_btn.click(fn=lora_load, inputs=[lora_path, lora_name], outputs=[lora_msg])
        lora_unload_btn.click(fn=lora_unload, outputs=[lora_msg])
        lora_on_btn.click(fn=lambda: lora_toggle(True), outputs=[lora_msg])
        lora_off_btn.click(fn=lambda: lora_toggle(False), outputs=[lora_msg])
        lora_scale_btn.click(fn=lora_scale, inputs=[lora_scale_num], outputs=[lora_msg])

    # ===== 可选入口③：LoRA / LoKr 训练 =====
    with gr.Accordion("③ 可选：LoRA / LoKr 训练（实验性，需独立 API Server 支持）", open=False):
        gr.Markdown(
            "训练走服务端 `/v1/dataset/scan` → `/v1/dataset/preprocess` → `/v1/training/start`（LoRA）"
            "或 `/v1/training/start_lokr`（LoKr）。\n\n"
            "**数据格式**：音频目录内含音频 + 同名 `.lyrics.txt`（歌词）+ 可选 `.json`（caption/bpm/keyscale/"
            "timesignature/language）。**重要**：训练数据须为你拥有版权/原创的内容，避免使用有版权争议的专辑；"
            "数据来源请你自行决定，本工具不会擅自抓取或假设。\n\n"
            "**显存提示**：本机 RTX 4060 Ti 16GB（实测空闲约 8.6GB）。官方 LoRA 训练建议 16GB 最低/20GB 推荐、"
            "约 100 首数据跑数百 epoch；显存吃紧时优先用 **LoKr**（更省显存）并减小 rank/batch、开启梯度检查点。"
        )
        gr.Markdown("**步骤 1：扫描数据集**")
        with gr.Row():
            ds_dir = gr.Textbox(label="音频数据集目录（绝对路径）", value="", scale=3)
            ds_name = gr.Textbox(label="数据集名", value="my_lora_dataset", scale=1)
        with gr.Row():
            ds_tag = gr.Textbox(label="自定义触发词 custom_tag（可选）", value="", scale=2)
            ds_tagpos = gr.Dropdown(label="触发词位置", choices=["replace", "append", "prepend"], value="replace", scale=1)
            ds_inst = gr.Checkbox(label="全部为纯音乐（无歌词）", value=False, scale=1)
            ds_scan_btn = gr.Button("扫描数据集", size="sm", scale=1)
        ds_msg = gr.Textbox(label="数据集操作结果", lines=4, interactive=False)

        gr.Markdown("**步骤 2：预处理为张量（训练前必须）**")
        with gr.Row():
            tensor_dir = gr.Textbox(label="张量输出目录（绝对路径）", value="", scale=3)
            ds_prep_btn = gr.Button("预处理", size="sm", scale=1)

        gr.Markdown("**步骤 3：启动训练**")
        with gr.Row():
            train_kind = gr.Radio(label="训练类型", choices=["LoKr", "LoRA"], value="LoKr")
            train_rank = gr.Number(label="rank / linear_dim", value=16, precision=0)
            train_alpha = gr.Number(label="alpha", value=32, precision=0)
        with gr.Row():
            train_epochs = gr.Number(label="epochs", value=3, precision=0)
            train_batch = gr.Number(label="batch size", value=1, precision=0)
            train_accum = gr.Number(label="grad accumulation", value=4, precision=0)
            train_save = gr.Number(label="每 N epoch 保存", value=1, precision=0)
        with gr.Row():
            train_lr = gr.Number(label="learning rate", value=0.0001, precision=6)
            train_shift = gr.Number(label="training shift", value=3.0, precision=2)
            train_seed = gr.Number(label="seed", value=20261118, precision=0)
        with gr.Row():
            train_fp8 = gr.Checkbox(label="fp8（仅 LoRA）", value=False)
            train_ckpt = gr.Checkbox(label="梯度检查点（省显存）", value=True)
        with gr.Row():
            train_out = gr.Textbox(label="权重输出目录（绝对路径，可留空用默认）", value="", scale=3)
            train_start_btn = gr.Button("启动训练", variant="primary", size="sm", scale=1)
        train_status_box = gr.Textbox(label="训练状态", value=training_status(), lines=4, interactive=False)
        with gr.Row():
            train_refresh_btn = gr.Button("刷新训练状态", size="sm")
            train_stop_btn = gr.Button("停止训练", size="sm")
        train_msg = gr.Textbox(label="训练操作结果", lines=4, interactive=False)

        gr.Markdown("**步骤 4：导出权重（导出后到区②刷新并加载）**")
        with gr.Row():
            exp_src = gr.Textbox(label="训练输出目录（源）", value="", scale=2)
            exp_dst = gr.Textbox(label="导出路径（目标 .safetensors）", value="", scale=2)
            exp_btn = gr.Button("导出", size="sm", scale=1)

        ds_scan_btn.click(
            fn=dataset_scan,
            inputs=[ds_dir, ds_name, ds_tag, ds_tagpos, ds_inst],
            outputs=[ds_msg],
        )
        ds_prep_btn.click(fn=dataset_preprocess, inputs=[tensor_dir], outputs=[ds_msg])
        train_start_btn.click(
            fn=training_start,
            inputs=[
                tensor_dir, train_kind, train_rank, train_alpha, train_epochs, train_batch,
                train_accum, train_lr, train_shift, train_save, train_out, train_seed,
                train_fp8, train_ckpt,
            ],
            outputs=[train_msg],
        )
        train_refresh_btn.click(fn=training_status, outputs=[train_status_box])
        train_stop_btn.click(fn=training_stop, outputs=[train_msg])
        exp_btn.click(fn=training_export, inputs=[exp_dst, exp_src], outputs=[train_msg])
