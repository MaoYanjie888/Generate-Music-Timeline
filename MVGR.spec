# -*- mode: python ; coding: utf-8 -*-
# MVGR 歌词时间戳工具 — PyInstaller 打包配置
# 输出：dist/MVGR/MVGR.exe（onedir）
# 外部资源（不打包进 EXE，随目录分发）：
#   models/large-v3.pt  Whisper 权重
#   configs/default.yaml  配置文件

import os
import sys

from PyInstaller.utils.hooks import collect_all

datas = []
binaries = []
hiddenimports = []

# 自动收集 site-packages 中所有带 version.txt 的包（gradio 6.x 依赖的
# safehttpx/groovy 等小包，缺 version.txt 会导致 EXE 启动崩溃）
sp = os.path.join(sys.prefix, "Lib", "site-packages")
version_pkgs = []
if os.path.isdir(sp):
    for name in os.listdir(sp):
        d = os.path.join(sp, name)
        if os.path.isdir(d) and os.path.isfile(os.path.join(d, "version.txt")):
            version_pkgs.append(name)

for pkg in [
    "gradio",
    "safehttpx",
    "groovy",
    "pandas",
    "whisper",
    "tiktoken",
    "yaml",
    "regex",
    "numpy",
    "scipy",
    "httpx",
    "httpcore",
    "h11",
    "anyio",
    "sniffio",
    "starlette",
    "fastapi",
    "pydantic",
    "pydantic_core",
    "typer",
    "click",
    "uvicorn",
    "websockets",
    "wsproto",
    "orjson",
    "certifi",
    "charset_normalizer",
    "idna",
    "huggingface_hub",
    "tokenizers",
] + version_pkgs:
    try:
        d, b, h = collect_all(pkg)
        datas += d
        binaries += b
        hiddenimports += h
    except Exception:
        pass

hiddenimports += [
    "src.webui.app",
    "src.webui.session",
    "src.model_manager",
    "src.utils.config",
    "src.utils.logger",
    "src.utils.paths",
    "whisper.audio",
    "whisper.tokenizer",
    "whisper.model",
    "whisper.transcribe",
    "whisper.timing",
    "tiktoken_ext.openai_public",
    "tiktoken_ext",
]

# 随 EXE 分发的本地资源（COLLECT 会重建 dist/MVGR，必须显式声明）
datas += [
    (os.path.join(r"E:/MVGR/models/large-v3.pt"), "models"),
    (os.path.join(r"E:/MVGR/configs/default.yaml"), "configs"),
    (os.path.join(r"E:/MVGR/models/Qwen3-ASR-1.7B-hf"), "models/Qwen3-ASR-1.7B-hf"),
    (os.path.join(r"E:/MVGR/models/Qwen3-ForcedAligner-0.6B-hf"), "models/Qwen3-ForcedAligner-0.6B-hf"),
]

a = Analysis(
    ["webui.py"],
    pathex=["E:/MVGR"],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        "tkinter",
        "matplotlib",
        "IPython",
        "jupyter",
        "notebook",
        "pytest",
        "setuptools",
    ],
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="MVGR",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,
    disable_windowed_traceback=False,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name="MVGR",
)
