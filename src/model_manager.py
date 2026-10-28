"""模型管理模块（精简版）— 支持 Qwen3-ASR + ForcedAligner 歌词识别。

负责 Qwen3-ASR / Qwen3-ForcedAligner 模型的加载、复用与显存释放。
"""

from __future__ import annotations

import gc
from typing import Any, Optional

from src.utils.logger import get_logger
from src.utils.paths import resource_path

logger = get_logger(__name__)

QWEN_ASR_PATH = resource_path("models/Qwen3-ASR-1.7B-hf")
QWEN_ALIGNER_PATH = resource_path("models/Qwen3-ForcedAligner-0.6B-hf")


class ModelManager:
    """模型管理器（精简版）：负责 Qwen3-ASR / ForcedAligner 的加载、复用与释放。"""

    def __init__(self) -> None:
        self._models: dict[str, Any] = {}
        self._config: dict[str, Any] = {}

    # ---- 模型加载 ----

    def load_model(self, model_type: str) -> Any:
        """加载指定模型。支持 qwen-asr / qwen-aligner。"""
        key = model_type
        if key in self._models and self._models[key] is not None:
            return self._models[key]
        if model_type == "qwen-asr":
            model = self._load_qwen_asr()
        elif model_type == "qwen-aligner":
            model = self._load_qwen_aligner()
        else:
            raise ValueError(f"不支持的模型类型: {model_type}")
        self._models[key] = model
        return model

    def get_model(self, model_type: str) -> Optional[Any]:
        """获取已加载模型，未加载返回 None。"""
        return self._models.get(model_type)

    def unload_model(self, model_type: str) -> None:
        """卸载模型并释放显存。"""
        model = self._models.pop(model_type, None)
        if model is None:
            return
        try:
            del model
        except Exception:
            pass
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    def unload_all(self) -> None:
        """卸载全部模型。"""
        for key in list(self._models):
            self.unload_model(key)

    # ---- Qwen3-ASR ----

    def _load_qwen_asr(self):
        """加载 Qwen3-ASR-1.7B（本地权重 models/Qwen3-ASR-1.7B-hf）。"""
        from transformers import AutoModelForMultimodalLM, AutoProcessor

        if not QWEN_ASR_PATH.exists():
            raise FileNotFoundError(
                f"未找到 Qwen3-ASR 权重: {QWEN_ASR_PATH}，请先下载模型"
            )
        logger.info(f"加载 Qwen3-ASR 模型: {QWEN_ASR_PATH}")
        processor = AutoProcessor.from_pretrained(str(QWEN_ASR_PATH))
        model = AutoModelForMultimodalLM.from_pretrained(
            str(QWEN_ASR_PATH),
            device_map="auto",
        )
        logger.info("Qwen3-ASR 加载完成")
        return {"processor": processor, "model": model}

    # ---- Qwen3-ForcedAligner ----

    def _load_qwen_aligner(self):
        """加载 Qwen3-ForcedAligner-0.6B（本地权重 models/Qwen3-ForcedAligner-0.6B-hf）。"""
        import torch
        from transformers import AutoModelForTokenClassification, AutoProcessor

        if not QWEN_ALIGNER_PATH.exists():
            raise FileNotFoundError(
                f"未找到 Qwen3-ForcedAligner 权重: {QWEN_ALIGNER_PATH}，请先下载模型"
            )
        logger.info(f"加载 Qwen3-ForcedAligner 模型: {QWEN_ALIGNER_PATH}")
        processor = AutoProcessor.from_pretrained(str(QWEN_ALIGNER_PATH))
        # 使用 fp32 加载：ForcedAligner 输出 token 级分类 logits，bf16 会损失
        # 词边界精度；0.6B 模型 fp32 仅约 2.4GB 显存，16GB 显存完全够用。
        model = AutoModelForTokenClassification.from_pretrained(
            str(QWEN_ALIGNER_PATH),
            dtype=torch.float32,
            device_map="auto",
        )
        logger.info("Qwen3-ForcedAligner 加载完成")
        return {"processor": processor, "model": model}
