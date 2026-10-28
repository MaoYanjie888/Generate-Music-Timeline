"""配置加载模块（精简版）— 歌词时间戳工具所需的最小配置。"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import yaml

from src.utils.logger import get_logger
from src.utils.paths import resource_path

logger = get_logger(__name__)

DEFAULT_CONFIG_PATH = resource_path("configs/default.yaml")


class ConfigManager:
    """配置管理器（精简版）。

    负责加载 configs/default.yaml，支持点号嵌套取值。
    """

    def __init__(self, config_path: Optional[str] = None) -> None:
        self.config_path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
        self.config: dict[str, Any] = {}
        self._load()

    def _load(self) -> None:
        if not self.config_path.exists():
            logger.warning(f"配置文件不存在: {self.config_path}，使用内置默认配置")
            self.config = self._default_config()
            return
        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                self.config = yaml.safe_load(f) or {}
            logger.info(f"配置加载完成: {self.config_path}")
        except Exception as e:
            logger.error(f"配置加载失败: {e}，使用内置默认配置")
            self.config = self._default_config()

    def _default_config(self) -> dict[str, Any]:
        return {
            "project": {"name": "MVGR", "version": "1.0.0", "workspace": "output"},
            "models": {
                "whisper": {"name": "large-v3", "device": "cuda", "language": "zh"},
            },
            "audio": {"language": "zh"},
        }

    @staticmethod
    def build_from_ui(
        audio_path: str = "",
        portrait_path: str = "",
        style_preset: str = "",
        segments: Optional[list[dict[str, Any]]] = None,
    ) -> dict[str, Any]:
        """从 UI 参数构建配置（精简版：仅回填音频路径）。"""
        cm = ConfigManager()
        config = dict(cm.config)
        config["audio_path"] = audio_path
        return config

    def get(self, key: str, default: Any = None) -> Any:
        """获取配置值，支持点号分隔的嵌套键（如 'audio.language'）。"""
        keys = key.split(".")
        value = self.config
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
                if value is None:
                    return default
            else:
                return default
        return value


# --- 便捷函数与别名 ---

def load_config(config_path: Optional[str] = None) -> dict[str, Any]:
    cm = ConfigManager(config_path)
    return cm.config


ConfigLoader = ConfigManager
