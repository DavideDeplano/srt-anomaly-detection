"""
Logger management module.

Provides a centralized logger configuration for the entire project.
Logging parameters (level, output file) are read from config/default.yaml.
"""

from pathlib import Path
import logging

def setup_logger(cfg: dict) -> None:
    """Configure project-wide logging from YAML config."""
    log_cfg = cfg.get("logging", {})
    level = getattr(logging, str(log_cfg.get("level", "INFO")).upper(), logging.INFO)
    log_file = Path(str(log_cfg.get("file", "results/run.log")))
    log_file.parent.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler(),
        ],
        force=True,
    )
