import yaml
from pathlib import Path

CONFIG_PATH = Path("config/default.yaml")

with open(CONFIG_PATH, "r") as f:
    cfg = yaml.safe_load(f)

random_seed = int(cfg["random_seed"])
paths = cfg["paths"]
pipeline = cfg["pipeline"]
simulation = cfg["simulation"]
filters = cfg["filters"]
logging_cfg = cfg["logging"]