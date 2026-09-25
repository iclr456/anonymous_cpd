import json
from pathlib import Path


def load_config(path):
    path = Path(path).resolve()
    config = json.loads(path.read_text(encoding="utf-8"))

    def resolve(value):
        candidate = Path(value).expanduser()
        return str((path.parent / candidate).resolve())

    config["input"]["path"] = resolve(config["input"]["path"])
    config["output_dir"] = resolve(config["output_dir"])
    if config.get("timesfm1_source"):
        config["timesfm1_source"] = resolve(config["timesfm1_source"])
    keys = [model["key"] for model in config["models"]]
    if not keys or len(set(keys)) != len(keys):
        raise ValueError("Provide at least one model, with unique model keys")
    for model in config["models"]:
        key = model["key"]
        if not key or key in {".", ".."} or any(c in key for c in '/\\:'):
            raise ValueError("Model keys must be simple directory names")
        if "path" in model:
            model["path"] = resolve(model["path"])
    for key, value in config.get("runtime", {}).items():
        if value:
            config["runtime"][key] = resolve(value)
    if config["embedding"]["context_length"] < 1 or config["embedding"]["stride"] < 1:
        raise ValueError("Context length and stride must be positive")
    if not config["scan"]["vote_thresholds"] or any(
        not 0 <= threshold <= 1 for threshold in config["scan"]["vote_thresholds"]
    ):
        raise ValueError("Voting thresholds should lie between [0, 1]")
    return config
