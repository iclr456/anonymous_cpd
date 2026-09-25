"""Extract one model at a time and save aligned statistics."""
import gc
from pathlib import Path
import numpy as np
from .utils import SummaryAdapter, generate_window_sizes, load_series, save_statistics
from .pipeline import compute_statistics


def extract(config):
    values = load_series(config["input"])
    embedding = config["embedding"]
    count = (len(values) - embedding["context_length"]) // embedding["stride"] + 1
    length = count - 1 - (2 if config["tv"]["drop_edge_statistics"] else 0)
    generate_window_sizes(length, **config["windows"])
    np.random.seed(config["windows"]["seed"])
    for model in config["models"]:
        if model["backend"] != "summary" and not Path(model["path"]).is_dir():
            raise FileNotFoundError(f"Missing weights for {model['key']}")
    if any(model["backend"] != "summary" for model in config["models"]):
        import torch
        torch.manual_seed(config["windows"]["seed"])
    for model in config["models"]:
        if model["backend"] == "summary":
            adapter = SummaryAdapter()
        else:
            import torch
            from .foundation_adapters import load_foundation_adapter
            adapter = load_foundation_adapter(
                model["backend"], Path(model["path"]), embedding["device"],
                timesfm1_source=Path(config["timesfm1_source"]),
            )
        print(f"Embedding: {model['key']}", flush=True)
        result = compute_statistics(values, adapter, config, batch_size=model["batch_size"])
        save_statistics(result, Path(config["output_dir"]) / model["key"])
        del adapter, result
        gc.collect()
        if model["backend"] != "summary" and torch.cuda.is_available():
            torch.cuda.empty_cache()
