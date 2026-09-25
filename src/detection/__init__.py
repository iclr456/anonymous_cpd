"""One job-based pipeline for time-series and video change-point detection."""
from .utils import load_config


def run_stage(config, stage):
    """Run extract or scan for all jobs in a configuration from load_config()."""
    from .runner import run_stage as run
    return run(config, stage)
