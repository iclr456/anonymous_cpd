"""Run extraction and detection together or in separate Python environments."""
import argparse
import subprocess
import sys
from .utils import load_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--stage", choices=("all", "extract", "scan"), default="all")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.stage == "all":
        for stage, runtime in (("extract", "foundation_python"), ("scan", "scan_python")):
            executable = config.get("runtime", {}).get(runtime) or sys.executable
            subprocess.run([executable, "-m", "tsfm_cpd", "--config", args.config,
                            "--stage", stage], check=True)
        return
    from .utils import save_manifest
    if args.stage == "extract":
        from .extraction import extract
        extract(config)
    else:
        from .detection import detect
        detect(config)
    save_manifest(config, args.stage)


if __name__ == "__main__":
    main()
