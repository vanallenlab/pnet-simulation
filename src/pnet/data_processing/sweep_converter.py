"""Convert a W&B sweep YAML file to bash commands and optionally execute them.
python sweep_converter.py sweep.yaml --dry    # just show commands
python sweep_converter.py sweep.yaml          # actually run them
"""

import itertools, json, subprocess, sys
from pathlib import Path

try:
    import yaml  # pip install pyyaml
except ImportError:
    print("Please `pip install pyyaml`", file=sys.stderr)
    sys.exit(1)


def load_cfg(p: Path):
    with open(p, "r") as f:
        return yaml.safe_load(f)


def expand(parameters: dict):
    fixed = {}
    axes = []
    for k, v in (parameters or {}).items():
        if isinstance(v, dict) and "values" in v:
            axes.append((k, v["values"]))
        elif isinstance(v, dict) and "value" in v:
            fixed[k] = v["value"]
        else:
            fixed[k] = v
    if not axes:
        return [fixed]
    keys = [k for k, _ in axes]
    return [dict(zip(keys, prod), **fixed) for prod in itertools.product(*[vals for _, vals in axes])]


def to_cli(k, v):
    k = f"--{k}"
    if v is None:
        return []
    if isinstance(v, bool):
        return [k] if v else []
    if isinstance(v, (list, dict)):
        return [k, json.dumps(v)]
    return [k, str(v)]


def main():
    if len(sys.argv) < 2:
        print("Usage: python sweep_min.py sweep.yaml [--dry]", file=sys.stderr)
        sys.exit(2)
    dry = "--dry" in sys.argv
    cfg = load_cfg(Path(sys.argv[1]))
    prog = cfg["program"]
    combos = expand(cfg.get("parameters", {}))

    n_fail = 0
    for params in combos:
        # ignore clearly W&B-only fields if present
        params = {k: v for k, v in params.items() if k not in {"wandb_group"}}
        cmd = [sys.executable, prog] if prog.endswith(".py") else [prog]
        for k, v in params.items():
            cmd += to_cli(k, v)
        print("$", " ".join(cmd))
        if not dry:
            rc = subprocess.call(cmd)
            if rc != 0:
                n_fail += 1
    if n_fail:
        sys.exit(1)


if __name__ == "__main__":
    main()
