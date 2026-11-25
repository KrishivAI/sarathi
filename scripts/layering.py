#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Layer rule check: a workspace crate may depend only on workspace crates in the same or a lower layer."""
import json
import subprocess
import sys
import tomllib
from pathlib import Path

root = Path(__file__).resolve().parent.parent
layers = tomllib.loads((root / "scripts" / "layers.toml").read_text())["layers"]
meta = json.loads(
    subprocess.run(
        ["cargo", "metadata", "--format-version", "1", "--no-deps", "--locked"],
        cwd=root, check=True, capture_output=True, text=True,
    ).stdout
)
members = {p["name"]: p for p in meta["packages"]}
errors = []
for name in sorted(set(members) - set(layers)):
    errors.append(f"{name}: not in scripts/layers.toml (add it with its layer)")
for name in sorted(set(layers) - set(members)):
    errors.append(f"{name}: in scripts/layers.toml but not a workspace crate")
for name, pkg in members.items():
    if name not in layers:
        continue
    for dep in pkg["dependencies"]:
        if dep["name"] in layers and layers[dep["name"]] > layers[name]:
            errors.append(
                f"{name} (layer {layers[name]}) depends on {dep['name']} (layer {layers[dep['name']]})"
            )
if errors:
    print("\n".join(f"layering: {e}" for e in errors), file=sys.stderr)
    sys.exit(1)
print(f"layering: OK ({len(members)} crates)")
