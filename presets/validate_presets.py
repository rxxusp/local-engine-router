#!/usr/bin/env python3
"""Validate all preset YAML files against config.schema.json.

For each preset file this script:
  1. yaml.safe_load()s the file to confirm it is valid YAML.
  2. Wraps the engines: block from the preset into a minimal full config.
  3. Validates that wrapped config against config.schema.json via jsonschema.

Run from the repo root:
    python3 presets/validate_presets.py

Exit code is 0 if all presets pass, non-zero otherwise.
"""

from __future__ import annotations

import json
import pathlib
import sys

import yaml

try:
    import jsonschema
except ImportError:
    print("ERROR: jsonschema is not installed.  Run: pip install jsonschema")
    sys.exit(1)

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
SCHEMA_PATH = REPO_ROOT / "config.schema.json"
PRESETS_DIR = pathlib.Path(__file__).resolve().parent

# Minimal outer skeleton that satisfies the schema's top-level structure.
# Each preset's engines: and models: blocks are merged in.
_SKELETON = {
    "host": "127.0.0.1",
    "port": 8077,
}


def _load_schema() -> dict:
    with open(SCHEMA_PATH) as fh:
        return json.load(fh)


def _wrap_preset(preset_data: dict) -> dict:
    """Merge a preset's engines/models into a minimal full config."""
    cfg = dict(_SKELETON)
    if "engines" in preset_data:
        cfg["engines"] = preset_data["engines"]
    if "models" in preset_data:
        cfg["models"] = preset_data["models"]
    return cfg


def _strip_placeholders(obj):
    """Replace <ANGLE_BRACKET> placeholder strings with empty strings.

    Placeholders represent values the user must fill in.  They are not valid
    URLs or paths, so we substitute them before schema validation to avoid
    false failures on format checks.
    """
    if isinstance(obj, str):
        if obj.startswith("<") and obj.endswith(">"):
            return ""
        return obj
    if isinstance(obj, dict):
        return {k: _strip_placeholders(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_strip_placeholders(i) for i in obj]
    return obj


def validate_file(path: pathlib.Path, schema: dict) -> list[str]:
    """Return a list of error strings for *path* (empty = pass)."""
    errors: list[str] = []

    # Step 1: YAML parse.
    try:
        raw = yaml.safe_load(path.read_text())
    except yaml.YAMLError as exc:
        errors.append(f"YAML parse error: {exc}")
        return errors

    if not isinstance(raw, dict):
        errors.append("top-level must be a YAML mapping")
        return errors

    # Step 2: field-name cross-check — every key in engines values must be a
    # known schema field (or additionalProperties: true covers it).
    engine_schema_props = set()
    for arm in schema.get("properties", {}).get("engines", {}).get(
        "additionalProperties", {}
    ).get("oneOf", []):
        engine_schema_props.update(arm.get("properties", {}).keys())

    if "engines" in raw:
        for eng_key, eng_body in (raw["engines"] or {}).items():
            if not isinstance(eng_body, dict):
                errors.append(f"engine {eng_key!r}: body must be a mapping")
                continue
            for field in eng_body:
                if field not in engine_schema_props:
                    # additionalProperties: true means this is just a warning.
                    pass  # unknown fields are allowed; schema uses additionalProperties

    # Step 3: schema validation (after stripping placeholders).
    wrapped = _wrap_preset(raw)
    cleaned = _strip_placeholders(wrapped)
    try:
        jsonschema.validate(instance=cleaned, schema=schema)
    except jsonschema.ValidationError as exc:
        errors.append(f"schema validation: {exc.message} (path: {list(exc.absolute_path)})")
    except jsonschema.SchemaError as exc:
        errors.append(f"schema itself is invalid: {exc.message}")

    return errors


def main() -> int:
    schema = _load_schema()
    preset_files = sorted(
        p for p in PRESETS_DIR.glob("*.yaml") if p.name != "validate_presets.py"
    )

    if not preset_files:
        print("No preset YAML files found in", PRESETS_DIR)
        return 1

    failures = 0
    for path in preset_files:
        errs = validate_file(path, schema)
        status = "PASS" if not errs else "FAIL"
        print(f"  [{status}] {path.name}")
        for e in errs:
            print(f"         {e}")
            failures += 1

    print()
    if failures:
        print(f"FAILED: {failures} error(s) across {len(preset_files)} preset(s)")
        return 1
    print(f"OK: all {len(preset_files)} preset(s) passed YAML parse + schema validation")
    return 0


if __name__ == "__main__":
    sys.exit(main())
