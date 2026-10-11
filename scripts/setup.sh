#!/usr/bin/env bash
set -euo pipefail
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_dir"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-/workspace/.cache/pip}"
python3 -m venv .venv
.venv/bin/python -m pip --disable-pip-version-check install -r requirements.txt
.venv/bin/python -m pip --disable-pip-version-check install --no-deps --no-build-isolation -e .
.venv/bin/python -m pip check
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/keiba-scraper --help
