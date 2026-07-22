#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat >&2 <<'USAGE'
usage: run-research-loop.sh --config PATH [--resume]
USAGE
}

config=""
resume=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)
      [[ $# -ge 2 && -n "$2" ]] || {
        echo "missing value for --config" >&2
        usage
        exit 2
      }
      config="$2"
      shift 2
      ;;
    --resume)
      resume=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "unknown option: $1" >&2
      usage
      exit 2
      ;;
  esac
done

[[ -n "$config" ]] || {
  echo "--config is required" >&2
  usage
  exit 2
}

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
repo_root=$(cd "$script_dir/../.." && pwd -P) || {
  echo "could not resolve repository root" >&2
  exit 2
}
[[ -d "$repo_root" && -f "$repo_root/pyproject.toml" ]] || {
  echo "invalid repository root: $repo_root" >&2
  exit 2
}

if [[ "$config" = /* ]]; then
  config_path="$config"
else
  config_path="$PWD/$config"
fi
config_dir=$(cd "$(dirname "$config_path")" 2>/dev/null && pwd -P) || {
  echo "config directory not found: $config" >&2
  exit 2
}
config_path="$config_dir/$(basename "$config_path")"
[[ -f "$config_path" ]] || {
  echo "config not found: $config_path" >&2
  exit 2
}
[[ ! -L "$config_path" ]] || {
  echo "config must be an existing project path: $config_path" >&2
  exit 2
}
case "$config_path" in
  "$repo_root"/*) ;;
  *)
    echo "config must be an existing project path: $config_path" >&2
    exit 2
    ;;
esac

cd "$repo_root"
if [[ "$resume" -eq 1 ]]; then
  uv run python -m power_forecasting.cli research-loop --config "$config_path" --resume
else
  uv run python -m power_forecasting.cli research-loop --config "$config_path"
fi
