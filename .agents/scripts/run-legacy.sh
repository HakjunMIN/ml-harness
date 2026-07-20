#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat >&2 <<'USAGE'
usage: run-legacy.sh --adapter PATH --run-dir DIR [--run-id ID]
USAGE
}

adapter=""
run_dir=""
run_id=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --adapter) [[ $# -ge 2 ]] || { echo "missing value for --adapter" >&2; exit 2; }; adapter="$2"; shift 2 ;;
    --run-dir) [[ $# -ge 2 ]] || { echo "missing value for --run-dir" >&2; exit 2; }; run_dir="$2"; shift 2 ;;
    --run-id) [[ $# -ge 2 ]] || { echo "missing value for --run-id" >&2; exit 2; }; run_id="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown option: $1" >&2; usage; exit 2 ;;
  esac
done
[[ -n "$adapter" ]] || { echo "--adapter is required" >&2; usage; exit 2; }
[[ -n "$run_dir" ]] || { echo "--run-dir is required" >&2; usage; exit 2; }

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
repo_root=$(cd "$script_dir/../.." && pwd -P)
export PYTHONPATH="$repo_root/.agents${PYTHONPATH:+:$PYTHONPATH}"
args=(uv run python -m harness.contract --adapter "$adapter" --run-dir "$run_dir")
if [[ -n "$run_id" ]]; then
  args+=(--run-id "$run_id")
fi
cd "$repo_root"
"${args[@]}"
