#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat >&2 <<'USAGE'
usage: run-aidm.sh --dataset PATH --run-dir DIR [--proposal PATH] [--legacy-predictions PATH] [--folds N] [--minimum-improvement X] [--max-plant-regression X] [--top-single-candidates N] [--seed N]
USAGE
}

dataset=""
run_dir=""
args=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset) [[ $# -ge 2 ]] || { echo "missing value for --dataset" >&2; exit 2; }; dataset="$2"; shift 2 ;;
    --run-dir) [[ $# -ge 2 ]] || { echo "missing value for --run-dir" >&2; exit 2; }; run_dir="$2"; shift 2 ;;
    --proposal|--legacy-predictions)
      [[ $# -ge 2 ]] || { echo "missing value for $1" >&2; exit 2; }
      [[ -f "$2" ]] || { echo "${1#--} not found: $2" >&2; exit 2; }
      args+=("$1" "$2")
      shift 2
      ;;
    --folds|--minimum-improvement|--max-plant-regression|--top-single-candidates|--seed)
      [[ $# -ge 2 ]] || { echo "missing value for $1" >&2; exit 2; }
      args+=("$1" "$2")
      shift 2
      ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown option: $1" >&2; usage; exit 2 ;;
  esac
done
[[ -n "$dataset" ]] || { echo "--dataset is required" >&2; usage; exit 2; }
[[ -f "$dataset" ]] || { echo "dataset not found: $dataset" >&2; exit 2; }
[[ -n "$run_dir" ]] || { echo "--run-dir is required" >&2; usage; exit 2; }

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
repo_root=$(cd "$script_dir/../.." && pwd -P)
source "$script_dir/validate-legacy-output-dir.sh"
validate_legacy_output_dir "$run_dir" "$repo_root"
run_dir="$VALIDATED_OUTPUT_DIR"
mkdir -p "$run_dir"
export PYTHONPATH="$repo_root/src${PYTHONPATH:+:$PYTHONPATH}"
cd "$repo_root"
uv run python -m power_forecasting.cli aidm --output "$run_dir" --dataset "$dataset" "${args[@]}"
