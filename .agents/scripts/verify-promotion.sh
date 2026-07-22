#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat >&2 <<'USAGE'
usage: verify-promotion.sh --run-dir DIR
USAGE
}

run_dir=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --run-dir) [[ $# -ge 2 ]] || { echo "missing value for --run-dir" >&2; exit 2; }; run_dir="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown option: $1" >&2; usage; exit 2 ;;
  esac
done
[[ -n "$run_dir" ]] || { echo "--run-dir is required" >&2; usage; exit 2; }
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
repo_root=$(cd "$script_dir/../.." && pwd -P)
source "$script_dir/validate-legacy-output-dir.sh"
validate_legacy_output_dir "$run_dir" "$repo_root"
run_dir="$VALIDATED_OUTPUT_DIR"
manifest="$run_dir/promotion_manifest.json"
evidence="$run_dir/promotion-evidence.json"
generated="$run_dir/generated/promoted_features.py"
patch="$run_dir/model-recipe-patch.json"
[[ -f "$manifest" ]] || { echo "promotion manifest not found: $manifest" >&2; exit 2; }
rm -f "$evidence" "$generated" "$patch"

export PYTHONPATH="$repo_root/src${PYTHONPATH:+:$PYTHONPATH}"
cd "$repo_root"
uv run python - <<'PY' "$manifest"
import json, sys
from pathlib import Path
payload = json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'))
if payload.get('decision') != 'promote':
    raise SystemExit('promotion manifest decision must be promote')
PY
if ! uv run python -m power_forecasting.cli aidd --output "$run_dir" --manifest "$manifest"; then
  rm -f "$generated" "$evidence" "$patch"
  exit 2
fi
if ! uv run python -m py_compile "$generated"; then
  rm -f "$generated" "$evidence" "$patch"
  exit 2
fi
if ! uv run python - <<'PY' "$manifest" "$patch"
import json, sys
from pathlib import Path
from power_forecasting.aidd import render_model_recipe_patch
payload = json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'))
if payload.get('selected_model_recipe') is not None:
    render_model_recipe_patch(payload, Path(sys.argv[2]))
PY
then
  rm -f "$generated" "$evidence" "$patch"
  exit 2
fi
if ! uv run python - <<'PY' "$manifest" "$generated" "$patch" "$evidence"
import hashlib, json, sys
from pathlib import Path

def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()
manifest = Path(sys.argv[1])
generated = Path(sys.argv[2])
patch = Path(sys.argv[3])
evidence = Path(sys.argv[4])
payload = {
    'schema_version': '1',
    'status': 'success',
    'manifest_sha256': sha(manifest),
    'generated_module_sha256': sha(generated),
    'generated_module': 'generated/promoted_features.py',
}
if patch.exists():
    payload['model_recipe_patch_sha256'] = sha(patch)
    payload['model_recipe_patch'] = 'model-recipe-patch.json'
evidence.write_text(json.dumps(payload, sort_keys=True, indent=2) + '\n', encoding='utf-8')
print(evidence)
PY
then
  rm -f "$generated" "$evidence" "$patch"
  exit 2
fi
