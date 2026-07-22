#!/usr/bin/env bash

_canonical_non_symlink_path() {
  local absolute="$1"
  local component current="/"
  local path_without_root="${absolute#/}"
  local IFS=/
  local -a components
  read -r -a components <<< "$path_without_root"
  for component in "${components[@]}"; do
    case "$component" in
      ""|".")
        continue
        ;;
      "..")
        current="${current%/*}"
        [[ -n "$current" ]] || current="/"
        ;;
      *)
        current="${current%/}/$component"
        if [[ -L "$current" ]]; then
          echo "run-dir contains symlinked path component: $current" >&2
          return 2
        fi
        ;;
    esac
  done

  CANONICAL_PATH="$current"
}

validate_output_dir() {
  local supplied="$1"
  local repository_root="$2"
  local absolute canonical allowed

  if [[ -z "$supplied" ]]; then
    echo "run-dir must be nonblank" >&2
    return 2
  fi

  if [[ "$supplied" == /* ]]; then
    absolute="$supplied"
  else
    absolute="$PWD/$supplied"
  fi
  _canonical_non_symlink_path "$absolute" || return 2
  canonical="$CANONICAL_PATH"

  for allowed in "$repository_root/.agents/runs" "$repository_root/.agents/output"; do
    _canonical_non_symlink_path "$allowed" || return 2
    if [[ "$canonical" == "$CANONICAL_PATH" || "$canonical" == "$CANONICAL_PATH"/* ]]; then
      VALIDATED_OUTPUT_DIR="$canonical"
      return 0
    fi
  done

  echo "run-dir must be contained in repository .agents/runs or .agents/output: $supplied" >&2
  return 2
}
