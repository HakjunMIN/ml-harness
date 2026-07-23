#!/usr/bin/env bash

_canonical_legacy_path() {
  local absolute="$1"
  local component current="/"
  local path_without_root="${absolute#/}"
  local IFS=/
  local -a components

  if [[ -z "$path_without_root" ]]; then
    LEGACY_OUTPUT_PATH="/"
    return 0
  fi
  read -r -a components <<< "$path_without_root"
  for component in "${components[@]}"; do
    case "$component" in
      ""|"." )
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

  LEGACY_OUTPUT_PATH="$current"
}

validate_legacy_output_dir() {
  local supplied="$1"
  local repository_root="$2"
  local absolute canonical

  if [[ -z "$supplied" ]]; then
    echo "run-dir must be nonblank" >&2
    return 2
  fi

  if [[ "$supplied" == /* ]]; then
    absolute="$supplied"
  else
    absolute="$PWD/$supplied"
  fi

  # Legacy tools historically accepted any local output directory. Keep that
  # contract while rejecting destinations that could overwrite repository code
  # or the repository itself.
  case "$absolute" in
    "$repository_root"|"$repository_root"/.git|"$repository_root"/.git/*|\
    "$repository_root"/src|"$repository_root"/src/*|\
    "$repository_root"/tests|"$repository_root"/tests/*|\
    "$repository_root"/docs|"$repository_root"/docs/*|\
    "$repository_root"/.agents)
      echo "run-dir targets protected repository content: $supplied" >&2
      return 2
      ;;
  esac

  _canonical_legacy_path "$absolute" || return 2
  canonical="$LEGACY_OUTPUT_PATH"
  if [[ "$canonical" == "/" ]]; then
    echo "run-dir must not be the filesystem root: $supplied" >&2
    return 2
  fi
  if [[ "$canonical" == "$repository_root" || "$canonical" == "$repository_root"/.git || \
        "$canonical" == "$repository_root"/.git/* || "$canonical" == "$repository_root"/src || \
        "$canonical" == "$repository_root"/src/* || "$canonical" == "$repository_root"/tests || \
        "$canonical" == "$repository_root"/tests/* || "$canonical" == "$repository_root"/docs || \
        "$canonical" == "$repository_root"/docs/* || "$canonical" == "$repository_root"/.agents || \
        "$canonical" == "$repository_root"/.agents/* ]]; then
    echo "run-dir targets protected repository content: $supplied" >&2
    return 2
  fi

  VALIDATED_OUTPUT_DIR="$canonical"
}
