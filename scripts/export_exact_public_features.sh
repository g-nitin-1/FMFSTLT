#!/usr/bin/env bash
set -euo pipefail

QUERY_PROJECT_ID="${QUERY_PROJECT_ID:-${PROJECT_ID:-ee-21cs01007}}"
SOURCE_PROJECT_ID="${SOURCE_PROJECT_ID:-ee-21cs01007}"
DATASET="${DATASET:-fmfstlt}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EXPORT_ROOT="${EXPORT_ROOT:-$ROOT_DIR/exports_exact_public}"
ROW_HEADROOM="${ROW_HEADROOM:-5000}"

TRAIN_TEST_DATES=(
  2024-04-23
  2024-05-30
  2024-06-16
  2024-07-28
  2024-08-06
  2024-09-26
  2024-10-06
  2024-11-16
  2024-12-14
  2025-01-03
)

ROBUSTNESS_DATES=(
  2025-02-06
  2025-03-04
)

TIERS=(
  "0-25"
  "25-100"
  "100-200"
  "200-400"
  "400+"
)

table_for_split() {
  case "$1" in
    train) echo "paper_exact_train_features_100ms_public" ;;
    test) echo "paper_exact_test_features_100ms_public" ;;
    robustness) echo "paper_exact_robustness_features_100ms_public" ;;
    *)
      echo "unknown split: $1" >&2
      return 1
      ;;
  esac
}

dates_for_split() {
  case "$1" in
    train|test)
      printf '%s\n' "${TRAIN_TEST_DATES[@]}"
      ;;
    robustness)
      printf '%s\n' "${ROBUSTNESS_DATES[@]}"
      ;;
    *)
      echo "unknown split: $1" >&2
      return 1
      ;;
  esac
}

tier_slug() {
  case "$1" in
    "0-25") echo "0_25" ;;
    "25-100") echo "25_100" ;;
    "100-200") echo "100_200" ;;
    "200-400") echo "200_400" ;;
    "400+") echo "400_plus" ;;
    *)
      echo "unknown tier: $1" >&2
      return 1
      ;;
  esac
}

count_rows() {
  local table="$1"
  local date="$2"
  local tier="$3"

  bq query \
    --project_id="$QUERY_PROJECT_ID" \
    --use_legacy_sql=false \
    --format=csv \
    "SELECT COUNT(*) AS row_count FROM \`${SOURCE_PROJECT_ID}.${DATASET}.${table}\` WHERE date = '${date}' AND speed_tier = '${tier}'" \
    | awk 'END {print $1}'
}

export_slice() {
  local split="$1"
  local table="$2"
  local date="$3"
  local tier="$4"
  local tier_file
  local row_count
  local expected_lines
  local actual_lines
  local max_rows
  local out_dir
  local out_file

  tier_file="$(tier_slug "$tier")"
  row_count="$(count_rows "$table" "$date" "$tier")"

  if [[ ! "$row_count" =~ ^[0-9]+$ ]]; then
    echo "failed to read row count for $split $date $tier: $row_count" >&2
    return 1
  fi

  if [[ "$row_count" -eq 0 ]]; then
    echo "skip $split $date $tier (0 rows)"
    return 0
  fi

  out_dir="${EXPORT_ROOT}/${split}"
  out_file="${out_dir}/${split}_${date//-/_}_features_100ms_${tier_file}.csv"
  mkdir -p "$out_dir"

  expected_lines=$((row_count + 1))
  if [[ -f "$out_file" ]]; then
    actual_lines="$(wc -l < "$out_file")"
    if [[ "$actual_lines" -eq "$expected_lines" ]]; then
      echo "skip $split $date $tier -> $out_file (already complete)"
      return 0
    fi
    echo "re-export $split $date $tier -> $out_file (found $actual_lines lines, expected $expected_lines)"
  fi

  max_rows=$((row_count + ROW_HEADROOM))
  echo "export $split $date $tier -> $out_file ($row_count rows)"
  bq query \
    --project_id="$QUERY_PROJECT_ID" \
    --max_rows="$max_rows" \
    --use_legacy_sql=false \
    --format=csv \
    "SELECT * FROM \`${SOURCE_PROJECT_ID}.${DATASET}.${table}\` WHERE date = '${date}' AND speed_tier = '${tier}' ORDER BY uuid, bucket_100ms" \
    > "$out_file"
}

export_split() {
  local split="$1"
  local table
  local date
  local tier

  table="$(table_for_split "$split")"

  while IFS= read -r date; do
    for tier in "${TIERS[@]}"; do
      export_slice "$split" "$table" "$date" "$tier"
    done
  done < <(dates_for_split "$split")
}

main() {
  export_split train
  export_split test
  export_split robustness
}

main "$@"
