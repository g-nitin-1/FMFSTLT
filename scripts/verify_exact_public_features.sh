#!/usr/bin/env bash
set -euo pipefail

QUERY_PROJECT_ID="${QUERY_PROJECT_ID:-${PROJECT_ID:-}}"
SOURCE_PROJECT_ID="${SOURCE_PROJECT_ID:-$QUERY_PROJECT_ID}"
DATASET="${DATASET:-fmfstlt}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EXPORT_ROOT="${EXPORT_ROOT:-$ROOT_DIR/exports_exact_public}"

if [[ -z "$QUERY_PROJECT_ID" ]]; then
  echo "set PROJECT_ID or QUERY_PROJECT_ID before verifying BigQuery exports" >&2
  exit 1
fi

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

verify_slice() {
  local split="$1"
  local table="$2"
  local date="$3"
  local tier="$4"
  local tier_file
  local row_count
  local expected_lines
  local out_file
  local actual_lines

  tier_file="$(tier_slug "$tier")"
  row_count="$(count_rows "$table" "$date" "$tier")"

  if [[ ! "$row_count" =~ ^[0-9]+$ ]]; then
    echo "failed to read row count for $split $date $tier: $row_count" >&2
    return 1
  fi

  if [[ "$row_count" -eq 0 ]]; then
    return 0
  fi

  out_file="${EXPORT_ROOT}/${split}/${split}_${date//-/_}_features_100ms_${tier_file}.csv"
  expected_lines=$((row_count + 1))

  if [[ ! -f "$out_file" ]]; then
    echo "missing $out_file" >&2
    return 1
  fi

  actual_lines="$(wc -l < "$out_file")"
  if [[ "$actual_lines" -ne "$expected_lines" ]]; then
    echo "mismatch $split $date $tier: expected $expected_lines got $actual_lines" >&2
    return 1
  fi

  echo "ok $split $date $tier ($actual_lines lines)"
}

verify_split() {
  local split="$1"
  local table
  local date
  local tier

  table="$(table_for_split "$split")"

  while IFS= read -r date; do
    for tier in "${TIERS[@]}"; do
      verify_slice "$split" "$table" "$date" "$tier"
    done
  done < <(dates_for_split "$split")
}

main() {
  verify_split train
  verify_split test
  verify_split robustness
}

main "$@"
