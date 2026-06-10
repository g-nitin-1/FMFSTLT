#!/usr/bin/env bash
set -euo pipefail

QUERY_PROJECT_ID="${QUERY_PROJECT_ID:-${PROJECT_ID:-}}"
SOURCE_PROJECT_ID="${SOURCE_PROJECT_ID:-$QUERY_PROJECT_ID}"
DATASET="${DATASET:-fmfstlt}"
GCS_BUCKET="${GCS_BUCKET:-}"
GCS_PREFIX="${GCS_PREFIX:-fmfstlt_exact_public}"

if [[ -z "$QUERY_PROJECT_ID" ]]; then
  echo "set PROJECT_ID or QUERY_PROJECT_ID before exporting BigQuery tables" >&2
  exit 1
fi

if [[ -z "$GCS_BUCKET" ]]; then
  echo "set GCS_BUCKET, for example: export GCS_BUCKET=my-bucket-name" >&2
  exit 1
fi

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

export_split() {
  local split="$1"
  local table
  local uri

  table="$(table_for_split "$split")"
  uri="gs://${GCS_BUCKET}/${GCS_PREFIX}/${split}_features_100ms_public_full-*.csv"

  echo "extract $split -> $uri"
  bq extract \
    --project_id="$QUERY_PROJECT_ID" \
    --destination_format=CSV \
    "${SOURCE_PROJECT_ID}:${DATASET}.${table}" \
    "$uri"
}

main() {
  export_split train
  export_split test
  export_split robustness
}

main "$@"
