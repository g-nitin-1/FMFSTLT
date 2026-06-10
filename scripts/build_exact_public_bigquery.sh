#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROJECT_ID="${PROJECT_ID:-}"
DATASET="${DATASET:-fmfstlt}"
LOCATION="${LOCATION:-US}"
DRY_RUN="${DRY_RUN:-0}"
MAXIMUM_BYTES_BILLED="${MAXIMUM_BYTES_BILLED:-}"
REFERENCE_DATASET_PATTERN="ee-21cs01007\\.fmfstlt"

SQL_FILES=(
  "01_exact_ndt7_base_meta.sql"
  "02_exact_ndt7_splits.sql"
  "03a_exact_train_features.sql"
  "03b_exact_test_features.sql"
  "03c_exact_robustness_features.sql"
)

if [[ -z "$PROJECT_ID" ]]; then
  echo "PROJECT_ID is required; it is the Google Cloud project billed for the queries." >&2
  exit 1
fi

if [[ ! "$PROJECT_ID" =~ ^[a-z][a-z0-9-]{4,28}[a-z0-9]$ ]]; then
  echo "invalid Google Cloud project ID: $PROJECT_ID" >&2
  exit 1
fi

if [[ ! "$DATASET" =~ ^[A-Za-z0-9_]{1,1024}$ ]]; then
  echo "invalid BigQuery dataset ID: $DATASET" >&2
  exit 1
fi

if [[ "$DRY_RUN" != "0" && "$DRY_RUN" != "1" ]]; then
  echo "DRY_RUN must be 0 or 1" >&2
  exit 1
fi

if ! command -v bq >/dev/null 2>&1; then
  echo "bq is required; install and authenticate the Google Cloud CLI first." >&2
  exit 1
fi

if ! bq show --project_id="$PROJECT_ID" "${PROJECT_ID}:${DATASET}" >/dev/null 2>&1; then
  echo "create dataset ${PROJECT_ID}:${DATASET} in ${LOCATION}"
  bq mk --project_id="$PROJECT_ID" --location="$LOCATION" --dataset "${PROJECT_ID}:${DATASET}"
fi

QUERY_FLAGS=(
  "--project_id=${PROJECT_ID}"
  "--location=${LOCATION}"
  "--use_legacy_sql=false"
)
if [[ "$DRY_RUN" == "1" ]]; then
  QUERY_FLAGS+=("--dry_run")
fi
if [[ -n "$MAXIMUM_BYTES_BILLED" ]]; then
  QUERY_FLAGS+=("--maximum_bytes_billed=${MAXIMUM_BYTES_BILLED}")
fi

for sql_file in "${SQL_FILES[@]}"; do
  sql_path="${ROOT_DIR}/sql/${sql_file}"
  if [[ "$DRY_RUN" == "1" ]]; then
    echo "dry run ${sql_file} -> ${PROJECT_ID}.${DATASET}"
  else
    echo "run ${sql_file} -> ${PROJECT_ID}.${DATASET}"
  fi
  sed "s/${REFERENCE_DATASET_PATTERN}/${PROJECT_ID}.${DATASET}/g" "$sql_path" |
    bq query "${QUERY_FLAGS[@]}"
done
