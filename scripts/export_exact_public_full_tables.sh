#!/usr/bin/env bash
set -euo pipefail

QUERY_PROJECT_ID="${QUERY_PROJECT_ID:-${PROJECT_ID:-ee-21cs01007}}"
SOURCE_PROJECT_ID="${SOURCE_PROJECT_ID:-ee-21cs01007}"
DATASET="${DATASET:-fmfstlt}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EXPORT_ROOT="${EXPORT_ROOT:-$ROOT_DIR/exports_exact_public_full}"
SLEEP_SECONDS="${SLEEP_SECONDS:-1800}"
MAX_ATTEMPTS="${MAX_ATTEMPTS:-0}"

is_complete_csv() {
  local path="$1"
  if [[ ! -f "$path" ]]; then
    return 1
  fi
  local header
  header="$(head -n 1 "$path" 2>/dev/null || true)"
  [[ "$header" == "date,uuid,speed_tier,test_time,y_true_mbps,bucket_100ms,measurement_count,inst_throughput_mbps,cumavg_throughput_mbps,pipe_full_samples,mean_rtt_us,std_rtt_us,mean_snd_cwnd,std_snd_cwnd,mean_bytes_in_flight,std_bytes_in_flight,mean_total_retrans,std_total_retrans,mean_dsack_dups,std_dsack_dups" ]]
}

print_error_streams() {
  local tmp_out="$1"
  local tmp_err="$2"
  [[ -s "$tmp_err" ]] && cat "$tmp_err" >&2
  [[ -s "$tmp_out" ]] && sed -n '1,20p' "$tmp_out" >&2
}

has_quota_error() {
  local tmp_out="$1"
  local tmp_err="$2"
  grep -q 'Quota exceeded' "$tmp_err" 2>/dev/null || grep -q 'Quota exceeded' "$tmp_out" 2>/dev/null
}

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
  local out_file
  local tmp_out
  local tmp_err
  local attempt=1

  table="$(table_for_split "$split")"
  mkdir -p "$EXPORT_ROOT"
  out_file="${EXPORT_ROOT}/${split}_features_100ms_public_full.csv"
  tmp_out="${out_file}.tmp"
  tmp_err="${out_file}.err"

  if is_complete_csv "$out_file"; then
    echo "skip $split -> $out_file (already complete)"
    return 0
  fi

  rm -f "$tmp_out" "$tmp_err"

  while true; do
    echo "export $split -> $out_file (attempt $attempt)"
    if bq query \
      --project_id="$QUERY_PROJECT_ID" \
      --use_legacy_sql=false \
      --format=csv \
      "SELECT * FROM \`${SOURCE_PROJECT_ID}.${DATASET}.${table}\` ORDER BY date, speed_tier, uuid, bucket_100ms" \
      > "$tmp_out" 2> "$tmp_err"; then
      [[ -s "$tmp_err" ]] && cat "$tmp_err" >&2
      if is_complete_csv "$tmp_out"; then
        mv "$tmp_out" "$out_file"
        rm -f "$tmp_err"
        echo "completed $split -> $out_file"
        return 0
      fi
      echo "export for $split returned success but did not produce the expected CSV header" >&2
      [[ -f "$tmp_out" ]] && sed -n '1,10p' "$tmp_out" >&2
      return 1
    fi

    print_error_streams "$tmp_out" "$tmp_err"
    if has_quota_error "$tmp_out" "$tmp_err"; then
      rm -f "$tmp_out" "$tmp_err"
      if [[ "$MAX_ATTEMPTS" -gt 0 && "$attempt" -ge "$MAX_ATTEMPTS" ]]; then
        echo "quota still exceeded after $attempt attempts for $split" >&2
        return 1
      fi
      echo "quota exceeded for $split; sleeping ${SLEEP_SECONDS}s before retry" >&2
      sleep "$SLEEP_SECONDS"
      attempt=$((attempt + 1))
      continue
    fi

    return 1
  done
}

main() {
  export_split train
  export_split test
  export_split robustness
}

main "$@"
