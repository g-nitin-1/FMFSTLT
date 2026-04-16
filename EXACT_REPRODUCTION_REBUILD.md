# Exact Reproduction Rebuild

This note defines the rebuild required for a paper-faithful public-data reproduction of TURBOTEST.

## Why The Current `exports/` Corpus Is Not Final

The existing CSV exports are a strong intermediate reproduction, but they are not exact enough for the paper implementation:

- they were built from `measurement-lab.ndt.unified_downloads`, not directly from `measurement-lab.ndt_intermediate.extended_ndt7_downloads`
- the current meta tables still include `cubic` rows, so the corpus is not ndt7/BBR-pure
- `measurement-lab.ndt.unified_downloads` defines `IsValidBest` in a way that still keeps `filter.IsEarlyExit = TRUE` rows
- the current bucket CSVs do not include the paper's throughput features
- the current bucket CSVs do not include the paper's BBR pipe-full feature

## Confirmed Public-Data Facts

These points are now confirmed from the public M-Lab schema and the public `m-lab/ndt-server` source:

- ndt7 early exit accepts only `early_exit=250`
- `early_exit=250` is converted to `250 * 1,000,000` bytes
- ndt7 download early exit stops once `TCPInfo.BytesAcked >= MaxBytes`
- final download throughput is computed from `BytesAcked / ElapsedTime`
- `extended_ndt7_downloads` exposes:
  - `filter.IsEarlyExit`
  - `a.MeanThroughputMbps`
  - raw per-measurement `TCPInfo`
  - raw per-measurement `BBRInfo`

## Exact Rebuild Rules

For the exact public-data reproduction, rebuild from:

- `measurement-lab.ndt_intermediate.extended_ndt7_downloads`

Apply these filters:

- sampled dates: the 12 monthly dates already used in this repo
- `client.Geo.CountryCode = 'US'`
- `LOWER(a.CongestionControl) = 'bbr'`
- `filter.IsComplete`
- `filter.IsProduction`
- `NOT filter.IsError`
- `NOT filter.IsOAM`
- `NOT filter.IsPlatformAnomaly`
- `NOT filter.IsSmall`
- `NOT filter.IsShort`
- `NOT filter.IsLong`
- `NOT filter._IsRFC1918`
- `NOT filter.IsEarlyExit`

## Feature Set We Can Rebuild Now

The paper states `13` features per `100 ms` interval:

- `2` throughput features
- `1` BBR pipe-full feature
- `5` tcp_info metrics, each summarized by mean and std

Using public ndt7 data, we can rebuild these now:

- instantaneous throughput from byte deltas
- cumulative average throughput from `BytesAcked / ElapsedTime`
- RTT mean/std
- congestion window mean/std
- bytes-in-flight mean/std
- retransmissions mean/std
- duplicate ACKs mean/std

The author clarified the public-data pipe-full reconstruction at a high level:

- use public BBR values over an interval
- treat pipe-full as reached after `3` consecutive signals

To make that reproducible from public ndt7 measurements, the SQL in this repo uses the standard BBR startup growth rule:

- public signal source: `BBRInfo.BW`
- substantial-growth reset: `BW >= 1.25 * prior_max_BW`
- pipe-full proxy: after `3` consecutive non-growth signals
- bucket feature: `pipe_full_samples = SUM(pipe_full_signal)` inside each `100 ms` bucket

That means the feature is no longer left as `NULL`. If you already built the exact feature tables before this clarification, regenerate them.

## Execution Order

1. Run [01_exact_ndt7_base_meta.sql](/mnt/e/FMFSTLT/sql/01_exact_ndt7_base_meta.sql)
2. Run [02_exact_ndt7_splits.sql](/mnt/e/FMFSTLT/sql/02_exact_ndt7_splits.sql)
3. Run [03_exact_ndt7_features_template.sql](/mnt/e/FMFSTLT/sql/03_exact_ndt7_features_template.sql) three times:
   - once for train
   - once for test
   - once for robustness
4. Export the new feature tables, not the older `tcpinfo_100ms` tables

## What Changes If The Author Replies Later

The second author clarification also confirms that source selection stayed within public BigQuery, followed by post-processing and bucketing. That is consistent with the current metadata rebuild path in this repo.

That is the reason this rebuild is structured as:

- metadata selection
- split generation
- feature generation

rather than as a one-off manual CSV export workflow.
