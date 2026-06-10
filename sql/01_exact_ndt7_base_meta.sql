-- Paper-faithful base metadata for the public-data TURBOTEST reproduction.
--
-- This intentionally does NOT use measurement-lab.ndt.unified_downloads.
-- It uses the ndt7 extended view so that we can:
-- 1. force ndt7-only / BBR-only rows
-- 2. explicitly exclude M-Lab's 250 MB early-exit rows
-- 3. later rebuild throughput-derived per-window features from raw ndt7 measurements

CREATE OR REPLACE TABLE `ee-21cs01007.fmfstlt.paper_exact_base_meta` AS
SELECT
  t.date,
  t.id AS uuid,
  t.a.TestTime AS test_time,
  t.a.MeanThroughputMbps AS mean_throughput_mbps,
  t.a.MinRTT AS min_rtt_ms,
  t.a.LossRate AS loss_rate,
  t.a.CongestionControl AS congestion_control,
  client.Geo.CountryCode AS country_code,
  client.Geo.ContinentCode AS continent_code,
  CASE
    WHEN t.a.MeanThroughputMbps < 25 THEN '0-25'
    WHEN t.a.MeanThroughputMbps < 100 THEN '25-100'
    WHEN t.a.MeanThroughputMbps < 200 THEN '100-200'
    WHEN t.a.MeanThroughputMbps < 400 THEN '200-400'
    ELSE '400+'
  END AS speed_tier
FROM `measurement-lab.ndt_intermediate.extended_ndt7_downloads` AS t
WHERE client.Geo.CountryCode = 'US'
  AND t.date IN (
    DATE '2024-04-23', DATE '2024-05-30', DATE '2024-06-16',
    DATE '2024-07-28', DATE '2024-08-06', DATE '2024-09-26',
    DATE '2024-10-06', DATE '2024-11-16', DATE '2024-12-14',
    DATE '2025-01-03', DATE '2025-02-06', DATE '2025-03-04'
  )
  AND t.id IS NOT NULL
  AND t.a.MeanThroughputMbps IS NOT NULL
  AND LOWER(t.a.CongestionControl) = 'bbr'
  AND filter.IsComplete
  AND filter.IsProduction
  AND NOT filter.IsError
  AND NOT filter.IsOAM
  AND NOT filter.IsPlatformAnomaly
  AND NOT filter.IsSmall
  AND NOT filter.IsShort
  AND NOT filter.IsLong
  AND NOT filter._IsRFC1918
  AND NOT filter.IsEarlyExit
;

-- Recommended validation after creation:
--
-- SELECT congestion_control, COUNT(*) AS n
-- FROM `ee-21cs01007.fmfstlt.paper_exact_base_meta`
-- GROUP BY congestion_control;
--
-- SELECT date, COUNT(*) AS n
-- FROM `ee-21cs01007.fmfstlt.paper_exact_base_meta`
-- GROUP BY date
-- ORDER BY date;
