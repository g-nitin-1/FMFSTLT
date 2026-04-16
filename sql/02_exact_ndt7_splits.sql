-- Deterministic train / test / robustness split generation for the exact ndt7 rebuild.
--
-- Train:
--   - first 10 sampled days
--   - 800k rows total
--   - balanced to 160k per speed tier
--
-- Test:
--   - first 10 sampled days
--   - 40k rows
--   - natural distribution
--
-- Robustness:
--   - last 2 sampled days
--   - 133k rows
--   - natural distribution

CREATE OR REPLACE TABLE `ee-21cs01007.fmfstlt.paper_exact_train_800k_meta` AS
WITH ordered_days AS (
  SELECT sample_date, ROW_NUMBER() OVER (ORDER BY sample_date) AS day_idx
  FROM UNNEST([
    DATE '2024-04-23',
    DATE '2024-05-30',
    DATE '2024-06-16',
    DATE '2024-07-28',
    DATE '2024-08-06',
    DATE '2024-09-26',
    DATE '2024-10-06',
    DATE '2024-11-16',
    DATE '2024-12-14',
    DATE '2025-01-03',
    DATE '2025-02-06',
    DATE '2025-03-04'
  ]) AS sample_date
),
train_pool AS (
  SELECT b.*
  FROM `ee-21cs01007.fmfstlt.paper_exact_base_meta` AS b
  JOIN ordered_days AS d
    ON b.date = d.sample_date
  WHERE d.day_idx <= 10
),
ranked AS (
  SELECT
    *,
    ROW_NUMBER() OVER (
      PARTITION BY speed_tier
      ORDER BY FARM_FINGERPRINT(CONCAT(uuid, '|paper_exact_train_v1'))
    ) AS rn
  FROM train_pool
)
SELECT
  date,
  uuid,
  test_time,
  mean_throughput_mbps,
  min_rtt_ms,
  loss_rate,
  country_code,
  continent_code,
  congestion_control,
  speed_tier
FROM ranked
WHERE rn <= 160000
;

CREATE OR REPLACE TABLE `ee-21cs01007.fmfstlt.paper_exact_test_40k_meta` AS
WITH ordered_days AS (
  SELECT sample_date, ROW_NUMBER() OVER (ORDER BY sample_date) AS day_idx
  FROM UNNEST([
    DATE '2024-04-23',
    DATE '2024-05-30',
    DATE '2024-06-16',
    DATE '2024-07-28',
    DATE '2024-08-06',
    DATE '2024-09-26',
    DATE '2024-10-06',
    DATE '2024-11-16',
    DATE '2024-12-14',
    DATE '2025-01-03',
    DATE '2025-02-06',
    DATE '2025-03-04'
  ]) AS sample_date
),
first_ten_days AS (
  SELECT sample_date
  FROM ordered_days
  WHERE day_idx <= 10
),
remaining AS (
  SELECT b.*
  FROM `ee-21cs01007.fmfstlt.paper_exact_base_meta` AS b
  JOIN first_ten_days AS d
    ON b.date = d.sample_date
  LEFT JOIN `ee-21cs01007.fmfstlt.paper_exact_train_800k_meta` AS t
    ON b.uuid = t.uuid
  WHERE t.uuid IS NULL
),
ranked AS (
  SELECT
    *,
    ROW_NUMBER() OVER (
      ORDER BY FARM_FINGERPRINT(CONCAT(uuid, '|paper_exact_test_v1'))
    ) AS rn
  FROM remaining
)
SELECT
  date,
  uuid,
  test_time,
  mean_throughput_mbps,
  min_rtt_ms,
  loss_rate,
  country_code,
  continent_code,
  congestion_control,
  speed_tier
FROM ranked
WHERE rn <= 40000
;

CREATE OR REPLACE TABLE `ee-21cs01007.fmfstlt.paper_exact_robustness_133k_meta` AS
WITH ordered_days AS (
  SELECT sample_date, ROW_NUMBER() OVER (ORDER BY sample_date) AS day_idx
  FROM UNNEST([
    DATE '2024-04-23',
    DATE '2024-05-30',
    DATE '2024-06-16',
    DATE '2024-07-28',
    DATE '2024-08-06',
    DATE '2024-09-26',
    DATE '2024-10-06',
    DATE '2024-11-16',
    DATE '2024-12-14',
    DATE '2025-01-03',
    DATE '2025-02-06',
    DATE '2025-03-04'
  ]) AS sample_date
),
robust_pool AS (
  SELECT b.*
  FROM `ee-21cs01007.fmfstlt.paper_exact_base_meta` AS b
  JOIN ordered_days AS d
    ON b.date = d.sample_date
  WHERE d.day_idx >= 11
),
ranked AS (
  SELECT
    *,
    ROW_NUMBER() OVER (
      ORDER BY FARM_FINGERPRINT(CONCAT(uuid, '|paper_exact_robust_v1'))
    ) AS rn
  FROM robust_pool
)
SELECT
  date,
  uuid,
  test_time,
  mean_throughput_mbps,
  min_rtt_ms,
  loss_rate,
  country_code,
  continent_code,
  congestion_control,
  speed_tier
FROM ranked
WHERE rn <= 133000
;

-- Recommended validation after creation:
--
-- SELECT 'train' AS split, speed_tier, COUNT(*) AS n
-- FROM `ee-21cs01007.fmfstlt.paper_exact_train_800k_meta`
-- GROUP BY split, speed_tier
-- UNION ALL
-- SELECT 'test' AS split, speed_tier, COUNT(*) AS n
-- FROM `ee-21cs01007.fmfstlt.paper_exact_test_40k_meta`
-- GROUP BY split, speed_tier
-- UNION ALL
-- SELECT 'robustness' AS split, speed_tier, COUNT(*) AS n
-- FROM `ee-21cs01007.fmfstlt.paper_exact_robustness_133k_meta`
-- GROUP BY split, speed_tier
-- ORDER BY split, speed_tier;
